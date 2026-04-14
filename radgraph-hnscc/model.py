"""
GAT Model for RadGraph
======================
Graph Attention Network with GTV-focused read-out.

Architecture:
  Input node features  (F)
        ↓
  Linear projection   → hidden_dim
        ↓
  GAT Layer 1         → hidden_dim × n_heads  (with multi-head attention)
        ↓  BatchNorm + Dropout
  GAT Layer 2         → hidden_dim × n_heads
        ↓  BatchNorm + Dropout
  Read-out             GTV node features (index 0 from each graph)
        ↓
  Concat clinical features  (optional)
        ↓
  FC layers            → 1 logit → Sigmoid → P(recurrence)

Usage:
    from model import RadGraphGAT
    model = RadGraphGAT(node_feature_dim=93, n_clinical_features=8)
    logits = model(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

import config


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits  : Tensor (N,)  raw logits
        targets : Tensor (N,)  binary labels  {0, 1}
        """
        targets = targets.float()
        probs   = torch.sigmoid(logits)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ─── GAT Layer Wrapper ────────────────────────────────────────────────────────

class GATBlock(nn.Module):
    """
    One GAT block = GATConv + BatchNorm + ELU + Dropout
    """

    def __init__(self, in_channels, out_channels, n_heads,
                 concat=True, dropout=0.3, negative_slope=0.2):
        super().__init__()
        self.conv   = GATConv(
            in_channels   = in_channels,
            out_channels  = out_channels,
            heads         = n_heads,
            concat        = concat,
            dropout       = dropout,
            negative_slope= negative_slope,
            add_self_loops= True,
            edge_dim      = 1,     # edge_attr dimension
        )
        out_dim         = out_channels * n_heads if concat else out_channels
        self.bn         = BatchNorm(out_dim)
        self.dropout    = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# ─── Main Model ───────────────────────────────────────────────────────────────

class RadGraphGAT(nn.Module):
    """
    Graph Attention Network for locoregional recurrence prediction.

    Parameters
    ----------
    node_feature_dim    : int   number of radiomic features per node
    n_clinical_features : int   number of clinical covariates
    hidden_dim          : int   hidden dimension per attention head
    n_heads             : int   number of attention heads
    n_layers            : int   number of GAT layers
    dropout             : float dropout probability
    negative_slope      : float LeakyReLU slope for attention
    use_clinical        : bool  whether to fuse clinical features
    """

    def __init__(
        self,
        node_feature_dim    = None,
        n_clinical_features = None,
        hidden_dim          = None,
        n_heads             = None,
        n_layers            = None,
        dropout             = None,
        negative_slope      = None,
        use_clinical        = True,
        task                = 'LR',   # used to auto-load Table S2 defaults
    ):
        super().__init__()

        # Load Table S2 defaults for this task, then allow overrides
        gat_cfg = config.get_gat_config(task)

        self.node_feature_dim    = config.N_FEATURES_TOTAL        if node_feature_dim    is None else node_feature_dim
        self.n_clinical_features = config.N_CLINICAL_FEATURES     if n_clinical_features is None else n_clinical_features
        self.hidden_dim          = gat_cfg['hidden_dim']           if hidden_dim          is None else hidden_dim
        self.n_heads             = gat_cfg['n_heads']              if n_heads             is None else n_heads
        self.n_layers            = gat_cfg['n_layers']             if n_layers            is None else n_layers
        self.dropout             = gat_cfg['dropout']              if dropout             is None else dropout
        self.neg_slope           = config.GAT_NEGATIVE_SLOPE       if negative_slope      is None else negative_slope
        self.use_clinical        = use_clinical

        # ── Input projection ────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        # ── GAT Layers ───────────────────────────────────────────────────────
        self.gat_layers = nn.ModuleList()

        for layer_idx in range(self.n_layers):
            is_last = (layer_idx == self.n_layers - 1)
            in_ch   = self.hidden_dim if layer_idx == 0 else \
                      self.hidden_dim * self.n_heads

            self.gat_layers.append(
                GATBlock(
                    in_channels   = in_ch,
                    out_channels  = self.hidden_dim,
                    n_heads       = 1 if is_last else self.n_heads,
                    concat        = False if is_last else True,
                    dropout       = self.dropout,
                    negative_slope= self.neg_slope
                )
            )

        # GTV node output dimension  (last layer: concat=False → hidden_dim)
        gtv_out_dim = self.hidden_dim

        # ── Clinical fusion ──────────────────────────────────────────────────
        if self.use_clinical and self.n_clinical_features > 0:
            self.clinical_proj = nn.Sequential(
                nn.Linear(self.n_clinical_features, 16),
                nn.BatchNorm1d(16),
                nn.ELU(),
                nn.Dropout(p=self.dropout)
            )
            fc_in_dim = gtv_out_dim + 16
        else:
            self.clinical_proj = None
            fc_in_dim          = gtv_out_dim

        # ── Classifier ──────────────────────────────────────────────────────
        if config.USE_FC_HIDDEN:
            self.classifier = nn.Sequential(
                nn.Linear(fc_in_dim, config.FC_HIDDEN_DIM),
                nn.BatchNorm1d(config.FC_HIDDEN_DIM),
                nn.ELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(config.FC_HIDDEN_DIM, 1)
            )
        else:
            self.classifier = nn.Linear(fc_in_dim, 1)

        # Parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"RadGraphGAT initialised:")
        print(f"  Node features   : {self.node_feature_dim}")
        print(f"  Hidden dim      : {self.hidden_dim}")
        print(f"  Attention heads : {self.n_heads}")
        print(f"  GAT layers      : {self.n_layers}")
        print(f"  Clinical feats  : {self.n_clinical_features} (used={use_clinical})")
        print(f"  Trainable params: {n_params:,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched graphs with attributes:
              x          : (N_total, F)
              edge_index : (2, E_total)
              edge_attr  : (E_total, 1)
              batch      : (N_total,)  — graph assignment vector
              clinical   : (B, n_clinical)  optional

        Returns
        -------
        logits : Tensor (B,)  — one logit per graph
        """
        x          = batch.x
        edge_index = batch.edge_index
        edge_attr  = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        batch_vec  = batch.batch

        # 1. Input projection
        x = self.input_proj(x)

        # 2. GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)

        # 3. Read-out: extract GTV node (index 0) from each graph
        gtv_features = self._extract_gtv_nodes(x, batch_vec)  # (B, hidden_dim)

        # 4. Clinical feature fusion
        if self.clinical_proj is not None and hasattr(batch, 'clinical'):
            clinical_emb  = self.clinical_proj(batch.clinical)  # (B, 16)
            combined      = torch.cat([gtv_features, clinical_emb], dim=1)
        else:
            combined = gtv_features

        # 5. Classification
        logits = self.classifier(combined).squeeze(-1)   # (B,)

        return logits

    def predict_proba(self, batch):
        """Return probabilities (0–1) instead of raw logits."""
        logits = self.forward(batch)
        return torch.sigmoid(logits)

    def get_attention_weights(self, batch):
        """
        Extract attention weights from the FINAL GAT layer.

        Paper (Methods — Graph attention atlas creation):
          "attention values from the GTV readout node to all other graph
           nodes were extracted from the final GAT layer of each model."

        Returns
        -------
        alpha       : Tensor  (E_total, n_heads)
        edge_index  : Tensor  (2, E_total)
        """
        x          = batch.x
        edge_index = batch.edge_index
        edge_attr  = getattr(batch, 'edge_attr', None)

        # Forward through all layers, capture attention at the final one
        x = self.input_proj(x)

        alpha      = None
        edge_index_ = None

        for layer_idx, gat_layer in enumerate(self.gat_layers):
            is_final = (layer_idx == len(self.gat_layers) - 1)

            if is_final:
                # Capture attention weights from the final layer only
                _, (edge_index_, alpha) = gat_layer.conv(
                    x, edge_index,
                    edge_attr             = edge_attr,
                    return_attention_weights = True
                )
                # Complete the final layer forward pass
                x = gat_layer.conv(x, edge_index, edge_attr=edge_attr)
                x = gat_layer.bn(x)
                x = gat_layer.activation(x)
                x = gat_layer.dropout(x)
            else:
                x = gat_layer(x, edge_index, edge_attr)

        return alpha, edge_index_

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_gtv_nodes(x, batch_vec):
        """
        Extract the feature vector of node 0 (GTV) from each graph in the batch.

        In a PyG batch, node indices are offset per graph. The GTV node is always
        the first node of each graph. We find these indices using the batch vector.

        Parameters
        ----------
        x         : Tensor (N_total, F)
        batch_vec : Tensor (N_total,)

        Returns
        -------
        gtv_feats : Tensor (B, F)
        """
        # For each graph g in the batch, find the index of its first node
        # (i.e., where batch_vec transitions from g-1 to g)
        batch_size  = batch_vec.max().item() + 1
        gtv_indices = []

        for g in range(batch_size):
            node_indices = (batch_vec == g).nonzero(as_tuple=True)[0]
            gtv_indices.append(node_indices[0])   # First node = GTV

        gtv_indices = torch.stack(gtv_indices)    # (B,)
        return x[gtv_indices]                      # (B, F)


# ─── Loss function factory ────────────────────────────────────────────────────

def get_loss_function(pos_weight=None):
    """
    Returns the appropriate loss function based on config.

    Parameters
    ----------
    pos_weight : float or None  — weight for positive class (for BCE)

    Returns
    -------
    criterion : nn.Module
    """
    if config.LOSS_FUNCTION == 'Focal':
        print(f"Using Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    else:
        if pos_weight is not None and config.HANDLE_IMBALANCE:
            weight = torch.tensor([pos_weight], dtype=torch.float)
            print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            print("Using BCEWithLogitsLoss (no weighting)")
            return nn.BCEWithLogitsLoss()


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    print("Testing RadGraphGAT with synthetic batch...")
    torch.manual_seed(42)

    N_NODES_PER_GRAPH = 21   # 1 GTV + 20 supervoxels
    N_FEATS           = 93
    N_CLINICAL        = 8
    BATCH_SIZE        = 4

    # Build a synthetic batch
    graphs = []
    for i in range(BATCH_SIZE):
        n = N_NODES_PER_GRAPH
        # Star edges
        src = [0]*(n-1) + list(range(1, n))
        dst = list(range(1, n)) + [0]*(n-1)
        g = Data(
            x          = torch.randn(n, N_FEATS),
            edge_index = torch.tensor([src, dst], dtype=torch.long),
            edge_attr  = torch.rand(len(src), 1),
            y          = torch.tensor([i % 2], dtype=torch.long),
            clinical   = torch.randn(N_CLINICAL),
        )
        graphs.append(g)

    batch = Batch.from_data_list(graphs)

    # Initialise model
    model = RadGraphGAT(
        node_feature_dim    = N_FEATS,
        n_clinical_features = N_CLINICAL,
    )
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        probs  = torch.sigmoid(logits)

    print(f"\nOutput logits : {logits}")
    print(f"Probabilities : {probs}")
    print(f"Output shape  : {logits.shape}  (expected: ({BATCH_SIZE},))")

    # Test loss
    criterion = get_loss_function(pos_weight=3.0)
    labels    = batch.y.float().squeeze()
    loss      = criterion(logits, labels)
    print(f"Loss          : {loss.item():.4f}")

    print("\nModel test passed!")
