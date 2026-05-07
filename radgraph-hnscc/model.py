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
        ↓  LayerNorm + Dropout
  GAT Layer 2         → hidden_dim × n_heads
        ↓  LayerNorm + Dropout
  Read-out             GTV node features (index 0 from each graph)
        ↓
  Concat clinical features  (optional)
        ↓
  FC layers            → 1 logit → Sigmoid → P(recurrence)

Normalisation note:
  GATBlock uses torch_geometric.nn.LayerNorm with the batch vector passed in,
  so normalisation is done per-graph (not across all graphs in the batch).
  This matches Joseph Bae's GAT_modeling.py implementation.

  input_proj and classifier use standard nn.LayerNorm (no graph structure
  needed at those points).

Usage:
    from model import RadGraphGAT
    model = RadGraphGAT(node_feature_dim=93, n_clinical_features=8)
    logits = model(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm

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
    One GAT block = GATConv + LayerNorm (per-graph) + ELU + Dropout.

    Uses torch_geometric.nn.LayerNorm with the batch vector so that
    normalisation is computed per-graph, not across all nodes in the batch.
    This matches Joseph Bae's GAT_modeling.py:
        x = gat_layer(x, batch)   ← batch vector passed to LayerNorm
    """

    def __init__(self, in_channels, out_channels, n_heads,
                 concat=True, dropout=0.3, negative_slope=0.2):
        super().__init__()
        self.conv = GATConv(
            in_channels    = in_channels,
            out_channels   = out_channels,
            heads          = n_heads,
            concat         = concat,
            dropout        = dropout,
            negative_slope = negative_slope,
            add_self_loops = True,
            edge_dim       = 1,
        )
        out_dim         = out_channels * n_heads if concat else out_channels
        self.bn         = LayerNorm(out_dim)   # PyG LayerNorm — needs batch vector
        self.dropout    = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Parameters
        ----------
        x          : Tensor (N, F)
        edge_index : Tensor (2, E)
        batch      : Tensor (N,) or None
            Graph assignment vector. When provided, LayerNorm normalises
            each graph independently. When None, normalises over all nodes.
        edge_attr  : Tensor (E, 1) or None
        """
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.bn(x, batch)     # per-graph normalisation (Joseph's approach)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# ─── Main Model ───────────────────────────────────────────────────────────────

class RadGraphGAT(nn.Module):
    """
    Graph Attention Network for locoregional recurrence prediction.

    Parameters
    ----------
    node_feature_dim    : int
    n_clinical_features : int
    hidden_dim          : int
    n_heads             : int
    n_layers            : int
    dropout             : float
    negative_slope      : float
    use_clinical        : bool
    task                : 'LR' or 'DM'  — auto-loads Table S2 defaults
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
        task                = 'LR',
    ):
        super().__init__()

        gat_cfg = config.get_gat_config(task)

        self.node_feature_dim    = config.N_FEATURES_TOTAL    if node_feature_dim    is None else node_feature_dim
        self.n_clinical_features = config.N_CLINICAL_FEATURES if n_clinical_features is None else n_clinical_features
        self.hidden_dim          = gat_cfg['hidden_dim']       if hidden_dim          is None else hidden_dim
        self.n_heads             = gat_cfg['n_heads']          if n_heads             is None else n_heads
        self.n_layers            = gat_cfg['n_layers']         if n_layers            is None else n_layers
        self.dropout             = gat_cfg['dropout']          if dropout             is None else dropout
        self.neg_slope           = config.GAT_NEGATIVE_SLOPE   if negative_slope      is None else negative_slope
        self.use_clinical        = use_clinical

        # ── Input projection ────────────────────────────────────────────────
        # Standard nn.LayerNorm here — no graph structure at this point
        self.input_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
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
                    in_channels    = in_ch,
                    out_channels   = self.hidden_dim,
                    n_heads        = 1 if is_last else self.n_heads,
                    concat         = False if is_last else True,
                    dropout        = self.dropout,
                    negative_slope = self.neg_slope
                )
            )

        gtv_out_dim = self.hidden_dim   # last layer: concat=False → hidden_dim

        # ── Clinical fusion ──────────────────────────────────────────────────
        # BatchNorm1d is correct here — clinical features are flat (B, C),
        # not a graph, so standard batch normalisation applies
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
        # Standard nn.LayerNorm — output is (B, fc_in_dim), not a graph
        if config.USE_FC_HIDDEN:
            self.classifier = nn.Sequential(
                nn.Linear(fc_in_dim, config.FC_HIDDEN_DIM),
                nn.LayerNorm(config.FC_HIDDEN_DIM),
                nn.ELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(config.FC_HIDDEN_DIM, 1)
            )
        else:
            self.classifier = nn.Linear(fc_in_dim, 1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"RadGraphGAT initialised:")
        print(f"  Node features   : {self.node_feature_dim}")
        print(f"  Hidden dim      : {self.hidden_dim}")
        print(f"  Attention heads : {self.n_heads}")
        print(f"  GAT layers      : {self.n_layers}")
        print(f"  Norm in GAT     : LayerNorm (per-graph, batch vector passed)")
        print(f"  Clinical feats  : {self.n_clinical_features} (used={use_clinical})")
        print(f"  Trainable params: {n_params:,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : torch_geometric.data.Batch

        Returns
        -------
        logits : Tensor (B,)
        """
        x          = batch.x
        edge_index = batch.edge_index
        edge_attr  = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        batch_vec  = batch.batch

        # 1. Input projection
        x = self.input_proj(x)

        # 2. GAT layers — batch_vec passed so LayerNorm normalises per-graph
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, batch_vec, edge_attr)

        # 3. GTV node read-out (index 0 of each graph)
        gtv_features = self._extract_gtv_nodes(x, batch_vec)   # (B, hidden_dim)

        # 4. Clinical feature fusion
        if self.clinical_proj is not None and hasattr(batch, 'clinical'):
            clinical_emb = self.clinical_proj(batch.clinical)   # (B, 16)
            combined     = torch.cat([gtv_features, clinical_emb], dim=1)
        else:
            combined = gtv_features

        # 5. Classification
        logits = self.classifier(combined).squeeze(-1)          # (B,)
        return logits

    def predict_proba(self, batch):
        """Return sigmoid probabilities instead of raw logits."""
        return torch.sigmoid(self.forward(batch))

    def get_attention_weights(self, batch):
        """
        Extract attention weights from the FINAL GAT layer.

        Paper: "attention values from the GTV readout node to all other
                graph nodes were extracted from the final GAT layer."

        batch_vec is passed to non-final layers so LayerNorm behaves
        identically to training (per-graph normalisation).

        Returns
        -------
        alpha      : Tensor (E_total, n_heads)
        edge_index : Tensor (2, E_total)
        """
        x          = batch.x
        edge_index = batch.edge_index
        edge_attr  = getattr(batch, 'edge_attr', None)
        batch_vec  = batch.batch   # needed for per-graph LayerNorm

        x = self.input_proj(x)

        alpha       = None
        edge_index_ = None

        for layer_idx, gat_layer in enumerate(self.gat_layers):
            is_final = (layer_idx == len(self.gat_layers) - 1)

            if is_final:
                # Capture attention from final layer
                _, (edge_index_, alpha) = gat_layer.conv(
                    x, edge_index,
                    edge_attr                = edge_attr,
                    return_attention_weights = True
                )
                # Complete the forward pass for the final layer
                x = gat_layer.conv(x, edge_index, edge_attr=edge_attr)
                x = gat_layer.bn(x, batch_vec)
                x = gat_layer.activation(x)
                x = gat_layer.dropout(x)
            else:
                # Pass batch_vec so LayerNorm behaviour matches training
                x = gat_layer(x, edge_index, batch_vec, edge_attr)

        return alpha, edge_index_

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_gtv_nodes(x, batch_vec):
        """
        Extract feature vector of node 0 (GTV hub) from each graph in batch.
        Node 0 is always the GTV — set in graph_builder.py.
        """
        batch_size  = batch_vec.max().item() + 1
        gtv_indices = []
        for g in range(batch_size):
            node_indices = (batch_vec == g).nonzero(as_tuple=True)[0]
            gtv_indices.append(node_indices[0])
        gtv_indices = torch.stack(gtv_indices)
        return x[gtv_indices]


# ─── Loss function factory ────────────────────────────────────────────────────

def get_loss_function(pos_weight=None):
    """
    Returns loss function based on config.LOSS_FUNCTION.
    'Focal' → FocalLoss,  else → BCEWithLogitsLoss.
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

    N_NODES = 21   # 1 GTV + 20 supervoxels
    N_FEATS = 93
    N_CLIN  = 8
    B       = 4

    graphs = []
    for i in range(B):
        src = [0]*(N_NODES-1) + list(range(1, N_NODES))
        dst = list(range(1, N_NODES)) + [0]*(N_NODES-1)
        g = Data(
            x          = torch.randn(N_NODES, N_FEATS),
            edge_index = torch.tensor([src, dst], dtype=torch.long),
            edge_attr  = torch.rand(len(src), 1),
            y          = torch.tensor([i % 2], dtype=torch.long),
            clinical   = torch.randn(N_CLIN),
        )
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    model = RadGraphGAT(node_feature_dim=N_FEATS, n_clinical_features=N_CLIN)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        probs  = torch.sigmoid(logits)

    print(f"\nLogits : {logits}")
    print(f"Probs  : {probs}")
    print(f"Shape  : {logits.shape}  (expected: ({B},))")

    criterion = get_loss_function(pos_weight=3.0)
    loss      = criterion(logits, batch.y.float().squeeze())
    print(f"Loss   : {loss.item():.4f}")
    print("\nModel test passed!")