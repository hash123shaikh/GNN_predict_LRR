"""
Graph Builder for RadGraph Implementation
==========================================
Builds patient-level graphs from supervoxel features.

Graph topology (Star / Hub-and-Spoke):
  - Node 0        : GTV node  (hub)
  - Nodes 1..K    : Top-K supervoxel nodes (spokes)
  - Edges         : GTV ↔ each supervoxel (bidirectional)
  - Edge features : 3D Euclidean distance between centroids

Selection of K supervoxels:
  The top-K supervoxels most similar to the GTV (by cosine
  similarity of radiomic features) are selected, following
  the RadGraph paper approach.

Usage:
    python graph_builder.py --all_patients
    python graph_builder.py --patient_id P001
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
import argparse
import json

import config


class GraphBuilder:
    """
    Converts per-patient supervoxel features into PyTorch Geometric Data objects.

    Each graph:
        x         : node features     (N, F)   float32
        edge_index: edge connectivity (2, E)   int64
        edge_attr : edge features     (E, 1)   float32  — normalised distance
        y         : label             (1,)     int64
        patient_id: str
        gtv_node_idx: int  (always 0)
    """

    def __init__(
        self,
        n_supervoxels_selected = None,
        distance_metric        = None,
        edge_weight_method     = None,
    ):
        """
        Parameters
        ----------
        n_supervoxels_selected : int
            Number of supervoxels to keep per patient (K).
        distance_metric : str
            'cosine' or 'euclidean' — used to rank supervoxel similarity to GTV.
        edge_weight_method : str
            'inverse_distance', 'gaussian', or 'uniform'.
        """
        self.K              = n_supervoxels_selected or config.N_SUPERVOXELS_SELECTED
        self.dist_metric    = distance_metric        or config.DISTANCE_METRIC
        self.edge_method    = edge_weight_method     or config.EDGE_WEIGHT_METHOD

        print(f"GraphBuilder initialised:")
        print(f"  Supervoxels per graph : {self.K}")
        print(f"  Similarity metric     : {self.dist_metric}")
        print(f"  Edge weight method    : {self.edge_method}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_patient_graph(self, patient_id, feature_dict, label, task='LR'):
        """
        Build one PyTorch Geometric Data object for a patient.

        Node features are filtered to the Table S3 selected features for the
        given task before graph construction.

        Parameters
        ----------
        patient_id   : str
        feature_dict : dict  (output of SupervoxelFeatureExtractor)
            Keys: 'gtv', 'supervoxels', 'centroids', 'n_supervoxels',
                  'feature_names'
        label        : int  0 or 1
        task         : 'LR' or 'DM'

        Returns
        -------
        graph : torch_geometric.data.Data  or None
        """
        gtv_feats  = feature_dict['gtv']         # (F,)
        sv_feats   = feature_dict['supervoxels']  # (n_sv, F)
        centroids  = feature_dict['centroids']    # (n_sv, 3)
        n_sv       = feature_dict['n_supervoxels']
        feat_names = feature_dict.get('feature_names', [])

        if n_sv == 0:
            print(f"  WARNING: No supervoxels for {patient_id} — skipping")
            return None

        # ── Filter to Table S3 selected features ─────────────────────────────
        selected_feat_names = config.get_selected_features(task)
        feature_indices     = self._get_feature_indices(feat_names, selected_feat_names)

        if len(feature_indices) > 0:
            gtv_feats = gtv_feats[feature_indices]
            sv_feats  = sv_feats[:, feature_indices]
            print(f"  Using {len(feature_indices)}/{len(feat_names)} Table S3 features "
                  f"for task={task}")
        else:
            print(f"  WARNING: Table S3 features not found in extracted features.")
            print(f"  Expected: {selected_feat_names}")
            print(f"  Available (first 5): {feat_names[:5]}")
            print(f"  Using all {len(feat_names)} features instead.")

        # 1. Select top-K supervoxels by similarity to GTV
        selected_idx = self._select_supervoxels(gtv_feats, sv_feats)

        sv_feats_sel  = sv_feats[selected_idx]    # (K, F)
        centroids_sel = centroids[selected_idx]   # (K, 3)

        # ── GTV-relative normalisation (paper exact method) ──────────────────
        # Paper: "Selected radiomic features for all supervoxels were then
        # normalized with respect to minimum and maximum GTV radiomic
        # feature expression."
        # This means: per patient, use the GTV's own feature values as the
        # [min, max] reference for normalising supervoxel features.
        sv_feats_sel, gtv_feats = self._gtv_relative_normalise(
            sv_feats_sel, gtv_feats
        )
        gtv_centroid  = centroids_sel.mean(axis=0, keepdims=True)   # (1, 3)

        # 3. Build node feature matrix  [GTV | SV_1 | ... | SV_K]
        #    Shape: (K+1, F)
        node_features = np.vstack([
            gtv_feats[np.newaxis, :],   # GTV node at index 0
            sv_feats_sel                 # Supervoxel nodes
        ]).astype(np.float32)

        # 4. All centroids: GTV first, then supervoxels
        all_centroids = np.vstack([gtv_centroid, centroids_sel]).astype(np.float32)

        # 5. Build edges (star topology: GTV ↔ each SV, bidirectional)
        edge_index, edge_attr = self._build_star_edges(
            n_nodes   = len(selected_idx) + 1,
            centroids = all_centroids
        )

        # 6. Assemble PyG Data object
        graph = Data(
            x          = torch.tensor(node_features, dtype=torch.float),
            edge_index = torch.tensor(edge_index,    dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,     dtype=torch.float),
            y          = torch.tensor([label],        dtype=torch.long),
        )

        # Store metadata as plain attributes
        graph.patient_id  = patient_id
        graph.gtv_node_idx = 0
        graph.n_nodes      = node_features.shape[0]
        graph.n_features   = node_features.shape[1]

        return graph

    def build_all_graphs(self, feature_cache_dir, clinical_df, task='LR', save_dir=None):
        """
        Build graphs for all patients and optionally save to disk.

        Parameters
        ----------
        feature_cache_dir : Path  — directory with <patient_id>_features.npz
        clinical_df       : pd.DataFrame
        task              : 'LR' or 'DM'
        save_dir          : Path or None  — save individual .pt files here

        Returns
        -------
        graphs  : list[Data]
        failed  : list[str]
        """
        feature_cache_dir = Path(feature_cache_dir)
        outcome_col       = config.get_outcome_column(task)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        graphs = []
        failed = []

        patient_ids = clinical_df[config.PATIENT_ID_COL].tolist()
        print(f"\nBuilding graphs for {len(patient_ids)} patients (task={task})...")

        for patient_id in tqdm(patient_ids, desc='Building graphs'):
            # Check cached .pt file first
            if save_dir:
                pt_file = save_dir / f'{patient_id}_{task}.pt'
                if pt_file.exists():
                    graphs.append(torch.load(pt_file))
                    continue

            # Load feature dict
            feat_file = feature_cache_dir / f'{patient_id}_features.npz'
            if not feat_file.exists():
                failed.append(patient_id)
                continue

            feature_dict = self._load_feature_dict(feat_file)
            if feature_dict is None:
                failed.append(patient_id)
                continue

            # Get label
            row   = clinical_df[clinical_df[config.PATIENT_ID_COL] == patient_id]
            if len(row) == 0:
                failed.append(patient_id)
                continue
            label = int(row[outcome_col].values[0])

            # Build graph
            graph = self.build_patient_graph(patient_id, feature_dict, label)
            if graph is None:
                failed.append(patient_id)
                continue

            graphs.append(graph)

            # Save .pt file
            if save_dir:
                torch.save(graph, save_dir / f'{patient_id}_{task}.pt')

        print(f"Graphs built: {len(graphs)} success, {len(failed)} failed")
        if failed:
            print(f"Failed: {failed}")

        return graphs, failed

    def get_graph_statistics(self, graphs):
        """
        Print summary statistics for a list of graphs.

        Parameters
        ----------
        graphs : list[Data]
        """
        if len(graphs) == 0:
            print("No graphs to summarise.")
            return

        n_nodes   = [g.num_nodes   for g in graphs]
        n_edges   = [g.num_edges   for g in graphs]
        n_features= [g.x.shape[1]  for g in graphs]
        labels    = [g.y.item()    for g in graphs]

        print("\n=== Graph Statistics ===")
        print(f"  Total graphs      : {len(graphs)}")
        print(f"  Nodes per graph   : {np.mean(n_nodes):.1f} ± {np.std(n_nodes):.1f}  "
              f"(range {np.min(n_nodes)}–{np.max(n_nodes)})")
        print(f"  Edges per graph   : {np.mean(n_edges):.1f} ± {np.std(n_edges):.1f}")
        print(f"  Features per node : {n_features[0]}")
        print(f"  Label distribution: {labels.count(0)} negative, {labels.count(1)} positive")
        print("=" * 30)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _select_supervoxels(self, gtv_feats, sv_feats):
        """
        Select top-K supervoxels most similar to GTV.

        Parameters
        ----------
        gtv_feats : np.ndarray  (F,)
        sv_feats  : np.ndarray  (n_sv, F)

        Returns
        -------
        selected_idx : np.ndarray  (K,)  indices into sv_feats
        """
        n_sv = sv_feats.shape[0]
        K    = min(self.K, n_sv)   # Can't select more than available

        if self.dist_metric == 'cosine':
            similarities = self._cosine_similarity(gtv_feats, sv_feats)
            # Higher similarity = more similar to GTV → select top-K
            selected_idx = np.argsort(similarities)[::-1][:K]

        else:   # Euclidean distance
            distances    = np.linalg.norm(sv_feats - gtv_feats[np.newaxis, :], axis=1)
            # Smaller distance = more similar → select smallest K
            selected_idx = np.argsort(distances)[:K]

        return selected_idx

    @staticmethod
    def _gtv_relative_normalise(sv_feats, gtv_feats):
        """
        Normalise supervoxel features relative to the GTV's own min/max values.

        Paper (Methods section):
          "Selected radiomic features for all supervoxels were then normalized
           with respect to minimum and maximum GTV radiomic feature expression."

        This is a per-patient normalisation:
          - For each feature dimension f:
              sv_norm[:, f] = (sv[:, f] - gtv_min[f]) / (gtv_max[f] - gtv_min[f])
          - Since GTV is a single vector (not a distribution), its min and max
            are taken element-wise from the full GTV feature vector range.
          - Practically: min = 0-clipped GTV value, max = GTV value itself
            → supervoxels are expressed as multiples of GTV expression.

        Implementation note:
          We use the GTV feature vector as the reference scale.
          Each supervoxel feature is divided by the GTV feature value
          (clamped to avoid division by zero), placing supervoxels in
          units of "GTV expression."

        Parameters
        ----------
        sv_feats  : np.ndarray  (K, F)  — selected supervoxel features
        gtv_feats : np.ndarray  (F,)    — GTV feature vector

        Returns
        -------
        sv_norm   : np.ndarray  (K, F)  — normalised supervoxels
        gtv_norm  : np.ndarray  (F,)    — GTV as reference (ones where nonzero)
        """
        gtv_abs  = np.abs(gtv_feats)
        scale    = np.where(gtv_abs > 1e-8, gtv_abs, 1.0)

        sv_norm  = sv_feats  / scale[np.newaxis, :]
        gtv_norm = gtv_feats / scale   # GTV normalises to ±1 per feature

        # Clip to reasonable range to prevent extreme values
        sv_norm  = np.clip(sv_norm,  -10.0, 10.0)
        gtv_norm = np.clip(gtv_norm, -10.0, 10.0)

        return sv_norm.astype(np.float32), gtv_norm.astype(np.float32)

    @staticmethod
    def _get_feature_indices(all_feature_names, selected_feature_names):
        """
        Find indices of Table S3 selected features in the full feature list.

        Matching is done flexibly — checks both exact match and suffix match
        since PyRadiomics may prepend 'original_' or other prefixes.

        Parameters
        ----------
        all_feature_names      : list[str]  — full extracted feature list
        selected_feature_names : list[str]  — Table S3 feature names

        Returns
        -------
        indices : list[int]  — may be shorter than selected_feature_names
                               if some features are missing
        """
        indices = []
        for sel_name in selected_feature_names:
            # Try exact match first
            if sel_name in all_feature_names:
                indices.append(all_feature_names.index(sel_name))
                continue

            # Try suffix match (e.g. 'GrayLevelNonUniformity' in
            # 'original_glszm_GrayLevelNonUniformity')
            sel_lower = sel_name.lower()
            found     = False
            for i, fname in enumerate(all_feature_names):
                if sel_lower in fname.lower() or fname.lower() in sel_lower:
                    indices.append(i)
                    found = True
                    break
            if not found:
                print(f"  WARNING: Feature not found: {sel_name}")

        return indices

    @staticmethod
    def _cosine_similarity(vec, matrix):
        """
        Cosine similarity between a vector and each row of a matrix.

        Returns
        -------
        sims : np.ndarray  (n_rows,)
        """
        vec_norm    = vec    / (np.linalg.norm(vec)           + 1e-8)
        mat_norms   = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return mat_norms @ vec_norm

    def _build_star_edges(self, n_nodes, centroids):
        """
        Build star-topology edge index and edge attributes.

        Node 0 = GTV (hub). All other nodes connect to node 0.
        Edges are bidirectional.

        Parameters
        ----------
        n_nodes   : int
        centroids : np.ndarray  (n_nodes, 3)

        Returns
        -------
        edge_index : np.ndarray  (2, 2*(n_nodes-1))
        edge_attr  : np.ndarray  (2*(n_nodes-1), 1)  normalised distances
        """
        src_edges = []
        dst_edges = []
        distances = []

        for sv_idx in range(1, n_nodes):
            dist = np.linalg.norm(centroids[0] - centroids[sv_idx])

            # GTV → SV
            src_edges.append(0)
            dst_edges.append(sv_idx)
            distances.append(dist)

            # SV → GTV  (bidirectional)
            src_edges.append(sv_idx)
            dst_edges.append(0)
            distances.append(dist)

        edge_index = np.array([src_edges, dst_edges], dtype=np.int64)
        distances  = np.array(distances, dtype=np.float32)

        # Normalise edge weights
        edge_attr = self._compute_edge_weights(distances)

        return edge_index, edge_attr[:, np.newaxis]  # (E, 1)

    def _compute_edge_weights(self, distances):
        """
        Compute edge weights from distances.

        Parameters
        ----------
        distances : np.ndarray  (E,)

        Returns
        -------
        weights : np.ndarray  (E,)
        """
        if self.edge_method == 'inverse_distance':
            weights = 1.0 / (distances + 1e-6)
            # Normalise to [0, 1]
            max_w = weights.max()
            if max_w > 0:
                weights /= max_w

        elif self.edge_method == 'gaussian':
            sigma   = config.EDGE_WEIGHT_SIGMA
            weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

        else:   # uniform
            weights = np.ones_like(distances)

        return weights.astype(np.float32)

    @staticmethod
    def _load_feature_dict(feat_file):
        """Load feature dict from .npz cache file."""
        try:
            data = np.load(feat_file, allow_pickle=True)
            return {
                'gtv'           : data['gtv'],
                'supervoxels'   : data['supervoxels'],
                'centroids'     : data['centroids'],
                'valid_sv_ids'  : data['valid_sv_ids'],
                'feature_names' : list(data['feature_names']),
                'n_supervoxels' : int(data['n_supervoxels']),
            }
        except Exception as e:
            print(f"  ERROR loading {feat_file}: {e}")
            return None


# ─── Feature normalisation helper ─────────────────────────────────────────────

def normalise_graph_features(graphs_train, graphs_val, graphs_test):
    """
    Min-max normalise node features using training set statistics.

    Per Appendix S1:
      "radiomic features used as input to models were the min-max normalized
       radiomic features from the gross target volume (GTV) region"

    Fit scaler on train graphs, apply to all three splits.

    Parameters
    ----------
    graphs_train, graphs_val, graphs_test : list[Data]

    Returns
    -------
    graphs_train, graphs_val, graphs_test : list[Data]  — normalised x
    scaler_min : np.ndarray  (F,)
    scaler_max : np.ndarray  (F,)
    """
    from sklearn.preprocessing import MinMaxScaler

    # Collect all node features from training graphs
    all_train_features = np.vstack([g.x.numpy() for g in graphs_train])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_train_features)

    def apply_normalisation(graphs):
        normalised = []
        for g in graphs:
            g_copy   = g.clone()
            x_norm   = scaler.transform(g_copy.x.numpy())
            g_copy.x = torch.tensor(x_norm, dtype=torch.float)
            normalised.append(g_copy)
        return normalised

    graphs_train = apply_normalisation(graphs_train)
    graphs_val   = apply_normalisation(graphs_val)
    graphs_test  = apply_normalisation(graphs_test)

    print(f"Node features min-max normalised to [0, 1] using train statistics")
    print(f"  Feature dim : {all_train_features.shape[1]}")
    print(f"  (Appendix S1: min-max normalization for radiomic features)")

    return graphs_train, graphs_val, graphs_test, scaler.data_min_, scaler.data_max_


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build graphs from supervoxel features')
    parser.add_argument('--task',           type=str, default='LR', choices=['LR', 'DM'])
    parser.add_argument('--all_patients',   action='store_true')
    parser.add_argument('--patient_id',     type=str, default=None)
    parser.add_argument('--feature_dir',    type=str,
                        default=str(config.OUTPUT_DIR / 'features_cache'))
    parser.add_argument('--save_dir',       type=str,
                        default=str(config.OUTPUT_DIR / 'graphs'))
    args = parser.parse_args()

    import pandas as pd
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)

    builder = GraphBuilder()

    if args.all_patients:
        graphs, failed = builder.build_all_graphs(
            feature_cache_dir = args.feature_dir,
            clinical_df       = clinical_df,
            task              = args.task,
            save_dir          = args.save_dir
        )
        builder.get_graph_statistics(graphs)

    elif args.patient_id:
        feat_file    = Path(args.feature_dir) / f'{args.patient_id}_features.npz'
        if not feat_file.exists():
            print(f"Feature file not found: {feat_file}")
            return

        feature_dict = GraphBuilder._load_feature_dict(feat_file)
        row = clinical_df[clinical_df[config.PATIENT_ID_COL] == args.patient_id]
        label = int(row[config.get_outcome_column(args.task)].values[0]) if len(row) else 0

        graph = builder.build_patient_graph(args.patient_id, feature_dict, label)
        if graph:
            print(f"\nGraph for {args.patient_id}:")
            print(f"  Nodes       : {graph.num_nodes}")
            print(f"  Edges       : {graph.num_edges}")
            print(f"  Node feat.  : {graph.x.shape}")
            print(f"  Edge feat.  : {graph.edge_attr.shape}")
            print(f"  Label       : {graph.y.item()}")
    else:
        print("Use --all_patients or --patient_id <id>")
        parser.print_help()


if __name__ == '__main__':
    main()
