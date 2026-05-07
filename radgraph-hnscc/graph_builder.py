"""
Graph Builder for RadGraph Implementation
==========================================
Builds patient-level graphs from supervoxel features.

Graph topology — two modes (selectable via config):

  Star / Hub-and-Spoke (paper topology):
    - Node 0        : GTV node  (hub)
    - Nodes 1..K    : Top-K supervoxel nodes (spokes)
    - Edges         : GTV ↔ each supervoxel (bidirectional)
    - Edge features : 3D Euclidean distance between centroids

  K-Nearest-Neighbour (Joseph Bae's GraphGeneration.py topology):
    - All supervoxel nodes including the GTV
    - Each node connects to its K nearest neighbours in radiomic feature space
    - Edge features : radiomic Euclidean distance between node feature vectors
    - Isolated nodes removed after graph construction

Feature normalisation (incorporated from Joseph's ScalerGeneration.py):
  - Z-score StandardScaler fitted across ALL training patients
  - Applied per-node before graph construction  (Joseph's approach)
  - Outlier clipping at ±5 after z-score norm   (Joseph's approach)
  - GTV-relative per-patient normalisation       (paper Appendix S1)
  Both are available. Set config.NORMALISATION_METHOD = 'zscore' or 'gtv_relative'.

Usage:
    python graph_builder.py --all_patients
    python graph_builder.py --patient_id P001
    python graph_builder.py --all_patients --topology knn --neighbors 10
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import joblib

import config


class GraphBuilder:
    """
    Converts per-patient supervoxel features into PyTorch Geometric Data objects.

    Supports two graph topologies:
        'star' : GTV hub connected to each supervoxel (paper topology)
        'knn'  : K-nearest-neighbour in radiomic feature space (Joseph's topology)

    Each graph:
        x         : node features     (N, F)   float32
        edge_index: edge connectivity (2, E)   int64
        edge_attr : edge features     (E, 1)   float32
        y         : label             (1,)     int64
        patient_id: str
        gtv_node_idx: int  (0 for star; last node index for knn)
    """

    def __init__(
        self,
        n_supervoxels_selected = None,
        distance_metric        = None,
        edge_weight_method     = None,
        topology               = None,
        n_neighbors            = None,
        scaler                 = None,
    ):
        """
        Parameters
        ----------
        n_supervoxels_selected : int
            Number of supervoxels to keep per patient (K). Star topology only.
        distance_metric : str
            'cosine' or 'euclidean' — similarity metric for supervoxel selection.
        edge_weight_method : str
            'inverse_distance', 'gaussian', or 'uniform'. Star topology only.
        topology : str
            'star' — GTV hub-and-spoke (paper).
            'knn'  — K-nearest-neighbour in radiomic space (Joseph's code).
        n_neighbors : int
            Number of neighbours per node. KNN topology only. Default 10.
        scaler : StandardScaler or None
            Pre-fitted z-score scaler from fit_dataset_scaler().
            If provided, z-score normalisation + clip at ±5 is applied (Joseph).
            If None, per-patient GTV-relative normalisation is used (paper).
        """
        self.K           = config.N_SUPERVOXELS_SELECTED if n_supervoxels_selected is None \
                           else n_supervoxels_selected
        self.dist_metric = config.DISTANCE_METRIC        if distance_metric        is None \
                           else distance_metric
        self.edge_method = config.EDGE_WEIGHT_METHOD     if edge_weight_method     is None \
                           else edge_weight_method
        self.topology    = topology    or getattr(config, 'GRAPH_TOPOLOGY', 'star')
        self.n_neighbors = n_neighbors or getattr(config, 'KNN_NEIGHBORS', 10)
        self.scaler      = scaler

        print(f"GraphBuilder initialised:")
        print(f"  Topology              : {self.topology}")
        print(f"  Supervoxels per graph : {self.K}  (star topology)")
        print(f"  KNN neighbors         : {self.n_neighbors}  (knn topology)")
        print(f"  Similarity metric     : {self.dist_metric}")
        print(f"  Normalisation         : {'zscore + clip±5 (Joseph)' if self.scaler else 'gtv_relative (paper)'}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_patient_graph(self, patient_id, feature_dict, label, task='LR'):
        """
        Build one PyTorch Geometric Data object for a patient.

        Incorporates from Joseph's GraphGeneration.py:
          - Z-score normalisation per node (if scaler provided)
          - Outlier clipping at ±5 after z-score norm
          - KNN topology option (radiomic distance edges, isolated node removal)

        Parameters
        ----------
        patient_id   : str
        feature_dict : dict  (output of SupervoxelFeatureExtractor)
        label        : int  0 or 1
        task         : 'LR' or 'DM'

        Returns
        -------
        graph : torch_geometric.data.Data  or None
        """
        gtv_feats  = feature_dict['gtv'].copy()
        sv_feats   = feature_dict['supervoxels'].copy()
        centroids  = feature_dict['centroids'].copy()
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
        else:
            print(f"  WARNING: Table S3 features not found — using all {len(feat_names)} features.")

        # ── Feature normalisation ─────────────────────────────────────────────
        if self.scaler is not None:
            # Joseph's approach: z-score normalisation then clip at ±5
            # "assume we performed z-score normalization, we are clipping
            #  outlier features" — GraphGeneration.py
            all_nodes = np.vstack([gtv_feats[np.newaxis, :], sv_feats])
            all_nodes = self.scaler.transform(all_nodes)
            all_nodes = np.clip(all_nodes, -5.0, 5.0)
            gtv_feats = all_nodes[0]
            sv_feats  = all_nodes[1:]
        # If no scaler: GTV-relative normalisation applied later inside _build_star_graph

        # ── Route to topology builder ─────────────────────────────────────────
        if self.topology == 'knn':
            return self._build_knn_graph(
                patient_id, gtv_feats, sv_feats, centroids, label
            )
        else:
            return self._build_star_graph(
                patient_id, gtv_feats, sv_feats, centroids, label, task
            )

    # ── Star topology ─────────────────────────────────────────────────────────

    def _build_star_graph(self, patient_id, gtv_feats, sv_feats,
                          centroids, label, task):
        """Hub-and-spoke star graph. Follows paper topology exactly."""
        selected_idx  = self._select_supervoxels(gtv_feats, sv_feats)
        sv_feats_sel  = sv_feats[selected_idx]
        centroids_sel = centroids[selected_idx]

        # Per-patient GTV-relative normalisation (paper method, only if no scaler)
        if self.scaler is None:
            sv_feats_sel, gtv_feats = self._gtv_relative_normalise(
                sv_feats_sel, gtv_feats
            )

        gtv_centroid  = centroids_sel.mean(axis=0, keepdims=True)
        node_features = np.vstack([
            gtv_feats[np.newaxis, :],   # GTV always at index 0
            sv_feats_sel
        ]).astype(np.float32)
        all_centroids = np.vstack([gtv_centroid, centroids_sel]).astype(np.float32)

        edge_index, edge_attr = self._build_star_edges(
            n_nodes=len(selected_idx) + 1, centroids=all_centroids
        )

        graph = Data(
            x          = torch.tensor(node_features, dtype=torch.float),
            edge_index = torch.tensor(edge_index,    dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,     dtype=torch.float),
            y          = torch.tensor([label],        dtype=torch.long),
        )
        graph.patient_id   = patient_id
        graph.gtv_node_idx = 0
        graph.n_nodes      = node_features.shape[0]
        graph.n_features   = node_features.shape[1]
        graph.topology     = 'star'
        return graph

    # ── KNN topology (Joseph's GraphGeneration.py) ────────────────────────────

    def _build_knn_graph(self, patient_id, gtv_feats, sv_feats,
                         centroids, label):
        """
        K-nearest-neighbour graph in radiomic feature space.

        Directly mirrors Joseph's make_rad_graph_gtv_readout():
          - pdist + squareform for pairwise radiomic distances
          - Each node connects to its K nearest neighbours
          - Isolated nodes removed (mirrors nx.isolates removal)
          - GTV appended as last node (Joseph's readout reads last node)
          - Edge attributes = radiomic distance (not physical distance)
        """
        # GTV is appended as last node (Joseph's convention)
        all_feats     = np.vstack([sv_feats, gtv_feats[np.newaxis, :]])  # (n_sv+1, F)
        n_nodes       = all_feats.shape[0]
        gtv_idx       = n_nodes - 1

        # Pairwise radiomic Euclidean distances (Joseph: pdist + squareform)
        distances    = pdist(all_feats)
        square_dist  = squareform(distances)

        edges_src = []
        edges_dst = []
        edge_dists= []

        for i in range(n_nodes):
            sorted_inds = np.argsort(square_dist[i, :])
            n_connected = 0
            for j in sorted_inds:
                if n_connected >= self.n_neighbors:
                    break
                if j == i:
                    continue
                edges_src.append(i);  edges_dst.append(j);  edge_dists.append(square_dist[i, j])
                edges_src.append(j);  edges_dst.append(i);  edge_dists.append(square_dist[i, j])
                n_connected += 1

        if not edges_src:
            print(f"  WARNING: No edges for {patient_id} — skipping")
            return None

        # ── Remove isolated nodes (mirrors nx.isolates removal) ───────────────
        connected_nodes = set(edges_src) | set(edges_dst)
        isolated        = set(range(n_nodes)) - connected_nodes

        if isolated:
            print(f"  Removing {len(isolated)} isolated nodes for {patient_id}")
            keep    = [i for i in range(n_nodes) if i not in isolated]
            remap   = {old: new for new, old in enumerate(keep)}
            all_feats = all_feats[keep]

            valid   = [(s, d, w) for s, d, w in zip(edges_src, edges_dst, edge_dists)
                       if s not in isolated and d not in isolated]
            if not valid:
                print(f"  WARNING: No valid edges after isolation removal — skipping")
                return None

            edges_src  = [remap[s] for s, _, _ in valid]
            edges_dst  = [remap[d] for _, d, _ in valid]
            edge_dists = [w        for _, _, w in valid]
            gtv_idx    = remap.get(n_nodes - 1, len(all_feats) - 1)

        node_features = all_feats.astype(np.float32)
        edge_index    = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_attr     = np.array(edge_dists, dtype=np.float32)[:, np.newaxis]

        graph = Data(
            x          = torch.tensor(node_features, dtype=torch.float),
            edge_index = torch.tensor(edge_index,    dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,     dtype=torch.float),
            y          = torch.tensor([label],        dtype=torch.long),
        )
        graph.patient_id   = patient_id
        graph.gtv_node_idx = int(gtv_idx)
        graph.n_nodes      = node_features.shape[0]
        graph.n_features   = node_features.shape[1]
        graph.topology     = 'knn'
        return graph

    # ── Batch builder ──────────────────────────────────────────────────────────

    def build_all_graphs(self, feature_cache_dir, clinical_df, task='LR', save_dir=None):
        feature_cache_dir = Path(feature_cache_dir)
        outcome_col       = config.get_outcome_column(task)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        graphs, failed = [], []
        patient_ids    = clinical_df[config.PATIENT_ID_COL].tolist()

        print(f"\nBuilding graphs for {len(patient_ids)} patients "
              f"(task={task}, topology={self.topology})...")

        for patient_id in tqdm(patient_ids, desc='Building graphs'):
            if save_dir:
                pt_file = save_dir / f'{patient_id}_{task}.pt'
                if pt_file.exists():
                    graphs.append(torch.load(pt_file, weights_only=False))
                    continue

            feat_file = feature_cache_dir / f'{patient_id}_features.npz'
            if not feat_file.exists():
                failed.append(patient_id); continue

            feature_dict = self._load_feature_dict(feat_file)
            if feature_dict is None:
                failed.append(patient_id); continue

            row = clinical_df[clinical_df[config.PATIENT_ID_COL] == patient_id]
            if len(row) == 0:
                failed.append(patient_id); continue

            label = int(row[outcome_col].values[0])
            graph = self.build_patient_graph(patient_id, feature_dict, label, task)
            if graph is None:
                failed.append(patient_id); continue

            graphs.append(graph)
            if save_dir:
                torch.save(graph, save_dir / f'{patient_id}_{task}.pt')

        print(f"Graphs built: {len(graphs)} success, {len(failed)} failed")
        if failed:
            print(f"Failed: {failed}")
        return graphs, failed

    def get_graph_statistics(self, graphs):
        if not graphs:
            print("No graphs to summarise."); return
        n_nodes  = [g.num_nodes  for g in graphs]
        n_edges  = [g.num_edges  for g in graphs]
        labels   = [g.y.item()   for g in graphs]
        print("\n=== Graph Statistics ===")
        print(f"  Total graphs      : {len(graphs)}")
        print(f"  Topology          : {graphs[0].topology if hasattr(graphs[0], 'topology') else 'unknown'}")
        print(f"  Nodes per graph   : {np.mean(n_nodes):.1f} ± {np.std(n_nodes):.1f}  "
              f"(range {np.min(n_nodes)}–{np.max(n_nodes)})")
        print(f"  Edges per graph   : {np.mean(n_edges):.1f} ± {np.std(n_edges):.1f}")
        print(f"  Features per node : {graphs[0].x.shape[1]}")
        print(f"  Label distribution: {labels.count(0)} negative, {labels.count(1)} positive")
        print("=" * 30)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _select_supervoxels(self, gtv_feats, sv_feats):
        n_sv = sv_feats.shape[0]
        K    = min(self.K, n_sv)
        if self.dist_metric == 'cosine':
            sims         = self._cosine_similarity(gtv_feats, sv_feats)
            selected_idx = np.argsort(sims)[::-1][:K]
        else:
            dists        = np.linalg.norm(sv_feats - gtv_feats[np.newaxis, :], axis=1)
            selected_idx = np.argsort(dists)[:K]
        return selected_idx

    @staticmethod
    def _gtv_relative_normalise(sv_feats, gtv_feats):
        """Per-patient GTV-relative normalisation (paper). Clips at ±10."""
        scale    = np.where(np.abs(gtv_feats) > 1e-8, np.abs(gtv_feats), 1.0)
        sv_norm  = np.clip(sv_feats  / scale[np.newaxis, :], -10.0, 10.0)
        gtv_norm = np.clip(gtv_feats / scale,                -10.0, 10.0)
        return sv_norm.astype(np.float32), gtv_norm.astype(np.float32)

    @staticmethod
    def _get_feature_indices(all_feature_names, selected_feature_names):
        indices = []
        for sel_name in selected_feature_names:
            if sel_name in all_feature_names:
                indices.append(all_feature_names.index(sel_name)); continue
            found = False
            for i, fname in enumerate(all_feature_names):
                if sel_name.lower() in fname.lower() or fname.lower() in sel_name.lower():
                    indices.append(i); found = True; break
            if not found:
                print(f"  WARNING: Feature not found: {sel_name}")
        return indices

    @staticmethod
    def _cosine_similarity(vec, matrix):
        vec_norm  = vec    / (np.linalg.norm(vec)                           + 1e-8)
        mat_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return mat_norms @ vec_norm

    def _build_star_edges(self, n_nodes, centroids):
        src, dst, dists = [], [], []
        for sv_idx in range(1, n_nodes):
            d = np.linalg.norm(centroids[0] - centroids[sv_idx])
            src.append(0);      dst.append(sv_idx); dists.append(d)
            src.append(sv_idx); dst.append(0);      dists.append(d)
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr  = self._compute_edge_weights(np.array(dists, dtype=np.float32))
        return edge_index, edge_attr[:, np.newaxis]

    def _compute_edge_weights(self, distances):
        if self.edge_method == 'inverse_distance':
            w = 1.0 / (distances + 1e-6)
            if w.max() > 0: w /= w.max()
        elif self.edge_method == 'gaussian':
            w = np.exp(-(distances ** 2) / (2 * config.EDGE_WEIGHT_SIGMA ** 2))
        else:
            w = np.ones_like(distances)
        return w.astype(np.float32)

    @staticmethod
    def _load_feature_dict(feat_file):
        try:
            data = np.load(feat_file, allow_pickle=True)
            return {
                'gtv'          : data['gtv'],
                'supervoxels'  : data['supervoxels'],
                'centroids'    : data['centroids'],
                'valid_sv_ids' : data['valid_sv_ids'],
                'feature_names': list(data['feature_names']),
                'n_supervoxels': int(data['n_supervoxels'][0]),
            }
        except Exception as e:
            print(f"  ERROR loading {feat_file}: {e}")
            return None


# ─── Dataset-level scaler (Joseph's ScalerGeneration.py approach) ─────────────

def fit_dataset_scaler(feature_cache_dir, patient_ids, scaler_type='zscore',
                       save_path=None):
    """
    Fit a scaler across ALL training patients BEFORE graph construction.

    This implements Joseph's ScalerGeneration.py approach:
      - Fit on training split only (never val/test)
      - Z-score StandardScaler (default) or MinMaxScaler
      - Pass the returned scaler to GraphBuilder(scaler=scaler)

    Parameters
    ----------
    feature_cache_dir : Path or str
    patient_ids       : list[str]  — TRAINING patients only
    scaler_type       : 'zscore' or 'minmax'
    save_path         : Path or None

    Returns
    -------
    scaler : fitted StandardScaler or MinMaxScaler

    Example
    -------
    scaler = fit_dataset_scaler(
        config.OUTPUT_DIR / 'features_cache',
        train_patient_ids,
        scaler_type='zscore',
        save_path=config.MODEL_DIR / 'feature_scaler.pkl'
    )
    builder = GraphBuilder(scaler=scaler)
    graphs  = builder.build_all_graphs(...)
    """
    feature_cache_dir = Path(feature_cache_dir)
    all_features      = []

    print(f"\nFitting {scaler_type} scaler across {len(patient_ids)} training patients...")

    for patient_id in tqdm(patient_ids, desc='Loading features for scaler'):
        feat_file = feature_cache_dir / f'{patient_id}_features.npz'
        if not feat_file.exists():
            continue
        try:
            data = np.load(feat_file, allow_pickle=True)
            # Stack GTV + supervoxels so scaler sees the full feature range
            all_features.append(np.vstack([
                data['gtv'][np.newaxis, :],
                data['supervoxels']
            ]))
        except Exception as e:
            print(f"  Skipping {patient_id}: {e}")

    if not all_features:
        raise RuntimeError("No features loaded — check feature_cache_dir.")

    feature_matrix = np.vstack(all_features)
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    scaler = StandardScaler() if scaler_type == 'zscore' else MinMaxScaler((0, 1))
    scaler.fit(feature_matrix)
    print(f"  Scaler fitted successfully.")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'scaler': scaler, 'type': scaler_type}, save_path)
        print(f"  Scaler saved to {save_path}")

    return scaler


# ─── Post-build normalisation (original approach) ─────────────────────────────

def normalise_graph_features(graphs_train, graphs_val, graphs_test):
    """
    Min-max normalise node features using training statistics, applied
    AFTER graphs are built. Per Appendix S1.
    """
    all_train = np.vstack([g.x.numpy() for g in graphs_train])
    scaler    = MinMaxScaler((0, 1))
    scaler.fit(all_train)

    def apply(graphs):
        out = []
        for g in graphs:
            c = g.clone()
            c.x = torch.tensor(scaler.transform(c.x.numpy()), dtype=torch.float)
            out.append(c)
        return out

    graphs_train = apply(graphs_train)
    graphs_val   = apply(graphs_val)
    graphs_test  = apply(graphs_test)

    print(f"Node features min-max normalised [0,1] (feature dim: {all_train.shape[1]})")
    return graphs_train, graphs_val, graphs_test, scaler.data_min_, scaler.data_max_


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build graphs from supervoxel features')
    parser.add_argument('--task',          type=str, default='LR', choices=['LR', 'DM'])
    parser.add_argument('--topology',      type=str, default='star', choices=['star', 'knn'],
                        help='star=hub-and-spoke (paper), knn=K-nearest-neighbour (Joseph)')
    parser.add_argument('--neighbors',     type=int, default=10,
                        help='KNN neighbours per node (knn topology only)')
    parser.add_argument('--normalisation', type=str, default='gtv_relative',
                        choices=['gtv_relative', 'zscore'],
                        help='gtv_relative=paper, zscore=Joseph')
    parser.add_argument('--all_patients',  action='store_true')
    parser.add_argument('--patient_id',    type=str, default=None)
    parser.add_argument('--feature_dir',   type=str,
                        default=str(config.OUTPUT_DIR / 'features_cache'))
    parser.add_argument('--save_dir',      type=str,
                        default=str(config.OUTPUT_DIR / 'graphs'))
    args = parser.parse_args()

    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)

    scaler = None
    if args.normalisation == 'zscore':
        scaler_path = config.MODEL_DIR / 'feature_scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)['scaler']
            print(f"Loaded z-score scaler from {scaler_path}")
        else:
            print("Z-score scaler not found — run fit_dataset_scaler() first.")
            print("Falling back to GTV-relative normalisation.")

    builder = GraphBuilder(
        topology    = args.topology,
        n_neighbors = args.neighbors,
        scaler      = scaler,
    )

    if args.all_patients:
        graphs, _ = builder.build_all_graphs(
            feature_cache_dir = args.feature_dir,
            clinical_df       = clinical_df,
            task              = args.task,
            save_dir          = args.save_dir
        )
        builder.get_graph_statistics(graphs)

    elif args.patient_id:
        feat_file    = Path(args.feature_dir) / f'{args.patient_id}_features.npz'
        if not feat_file.exists():
            print(f"Feature file not found: {feat_file}"); return
        feature_dict = GraphBuilder._load_feature_dict(feat_file)
        row   = clinical_df[clinical_df[config.PATIENT_ID_COL] == args.patient_id]
        label = int(row[config.get_outcome_column(args.task)].values[0]) if len(row) else 0
        graph = builder.build_patient_graph(args.patient_id, feature_dict, label)
        if graph:
            print(f"\nGraph for {args.patient_id}:")
            print(f"  Topology    : {graph.topology}")
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