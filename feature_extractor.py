"""
Feature Extractor for RadGraph Implementation
==========================================
Extracts PyRadiomics features from:
  - Whole GTV (one feature vector per patient)
  - Each supervoxel (one feature vector per supervoxel)

Features extracted per region:
  - First-order statistics (18 features)
  - GLCM texture features (24 features)
  - GLRLM features (16 features)
  - GLSZM features (16 features)
  - GLDM features (14 features)
  - NGTDM features (5 features)
  Total: ~93 features

Usage:
    python feature_extractor.py --patient_id P001
    python feature_extractor.py --all_patients
    python feature_extractor.py --all_patients --skip_existing
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

try:
    from radiomics import featureextractor
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False
    print("WARNING: PyRadiomics not installed. Feature extraction disabled.")
    print("Install with: pip install pyradiomics==3.0.1")

import config

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
radiomics_logger = logging.getLogger('radiomics')
radiomics_logger.setLevel(logging.ERROR)   # Suppress verbose radiomics logs


# ─── PyRadiomics Parameter YAML (built in-memory) ─────────────────────────────
PYRADIOMICS_PARAMS = {
    'setting': {
        'binWidth': 25,
        'resampledPixelSpacing': config.TARGET_SPACING,
        'interpolator': 'sitkBSpline',
        'normalize': True,
        'normalizeScale': 100,
        'removeOutliers': None,
        'minimumROIDimensions': 1,
        'minimumROISize': 1,
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
        'glrlm': [],
        'glszm': [],
        'gldm': [],
        'ngtdm': [],
    }
}


class SupervoxelFeatureExtractor:
    """
    Extracts radiomic features from GTV and supervoxels using PyRadiomics.

    Pipeline per patient:
        1. Load preprocessed CT + GTV mask
        2. Extract GTV-level features (whole ROI)
        3. For each supervoxel, create a binary mask and extract features
        4. Save results to CSV cache
    """

    def __init__(self, cache_dir=None):
        """
        Parameters
        ----------
        cache_dir : Path or str, optional
            Directory to cache extracted features.
            Defaults to config.OUTPUT_DIR / 'features_cache'
        """
        self.cache_dir = Path(cache_dir) if cache_dir else \
            config.OUTPUT_DIR / 'features_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_PYRADIOMICS:
            raise ImportError(
                "PyRadiomics is required. Install with: pip install pyradiomics==3.0.1"
            )

        # Initialise extractor with params
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.disableAllFeatures()

        for feature_class in PYRADIOMICS_PARAMS['featureClass']:
            self.extractor.enableFeatureClassByName(feature_class)

        for key, val in PYRADIOMICS_PARAMS['setting'].items():
            if val is not None:
                self.extractor.settings[key] = val

        print("PyRadiomics extractor initialised")
        print(f"Feature cache directory: {self.cache_dir}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_patient_features(
        self,
        patient_id,
        ct_array,
        gtv_array,
        supervoxel_labels,
        spacing=(1.0, 1.0, 1.0),
        skip_if_cached=True
    ):
        """
        Extract features for one patient (GTV + all supervoxels).

        Parameters
        ----------
        patient_id : str
        ct_array   : np.ndarray  (Z, H, W) — HU values
        gtv_array  : np.ndarray  (Z, H, W) — binary GTV mask
        supervoxel_labels : np.ndarray (Z, H, W) — integer label map (0 = background)
        spacing    : tuple  voxel spacing in mm
        skip_if_cached : bool  load from disk if already extracted

        Returns
        -------
        features : dict
            {
              'gtv'        : np.ndarray  shape (n_features,)
              'supervoxels': np.ndarray  shape (n_sv, n_features)
              'feature_names': list[str]
              'n_supervoxels': int
              'centroids'  : np.ndarray  shape (n_sv, 3)  — voxel coordinates
            }
        """
        cache_file = self.cache_dir / f'{patient_id}_features.npz'

        if skip_if_cached and cache_file.exists():
            print(f"  [CACHE] Loading features for {patient_id}")
            return self._load_from_cache(cache_file)

        print(f"  Extracting features for {patient_id}...")

        # Convert arrays to SimpleITK images
        ct_sitk   = self._array_to_sitk(ct_array,  spacing)
        gtv_sitk  = self._array_to_sitk(gtv_array.astype(np.uint8), spacing)

        # 1. GTV-level features
        gtv_features, feature_names = self._extract_roi_features(ct_sitk, gtv_sitk)

        if gtv_features is None:
            print(f"  WARNING: GTV feature extraction failed for {patient_id}")
            return None

        # 2. Supervoxel-level features
        n_supervoxels  = int(supervoxel_labels.max())
        sv_features    = []
        sv_centroids   = []
        valid_sv_ids   = []

        for sv_id in range(1, n_supervoxels + 1):
            sv_mask = (supervoxel_labels == sv_id).astype(np.uint8)

            if sv_mask.sum() < 8:      # Skip tiny supervoxels
                continue

            sv_sitk = self._array_to_sitk(sv_mask, spacing)
            feats, _ = self._extract_roi_features(ct_sitk, sv_sitk)

            if feats is not None:
                sv_features.append(feats)
                centroid = self._compute_centroid(sv_mask)
                sv_centroids.append(centroid)
                valid_sv_ids.append(sv_id)

        if len(sv_features) == 0:
            print(f"  WARNING: No valid supervoxels for {patient_id}")
            return None

        sv_features_arr  = np.array(sv_features,   dtype=np.float32)
        sv_centroids_arr = np.array(sv_centroids,  dtype=np.float32)
        gtv_features_arr = np.array(gtv_features,  dtype=np.float32)

        # Replace NaN/Inf with 0
        sv_features_arr  = np.nan_to_num(sv_features_arr,  nan=0.0, posinf=0.0, neginf=0.0)
        gtv_features_arr = np.nan_to_num(gtv_features_arr, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'gtv'           : gtv_features_arr,
            'supervoxels'   : sv_features_arr,
            'feature_names' : feature_names,
            'n_supervoxels' : len(sv_features),
            'centroids'     : sv_centroids_arr,
            'valid_sv_ids'  : np.array(valid_sv_ids),
        }

        # Cache to disk
        self._save_to_cache(cache_file, result)
        print(f"  Done: {len(sv_features)} supervoxels, {len(feature_names)} features")

        return result

    def extract_all_patients(
        self,
        patient_ids,
        preprocessed_data_dir,
        skip_existing=True
    ):
        """
        Batch extract features for all patients.

        Parameters
        ----------
        patient_ids : list[str]
        preprocessed_data_dir : Path  — directory with saved preprocessed .npz files
        skip_existing : bool

        Returns
        -------
        results : dict  {patient_id: feature_dict}
        failed  : list[str]  — patient IDs that failed
        """
        results = {}
        failed  = []

        print(f"\nExtracting features for {len(patient_ids)} patients...")

        for patient_id in tqdm(patient_ids, desc='Feature extraction'):
            try:
                # Load preprocessed data (saved by preprocessing.py)
                npz_file = Path(preprocessed_data_dir) / f'{patient_id}_preprocessed.npz'

                if not npz_file.exists():
                    print(f"  WARNING: Preprocessed file not found for {patient_id}")
                    failed.append(patient_id)
                    continue

                data = np.load(npz_file, allow_pickle=True)
                ct_array          = data['ct_array']
                gtv_array         = data['gtv_array']
                supervoxel_labels = data['supervoxel_labels']
                spacing           = tuple(data['spacing']) if 'spacing' in data else (1.0, 1.0, 1.0)

                feats = self.extract_patient_features(
                    patient_id        = patient_id,
                    ct_array          = ct_array,
                    gtv_array         = gtv_array,
                    supervoxel_labels = supervoxel_labels,
                    spacing           = spacing,
                    skip_if_cached    = skip_existing
                )

                if feats is not None:
                    results[patient_id] = feats
                else:
                    failed.append(patient_id)

            except Exception as e:
                print(f"  ERROR for {patient_id}: {e}")
                failed.append(patient_id)

        print(f"\nExtraction complete: {len(results)} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed patients: {failed}")

        return results, failed

    def save_gtv_features_csv(self, results, save_path):
        """
        Save GTV-level features to a CSV (one row per patient).
        Useful for baseline model or comparison.

        Parameters
        ----------
        results   : dict  {patient_id: feature_dict}
        save_path : Path or str
        """
        rows = []
        for pid, feat_dict in results.items():
            row = {'patient_id': pid}
            for i, fname in enumerate(feat_dict['feature_names']):
                row[fname] = feat_dict['gtv'][i]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"GTV features saved to {save_path}  ({len(df)} patients, {len(df.columns)-1} features)")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_roi_features(self, ct_sitk, mask_sitk):
        """
        Run PyRadiomics on a single ROI.

        Returns
        -------
        features     : list[float] or None
        feature_names: list[str]  or []
        """
        try:
            result = self.extractor.execute(ct_sitk, mask_sitk)

            features      = []
            feature_names = []

            for key, val in result.items():
                # Skip diagnostic entries
                if key.startswith('diagnostics_'):
                    continue
                try:
                    features.append(float(val))
                    feature_names.append(key)
                except (TypeError, ValueError):
                    continue

            if len(features) == 0:
                return None, []

            return features, feature_names

        except Exception as e:
            # Log at DEBUG so verbose runs can diagnose failures per-supervoxel
            # without flooding the console during normal batch extraction
            logging.getLogger('radgraph.extractor').debug(
                f"PyRadiomics extraction failed: {type(e).__name__}: {e}"
            )
            return None, []

    @staticmethod
    def _array_to_sitk(array, spacing=(1.0, 1.0, 1.0)):
        """Convert numpy array to SimpleITK image with given spacing."""
        img = sitk.GetImageFromArray(array)
        img.SetSpacing(tuple(float(s) for s in spacing))
        return img

    @staticmethod
    def _compute_centroid(binary_mask):
        """
        Compute voxel centroid of a binary mask.

        Returns
        -------
        centroid : np.ndarray  shape (3,) — (z, y, x) coordinates
        """
        coords = np.argwhere(binary_mask > 0)
        if len(coords) == 0:
            return np.zeros(3, dtype=np.float32)
        return coords.mean(axis=0).astype(np.float32)

    @staticmethod
    def _save_to_cache(cache_file, result):
        """Save feature dict to .npz file."""
        np.savez_compressed(
            cache_file,
            gtv           = result['gtv'],
            supervoxels   = result['supervoxels'],
            centroids     = result['centroids'],
            valid_sv_ids  = result['valid_sv_ids'],
            feature_names = np.array(result['feature_names']),
            n_supervoxels = np.array([result['n_supervoxels']])   # 1-D for safe int() loading
        )

    @staticmethod
    def _load_from_cache(cache_file):
        """Load feature dict from .npz file."""
        data = np.load(cache_file, allow_pickle=True)
        return {
            'gtv'           : data['gtv'],
            'supervoxels'   : data['supervoxels'],
            'centroids'     : data['centroids'],
            'valid_sv_ids'  : data['valid_sv_ids'],
            'feature_names' : list(data['feature_names']),
            'n_supervoxels' : int(data['n_supervoxels'][0]),   # stored as 1-D array
        }


# ─── Preprocessing + Supervoxel save helper ───────────────────────────────────

def preprocess_and_save_all(patient_ids, loader, preprocessor, sv_generator, save_dir):
    """
    Run preprocessing + supervoxel generation and save .npz files.
    These are then consumed by extract_all_patients().

    Parameters
    ----------
    patient_ids  : list[str]
    loader       : HNSCCDataLoader
    preprocessor : CTPreprocessor
    sv_generator : SupervoxelGenerator
    save_dir     : Path  — where to save .npz files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    failed  = []
    saved   = 0
    skipped = 0

    for patient_id in tqdm(patient_ids, desc='Preprocessing + supervoxels'):
        out_file = save_dir / f'{patient_id}_preprocessed.npz'

        if out_file.exists():
            skipped += 1
            continue

        try:
            # Load raw data
            data = loader.load_patient_data(patient_id)
            if data is None or data['gtv_mask'] is None:
                print(f"  Skipping {patient_id}: missing CT or GTV")
                failed.append(patient_id)
                continue

            # Preprocess
            processed = preprocessor.preprocess_patient(
                ct_image  = data['ct_image'],
                gtv_mask  = data['gtv_mask']
            )

            # Generate supervoxels
            sv_labels, _ = sv_generator.generate_supervoxels(
                ct_array    = processed['ct_array'],
                region_mask = processed['region_array']
            )

            # Save
            np.savez_compressed(
                out_file,
                ct_array          = processed['ct_array'].astype(np.float32),
                gtv_array         = processed['gtv_array'].astype(np.uint8),
                region_array      = processed['region_array'].astype(np.uint8),
                supervoxel_labels = sv_labels.astype(np.int32),
                spacing           = np.array(config.TARGET_SPACING, dtype=np.float32)
            )
            saved += 1

        except Exception as e:
            print(f"  ERROR preprocessing {patient_id}: {e}")
            failed.append(patient_id)

    print(f"\nPreprocessing done.")
    print(f"  Newly saved : {saved}")
    print(f"  Skipped     : {skipped}  (already existed)")
    print(f"  Failed      : {len(failed)}")
    return failed


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract radiomic features from CT supervoxels')
    parser.add_argument('--patient_id',    type=str,  default=None,
                        help='Single patient ID to process')
    parser.add_argument('--all_patients',  action='store_true',
                        help='Process all patients in clinical data')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip patients already in cache (default: True)')
    parser.add_argument('--no_skip', action='store_true', default=False,
                        help='Re-extract features even if already cached')
    parser.add_argument('--preprocessed_dir', type=str,
                        default=str(config.OUTPUT_DIR / 'preprocessed'),
                        help='Directory with preprocessed .npz files')
    args = parser.parse_args()

    from data_loader       import HNSCCDataLoader
    from preprocessing     import CTPreprocessor
    from supervoxel_generator import SupervoxelGenerator

    # Initialise components
    loader       = HNSCCDataLoader(config.CT_SCANS_DIR, config.RTSTRUCT_DIR,
                                   config.CLINICAL_DATA_FILE)
    preprocessor = CTPreprocessor(target_spacing=config.TARGET_SPACING)
    sv_gen       = SupervoxelGenerator(n_segments  = config.N_SUPERVOXELS_TARGET,
                                       compactness = config.SLIC_COMPACTNESS,
                                       sigma       = config.SLIC_SIGMA)
    extractor    = SupervoxelFeatureExtractor()

    preprocessed_dir = Path(args.preprocessed_dir)
    # --no_skip overrides the default skip_existing=True
    skip_existing = not args.no_skip

    if args.all_patients:
        patient_ids = loader.filter_patients_by_followup(config.MIN_FOLLOWUP_MONTHS)

        # Step 1: Preprocess + supervoxels
        print("\n=== Step 1: Preprocessing & Supervoxel Generation ===")
        preprocess_and_save_all(patient_ids, loader, preprocessor, sv_gen, preprocessed_dir)

        # Step 2: Feature extraction
        print("\n=== Step 2: Feature Extraction ===")
        results, failed = extractor.extract_all_patients(
            patient_ids           = patient_ids,
            preprocessed_data_dir = preprocessed_dir,
            skip_existing         = skip_existing
        )

        # Save GTV features to CSV (for baseline comparison)
        if results:
            csv_path = config.OUTPUT_DIR / 'gtv_features_extracted.csv'
            extractor.save_gtv_features_csv(results, csv_path)

    elif args.patient_id:
        patient_id = args.patient_id
        print(f"\nProcessing single patient: {patient_id}")

        npz_file = preprocessed_dir / f'{patient_id}_preprocessed.npz'
        if not npz_file.exists():
            print(f"Preprocessed file not found: {npz_file}")
            print("Run preprocessing first.")
            return

        data = np.load(npz_file, allow_pickle=True)
        feats = extractor.extract_patient_features(
            patient_id        = patient_id,
            ct_array          = data['ct_array'],
            gtv_array         = data['gtv_array'],
            supervoxel_labels = data['supervoxel_labels'],
            spacing           = tuple(data['spacing']),
            skip_if_cached    = False
        )

        if feats:
            print(f"\nResults for {patient_id}:")
            print(f"  GTV features shape    : {feats['gtv'].shape}")
            print(f"  Supervoxel features   : {feats['supervoxels'].shape}")
            print(f"  Number of supervoxels : {feats['n_supervoxels']}")
            print(f"  Feature names (first 5): {feats['feature_names'][:5]}")
    else:
        print("Please specify --patient_id <id> or --all_patients")
        parser.print_help()


if __name__ == '__main__':
    main()
