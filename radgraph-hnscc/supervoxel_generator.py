"""
Supervoxel generation using SLIC (Simple Linear Iterative Clustering)
"""

import numpy as np
from skimage.segmentation import slic
import config


class SupervoxelGenerator:
    """
    Generate supervoxels using SLIC algorithm
    """
    
    def __init__(self, n_segments=100, compactness=10, sigma=1):
        """
        Parameters:
        -----------
        n_segments : int
            Target number of supervoxels
        compactness : float
            Balance between spatial and intensity similarity
            Higher = more compact/spherical supervoxels
        sigma : float
            Gaussian smoothing before segmentation
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
    
    def generate_supervoxels(self, ct_array, region_mask, gtv_array=None):
        """
        Generate supervoxels in peritumoral region only.

        Incorporates Joseph's RadiomicsSupervoxels.py approach:
          - GTV voxels are EXCLUDED from the SLIC region before running.
            The GTV is treated as its own separate node, not segmented
            into supervoxels. This means supervoxels only cover peritumoral
            tissue, matching the paper's intent.
          - slic_zero=True implements the zero-parameter SLIC described
            in the paper title. Compactness is automatically adapted
            per-region rather than being a fixed global value.

        Parameters:
        -----------
        ct_array : numpy array  (Z, H, W) — CT HU values
        region_mask : numpy array  (Z, H, W) — binary peritumoral region mask
        gtv_array : numpy array or None  (Z, H, W) — binary GTV mask
            If provided, GTV voxels are removed from the SLIC region so
            supervoxels are purely peritumoral (Joseph's approach).

        Returns:
        --------
        supervoxel_labels : numpy array
            Label map: 0 = background, 1..n = peritumoral supervoxels.
            GTV voxels are set to 0 (not labelled as supervoxels).
        n_supervoxels : int
        """
        print(f"Generating supervoxels with SLIC (slic_zero=True)...")
        print(f"  Target segments : {self.n_segments}")
        print(f"  GTV excluded    : {gtv_array is not None}")
        
        # ── Build peritumoral mask (exclude GTV) ─────────────────────────────
        # Joseph: boundingbox[mask_array>0] = 0
        # Supervoxels should only cover peritumoral tissue, not the tumour itself
        peritumoral_mask = region_mask.copy()
        if gtv_array is not None:
            peritumoral_mask[gtv_array > 0] = 0

        # Normalise CT within the peritumoral region only
        ct_in_region = ct_array[peritumoral_mask > 0]
        if ct_in_region.size == 0:
            raise ValueError("Peritumoral mask is empty — no voxels to process")
        region_min    = ct_in_region.min()
        region_max    = ct_in_region.max()
        ct_normalized = (ct_array - region_min) / (region_max - region_min + 1e-6)
        ct_normalized = np.clip(ct_normalized, 0.0, 1.0)

        # Run SLIC with mask= parameter (Joseph's approach)
        # slic_zero=True  — zero-parameter SLIC: compactness auto-adapted per region
        # mask=           — restricts SLIC to the peritumoral region (GTV excluded)
        try:
            labels = slic(
                ct_normalized,
                mask              = peritumoral_mask,  # GTV already excluded
                n_segments        = self.n_segments,
                slic_zero         = True,              # zero-parameter SLIC (paper title)
                channel_axis      = None,
                enforce_connectivity = True,
                start_label       = 1
            )
        except Exception as e:
            print(f"Error in SLIC: {e}")
            labels = slic(
                ct_normalized,
                mask         = peritumoral_mask,
                n_segments   = self.n_segments,
                slic_zero    = True,
                channel_axis = None,
                start_label  = 1
            )

        # Keep only labels within the peritumoral region
        labels = labels * peritumoral_mask
        
        # Renumber labels to be contiguous (1, 2, 3, ...)
        labels = self._renumber_labels(labels)
        
        n_supervoxels = int(labels.max())   # convert numpy scalar to Python int

        print(f"  Generated {n_supervoxels} supervoxels")
        
        # Sanity check
        if n_supervoxels < 20:
            print(f"Warning: Only {n_supervoxels} supervoxels generated (target: {self.n_segments})")
            print("Consider adjusting n_segments or compactness parameters")
        
        return labels, n_supervoxels
    
    def _renumber_labels(self, labels):
        """
        Renumber labels to be contiguous starting from 1
        
        Parameters:
        -----------
        labels : numpy array
            Label map
            
        Returns:
        --------
        renumbered : numpy array
            Renumbered label map
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
        
        renumbered = np.zeros_like(labels)
        
        for new_label, old_label in enumerate(unique_labels, start=1):
            renumbered[labels == old_label] = new_label
        
        return renumbered
    
    def get_supervoxel_properties(self, labels, ct_array):
        """
        Get properties of each supervoxel
        
        Parameters:
        -----------
        labels : numpy array
            Supervoxel label map
        ct_array : numpy array
            CT image
            
        Returns:
        --------
        properties : list of dict
            List of supervoxel properties
        """
        properties = []
        n_supervoxels = labels.max()
        
        for sv_id in range(1, n_supervoxels + 1):
            mask = (labels == sv_id)
            n_voxels = mask.sum()
            
            if n_voxels == 0:
                continue
            
            # Get CT values in this supervoxel
            sv_intensities = ct_array[mask]
            
            # Calculate properties
            prop = {
                'id': sv_id,
                'n_voxels': n_voxels,
                'mean_intensity': sv_intensities.mean(),
                'std_intensity': sv_intensities.std(),
                'min_intensity': sv_intensities.min(),
                'max_intensity': sv_intensities.max()
            }
            
            properties.append(prop)
        
        return properties
    
    def visualize_supervoxels(self, labels, slice_idx=None):
        """
        Visualize supervoxels on a slice
        
        Parameters:
        -----------
        labels : numpy array
            Supervoxel label map (Z x H x W)
        slice_idx : int, optional
            Slice index to visualize (if None, use middle slice)
        """
        import matplotlib.pyplot as plt
        
        if slice_idx is None:
            slice_idx = labels.shape[0] // 2
        
        plt.figure(figsize=(10, 10))
        plt.imshow(labels[slice_idx], cmap='nipy_spectral')
        plt.colorbar(label='Supervoxel ID')
        plt.title(f'Supervoxels (Slice {slice_idx})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def test_supervoxel_generation():
    """Test supervoxel generation"""
    from data_loader import HNSCCDataLoader
    from preprocessing import CTPreprocessor
    
    print("Testing supervoxel generation...")
    
    # Load data
    loader = HNSCCDataLoader(
        ct_dir=config.CT_SCANS_DIR,
        rtstruct_dir=config.RTSTRUCT_DIR,
        clinical_file=config.CLINICAL_DATA_FILE
    )
    
    patients = loader.filter_patients_by_followup(config.MIN_FOLLOWUP_MONTHS)
    
    if len(patients) == 0:
        print("No valid patients found")
        return
    
    # Test with first patient
    patient_id = patients[0]
    print(f"\nTesting with patient: {patient_id}")
    
    data = loader.load_patient_data(patient_id)
    
    if data is None or data['gtv_mask'] is None:
        print("Failed to load patient data")
        return
    
    # Preprocess
    preprocessor = CTPreprocessor(target_spacing=config.TARGET_SPACING)
    processed = preprocessor.preprocess_patient(
        ct_image=data['ct_image'],
        gtv_mask=data['gtv_mask']
    )
    
    # Generate supervoxels
    sv_generator = SupervoxelGenerator(
        n_segments=config.N_SUPERVOXELS_TARGET,
        compactness=config.SLIC_COMPACTNESS,
        sigma=config.SLIC_SIGMA
    )
    
    supervoxel_labels, n_supervoxels = sv_generator.generate_supervoxels(
        ct_array=processed['ct_array'],
        region_mask=processed['region_array']
    )
    
    # Get properties
    properties = sv_generator.get_supervoxel_properties(
        labels=supervoxel_labels,
        ct_array=processed['ct_array']
    )
    
    print(f"\nSupervoxel statistics:")
    print(f"  Total supervoxels: {len(properties)}")
    if len(properties) > 0:
        voxel_counts = [p['n_voxels'] for p in properties]
        print(f"  Voxels per supervoxel: {np.mean(voxel_counts):.1f} ± {np.std(voxel_counts):.1f}")
        print(f"  Range: {np.min(voxel_counts)} - {np.max(voxel_counts)}")
    
    # Visualize (optional)
    # sv_generator.visualize_supervoxels(supervoxel_labels)
    
    print("\nSupervoxel generation test successful!")


if __name__ == '__main__':
    test_supervoxel_generation()