"""
Supervoxel generation using SLIC (Simple Linear Iterative Clustering)
"""

import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
import SimpleITK as sitk
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
    
    def generate_supervoxels(self, ct_array, region_mask):
        """
        Generate supervoxels in peritumoral region
        
        Parameters:
        -----------
        ct_array : numpy array
            CT image (H x W x D)
        region_mask : numpy array
            Binary mask of peritumoral region (H x W x D)
            
        Returns:
        --------
        supervoxel_labels : numpy array
            Label map where each voxel is assigned to a supervoxel ID
            Shape: (H x W x D), values: 0 (background) or 1 to n_supervoxels
        n_supervoxels : int
            Actual number of supervoxels generated
        """
        print(f"Generating supervoxels with SLIC...")
        print(f"  Target segments: {self.n_segments}")
        print(f"  Compactness: {self.compactness}")
        print(f"  Sigma: {self.sigma}")
        
        # Normalize CT for SLIC (expects [0, 1] or similar range)
        ct_normalized = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min() + 1e-6)
        
        # Apply SLIC only in the region of interest
        # Mask out regions outside peritumoral area
        ct_masked = ct_normalized * region_mask
        
        # Run SLIC
        try:
            labels = slic(
                ct_masked,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                multichannel=False,
                enforce_connectivity=True,
                max_num_iter=config.SLIC_MAX_ITER,
                start_label=1  # Start labels from 1 (0 = background)
            )
        except Exception as e:
            print(f"Error in SLIC: {e}")
            # Fallback: simpler parameters
            labels = slic(
                ct_masked,
                n_segments=self.n_segments,
                compactness=10,
                multichannel=False
            )
        
        # Mask out supervoxels outside region
        labels = labels * region_mask
        
        # Renumber labels to be contiguous (1, 2, 3, ...)
        labels = self._renumber_labels(labels)
        
        n_supervoxels = labels.max()
        
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
