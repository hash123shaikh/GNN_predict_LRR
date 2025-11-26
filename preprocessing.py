"""
Preprocessing module for CT scans
Handles resampling, normalization, and peritumoral region definition
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import config


class CTPreprocessor:
    """
    Preprocess CT scans for RadGraph analysis
    """
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0)):
        """
        Parameters:
        -----------
        target_spacing : tuple
            Target isotropic spacing in mm
        """
        self.target_spacing = target_spacing
    
    def resample_image(self, image, interpolator=sitk.sitkLinear):
        """
        Resample image to target spacing
        
        Parameters:
        -----------
        image : SimpleITK.Image
            Input image
        interpolator : SimpleITK interpolator
            Interpolation method
            
        Returns:
        --------
        resampled : SimpleITK.Image
            Resampled image
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]
        
        # Setup resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(interpolator)
        
        resampled = resampler.Execute(image)
        
        return resampled
    
    def resample_mask(self, mask):
        """
        Resample binary mask using nearest neighbor interpolation
        
        Parameters:
        -----------
        mask : SimpleITK.Image
            Binary mask
            
        Returns:
        --------
        resampled_mask : SimpleITK.Image
            Resampled mask
        """
        return self.resample_image(mask, interpolator=sitk.sitkNearestNeighbor)
    
    def normalize_intensity(self, ct_array, clip_range=(-1000, 400)):
        """
        Normalize CT intensity values
        
        Parameters:
        -----------
        ct_array : numpy array
            CT image as array
        clip_range : tuple
            (min, max) Hounsfield units to clip
            
        Returns:
        --------
        normalized : numpy array
            Normalized CT array
        """
        # Clip to range
        ct_clipped = np.clip(ct_array, clip_range[0], clip_range[1])
        
        # Normalize to [0, 1]
        normalized = (ct_clipped - clip_range[0]) / (clip_range[1] - clip_range[0])
        
        return normalized
    
    def define_peritumoral_region(self, gtv_mask, margin_mm=50):
        """
        Define peritumoral region around GTV
        
        Parameters:
        -----------
        gtv_mask : SimpleITK.Image
            Binary GTV mask
        margin_mm : float
            Margin in mm (5cm = 50mm as in paper)
            
        Returns:
        --------
        region_mask : SimpleITK.Image
            Binary mask of peritumoral region (including GTV)
        bbox : tuple
            Bounding box coordinates (x_start, y_start, z_start, x_size, y_size, z_size)
        """
        # Get GTV bounding box
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(gtv_mask)
        
        if stats.GetNumberOfLabels() == 0:
            raise ValueError("GTV mask is empty")
        
        bbox = stats.GetBoundingBox(1)  # Label 1
        
        # Get spacing
        spacing = gtv_mask.GetSpacing()
        
        # Calculate margin in voxels for each dimension
        margin_voxels = [int(np.ceil(margin_mm / spacing[i])) for i in range(3)]
        
        # Expand bounding box
        size = gtv_mask.GetSize()
        
        x_start = max(0, bbox[0] - margin_voxels[0])
        y_start = max(0, bbox[1] - margin_voxels[1])
        z_start = max(0, bbox[2] - margin_voxels[2])
        
        x_end = min(size[0], bbox[0] + bbox[3] + margin_voxels[0])
        y_end = min(size[1], bbox[1] + bbox[4] + margin_voxels[1])
        z_end = min(size[2], bbox[2] + bbox[5] + margin_voxels[2])
        
        # Create region mask
        region_array = np.zeros(sitk.GetArrayFromImage(gtv_mask).shape, dtype=np.uint8)
        region_array[z_start:z_end, y_start:y_end, x_start:x_end] = 1
        
        region_mask = sitk.GetImageFromArray(region_array)
        region_mask.CopyInformation(gtv_mask)
        
        expanded_bbox = (x_start, y_start, z_start, 
                        x_end - x_start, y_end - y_start, z_end - z_start)
        
        return region_mask, expanded_bbox
    
    def crop_to_region(self, image, bbox):
        """
        Crop image to bounding box
        
        Parameters:
        -----------
        image : SimpleITK.Image
            Image to crop
        bbox : tuple
            Bounding box (x_start, y_start, z_start, x_size, y_size, z_size)
            
        Returns:
        --------
        cropped : SimpleITK.Image
            Cropped image
        """
        cropped = sitk.RegionOfInterest(
            image,
            size=[bbox[3], bbox[4], bbox[5]],
            index=[bbox[0], bbox[1], bbox[2]]
        )
        
        return cropped
    
    def preprocess_patient(self, ct_image, gtv_mask):
        """
        Complete preprocessing pipeline for one patient
        
        Parameters:
        -----------
        ct_image : SimpleITK.Image
            Original CT image
        gtv_mask : SimpleITK.Image
            Original GTV mask
            
        Returns:
        --------
        processed_data : dict
            Dictionary with preprocessed data
        """
        print(f"Original CT size: {ct_image.GetSize()}, spacing: {ct_image.GetSpacing()}")
        
        # Step 1: Resample to isotropic spacing
        ct_resampled = self.resample_image(ct_image)
        gtv_resampled = self.resample_mask(gtv_mask)
        
        print(f"Resampled CT size: {ct_resampled.GetSize()}, spacing: {ct_resampled.GetSpacing()}")
        
        # Step 2: Define peritumoral region
        region_mask, bbox = self.define_peritumoral_region(
            gtv_resampled, 
            margin_mm=config.PERITUMORAL_MARGIN_MM
        )
        
        print(f"Peritumoral region bbox: {bbox}")
        
        # Step 3: Crop to region (optional - saves memory)
        # ct_cropped = self.crop_to_region(ct_resampled, bbox)
        # gtv_cropped = self.crop_to_region(gtv_resampled, bbox)
        # region_cropped = self.crop_to_region(region_mask, bbox)
        
        # For now, keep full images
        ct_cropped = ct_resampled
        gtv_cropped = gtv_resampled
        region_cropped = region_mask
        
        # Convert to numpy arrays
        ct_array = sitk.GetArrayFromImage(ct_cropped)
        gtv_array = sitk.GetArrayFromImage(gtv_cropped)
        region_array = sitk.GetArrayFromImage(region_cropped)
        
        # Normalize CT intensity
        ct_normalized = self.normalize_intensity(ct_array)
        
        processed_data = {
            'ct_image': ct_cropped,
            'gtv_mask': gtv_cropped,
            'region_mask': region_cropped,
            'ct_array': ct_array,  # Original HU values
            'ct_normalized': ct_normalized,  # Normalized [0,1]
            'gtv_array': gtv_array.astype(np.uint8),
            'region_array': region_array.astype(np.uint8),
            'bbox': bbox
        }
        
        return processed_data


def test_preprocessing():
    """Test preprocessing module"""
    from data_loader import HNSCCDataLoader
    
    print("Testing preprocessing...")
    
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
    
    print(f"\nProcessed data:")
    print(f"CT array shape: {processed['ct_array'].shape}")
    print(f"CT normalized range: [{processed['ct_normalized'].min():.3f}, {processed['ct_normalized'].max():.3f}]")
    print(f"GTV voxels: {processed['gtv_array'].sum()}")
    print(f"Region voxels: {processed['region_array'].sum()}")
    
    print("\nPreprocessing test successful!")


if __name__ == '__main__':
    test_preprocessing()
