"""
Data loading module for RadGraph
Handles loading CT scans, RT structures, and clinical data
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from rt_utils import RTStructBuilder
    HAS_RT_UTILS = True
except:
    HAS_RT_UTILS = False
    print("Warning: rt-utils not installed. Will try alternative GTV extraction methods.")

import config


class HNSCCDataLoader:
    """
    Data loader for HNSCC CT scans and clinical data
    """
    
    def __init__(self, ct_dir, rtstruct_dir, clinical_file):
        """
        Parameters:
        -----------
        ct_dir : Path or str
            Directory containing CT scan folders (one folder per patient)
        rtstruct_dir : Path or str
            Directory containing RT structure files
        clinical_file : Path or str
            CSV file with clinical data
        """
        self.ct_dir = Path(ct_dir)
        self.rtstruct_dir = Path(rtstruct_dir)
        self.clinical_file = Path(clinical_file)
        
        # Load clinical data
        self.clinical_data = pd.read_csv(self.clinical_file)
        print(f"Loaded clinical data for {len(self.clinical_data)} patients")
        
    def get_patient_list(self):
        """
        Get list of patient IDs
        
        Returns:
        --------
        patient_ids : list
            List of patient IDs
        """
        return self.clinical_data[config.PATIENT_ID_COL].tolist()
    
    def load_ct_scan(self, patient_id):
        """
        Load CT scan for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        ct_image : SimpleITK.Image
            CT image
        """
        # Find patient CT directory
        patient_ct_dir = self.ct_dir / str(patient_id)
        
        if not patient_ct_dir.exists():
            # Try with different naming conventions
            patient_ct_dir = self.ct_dir / f"Patient_{patient_id}"
            if not patient_ct_dir.exists():
                patient_ct_dir = self.ct_dir / f"patient_{patient_id}"
        
        if not patient_ct_dir.exists():
            raise ValueError(f"CT directory not found for patient {patient_id}")
        
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(patient_ct_dir))
        
        if len(dicom_names) == 0:
            raise ValueError(f"No DICOM files found in {patient_ct_dir}")
        
        reader.SetFileNames(dicom_names)
        ct_image = reader.Execute()
        
        return ct_image
    
    def load_rtstruct(self, patient_id):
        """
        Load RT structure set for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        rtstruct_file : Path
            Path to RT structure file
        """
        # Find RT structure file
        possible_names = [
            f"{patient_id}.dcm",
            f"RS.{patient_id}.dcm",
            f"RTSTRUCT_{patient_id}.dcm",
            f"rtstruct_{patient_id}.dcm",
            f"{patient_id}_rtstruct.dcm"
        ]
        
        for name in possible_names:
            rtstruct_file = self.rtstruct_dir / name
            if rtstruct_file.exists():
                return rtstruct_file
        
        # If not found, search in subdirectories
        for rtstruct_file in self.rtstruct_dir.rglob(f"*{patient_id}*.dcm"):
            ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
            if ds.Modality == 'RTSTRUCT':
                return rtstruct_file
        
        raise ValueError(f"RT structure file not found for patient {patient_id}")
    
    def extract_gtv_mask(self, patient_id, ct_image):
        """
        Extract GTV mask from RT structure
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
        ct_image : SimpleITK.Image
            CT image
            
        Returns:
        --------
        gtv_mask : SimpleITK.Image
            Binary mask of GTV
        gtv_name : str
            Name of the GTV contour found
        """
        try:
            rtstruct_file = self.load_rtstruct(patient_id)
        except ValueError as e:
            print(f"Warning: {e}")
            return None, None
        
        # Try using rt-utils if available
        if HAS_RT_UTILS:
            try:
                # Get CT directory
                patient_ct_dir = self.ct_dir / str(patient_id)
                
                # Load RT structure
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=str(patient_ct_dir),
                    rt_struct_path=str(rtstruct_file)
                )
                
                # Get ROI names
                roi_names = rtstruct.get_roi_names()
                
                # Find GTV contour
                gtv_name = None
                for name_pattern in config.GTV_NAMES:
                    for roi_name in roi_names:
                        if name_pattern.lower() in roi_name.lower():
                            gtv_name = roi_name
                            break
                    if gtv_name:
                        break
                
                if gtv_name is None:
                    print(f"Warning: No GTV contour found for patient {patient_id}")
                    print(f"Available ROIs: {roi_names}")
                    return None, None
                
                # Get mask as numpy array
                mask_array = rtstruct.get_roi_mask_by_name(gtv_name)
                
                # Convert to SimpleITK image
                gtv_mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
                gtv_mask.CopyInformation(ct_image)
                
                return gtv_mask, gtv_name
                
            except Exception as e:
                print(f"Error using rt-utils for patient {patient_id}: {e}")
                return None, None
        
        else:
            # Alternative method using pydicom
            try:
                gtv_mask, gtv_name = self._extract_gtv_pydicom(rtstruct_file, ct_image)
                return gtv_mask, gtv_name
            except Exception as e:
                print(f"Error extracting GTV for patient {patient_id}: {e}")
                return None, None
    
    def _extract_gtv_pydicom(self, rtstruct_file, ct_image):
        """
        Extract GTV using pydicom (fallback method)
        This is a simplified implementation - you may need to adapt based on your data
        """
        ds = pydicom.dcmread(rtstruct_file)
        
        # Find GTV structure
        gtv_name = None
        gtv_contour = None
        
        for roi_seq in ds.StructureSetROISequence:
            roi_name = roi_seq.ROIName
            for name_pattern in config.GTV_NAMES:
                if name_pattern.lower() in roi_name.lower():
                    gtv_name = roi_name
                    roi_number = roi_seq.ROINumber
                    
                    # Find corresponding contour
                    for contour_seq in ds.ROIContourSequence:
                        if contour_seq.ReferencedROINumber == roi_number:
                            gtv_contour = contour_seq
                            break
                    break
            if gtv_name:
                break
        
        if gtv_name is None:
            print(f"No GTV found in RT structure")
            return None, None
        
        # For simplicity, this returns None and recommends using rt-utils
        print(f"Found GTV: {gtv_name}")
        print("Note: Complete GTV extraction requires rt-utils package")
        print("Please install: pip install rt-utils")
        
        return None, gtv_name
    
    def get_clinical_features(self, patient_id):
        """
        Get clinical features for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        features : dict
            Dictionary of clinical features
        """
        patient_row = self.clinical_data[
            self.clinical_data[config.PATIENT_ID_COL] == patient_id
        ]
        
        if len(patient_row) == 0:
            raise ValueError(f"Patient {patient_id} not found in clinical data")
        
        patient_row = patient_row.iloc[0]
        
        # Extract clinical features
        features = {}
        for feat_name in config.CLINICAL_FEATURES:
            if feat_name in patient_row:
                features[feat_name] = patient_row[feat_name]
            else:
                print(f"Warning: Feature {feat_name} not found for patient {patient_id}")
                features[feat_name] = 0  # Default value
        
        return features
    
    def get_outcome(self, patient_id, task='LR'):
        """
        Get outcome label for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
        task : str
            'LR' for locoregional recurrence or 'DM' for distant metastasis
            
        Returns:
        --------
        outcome : int
            0 or 1
        """
        patient_row = self.clinical_data[
            self.clinical_data[config.PATIENT_ID_COL] == patient_id
        ]
        
        if len(patient_row) == 0:
            raise ValueError(f"Patient {patient_id} not found in clinical data")
        
        patient_row = patient_row.iloc[0]
        
        outcome_col = config.get_outcome_column(task)
        
        if outcome_col not in patient_row:
            raise ValueError(f"Outcome column {outcome_col} not found")
        
        return int(patient_row[outcome_col])
    
    def get_followup_time(self, patient_id):
        """
        Get follow-up time for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        followup : float
            Follow-up time in months
        """
        patient_row = self.clinical_data[
            self.clinical_data[config.PATIENT_ID_COL] == patient_id
        ]
        
        if len(patient_row) == 0:
            return 0
        
        patient_row = patient_row.iloc[0]
        
        if config.FOLLOWUP_TIME in patient_row:
            return float(patient_row[config.FOLLOWUP_TIME])
        else:
            return 999  # Assume adequate follow-up if not specified
    
    def filter_patients_by_followup(self, min_followup_months=24):
        """
        Filter patients with adequate follow-up
        
        Parameters:
        -----------
        min_followup_months : int
            Minimum follow-up in months
            
        Returns:
        --------
        valid_patients : list
            List of patient IDs with adequate follow-up
        """
        all_patients = self.get_patient_list()
        valid_patients = []
        
        for patient_id in all_patients:
            followup = self.get_followup_time(patient_id)
            if followup >= min_followup_months:
                valid_patients.append(patient_id)
        
        print(f"Patients with >={min_followup_months} months follow-up: {len(valid_patients)}/{len(all_patients)}")
        
        return valid_patients
    
    def load_patient_data(self, patient_id):
        """
        Load complete data for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        data : dict
            Dictionary with 'ct_image', 'gtv_mask', 'clinical_features', 'gtv_name'
        """
        try:
            # Load CT
            ct_image = self.load_ct_scan(patient_id)
            
            # Extract GTV
            gtv_mask, gtv_name = self.extract_gtv_mask(patient_id, ct_image)
            
            # Get clinical features
            clinical_features = self.get_clinical_features(patient_id)
            
            data = {
                'patient_id': patient_id,
                'ct_image': ct_image,
                'gtv_mask': gtv_mask,
                'gtv_name': gtv_name,
                'clinical_features': clinical_features
            }
            
            return data
            
        except Exception as e:
            print(f"Error loading data for patient {patient_id}: {e}")
            return None


def test_data_loader():
    """Test the data loader"""
    print("Testing data loader...")
    
    loader = HNSCCDataLoader(
        ct_dir=config.CT_SCANS_DIR,
        rtstruct_dir=config.RTSTRUCT_DIR,
        clinical_file=config.CLINICAL_DATA_FILE
    )
    
    # Get patient list
    patients = loader.get_patient_list()
    print(f"Total patients: {len(patients)}")
    
    # Filter by follow-up
    valid_patients = loader.filter_patients_by_followup(config.MIN_FOLLOWUP_MONTHS)
    
    if len(valid_patients) > 0:
        # Try loading first patient
        test_patient = valid_patients[0]
        print(f"\nTesting with patient: {test_patient}")
        
        data = loader.load_patient_data(test_patient)
        
        if data is not None:
            print(f"CT image size: {data['ct_image'].GetSize()}")
            print(f"CT spacing: {data['ct_image'].GetSpacing()}")
            if data['gtv_mask'] is not None:
                print(f"GTV mask size: {data['gtv_mask'].GetSize()}")
                print(f"GTV contour: {data['gtv_name']}")
            print(f"Clinical features: {data['clinical_features']}")
            
            print("\nData loader test successful!")
        else:
            print("Failed to load patient data")
    else:
        print("No valid patients found")


if __name__ == '__main__':
    test_data_loader()
