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
except ImportError:
    HAS_RT_UTILS = False
    print("Warning: rt-utils not installed. Will try alternative GTV extraction methods.")
    print("Install with: pip install rt-utils==1.2.7")

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
        
        # Fallback: search subdirectories
        # Check filename for common RTSTRUCT prefixes before opening (faster)
        for rtstruct_file in self.rtstruct_dir.rglob(f"*{patient_id}*.dcm"):
            fname_lower = rtstruct_file.name.lower()
            # Quick filename heuristic — avoids opening every DICOM file
            if any(hint in fname_lower for hint in ('rs', 'rtstruct', 'struct')):
                try:
                    ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
                    if ds.Modality == 'RTSTRUCT':
                        return rtstruct_file
                except Exception:
                    continue

        # Last resort: open every file matching patient_id
        for rtstruct_file in self.rtstruct_dir.rglob(f"*{patient_id}*.dcm"):
            try:
                ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
                if ds.Modality == 'RTSTRUCT':
                    return rtstruct_file
            except Exception:
                continue
        
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
                # Resolve CT directory using the same naming conventions as
                # load_ct_scan (Bug fix: was hardcoded to str(patient_id) only)
                patient_ct_dir = self.ct_dir / str(patient_id)
                if not patient_ct_dir.exists():
                    patient_ct_dir = self.ct_dir / f"Patient_{patient_id}"
                if not patient_ct_dir.exists():
                    patient_ct_dir = self.ct_dir / f"patient_{patient_id}"
                if not patient_ct_dir.exists():
                    raise ValueError(
                        f"CT directory not found for patient {patient_id}"
                    )
                
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
        Fallback GTV extraction using pydicom only.

        NOTE: Full contour-to-mask rasterisation requires rt-utils.
        This fallback finds the GTV ROI name but cannot build the binary
        mask without rt-utils. It raises a clear RuntimeError so the
        caller knows exactly what is missing rather than silently
        returning None and causing a cryptic failure downstream.

        Install rt-utils to enable full GTV extraction:
            pip install rt-utils==1.2.7
        """
        ds = pydicom.dcmread(rtstruct_file)

        # Search for GTV ROI name
        gtv_name = None
        for roi_seq in ds.StructureSetROISequence:
            roi_name = roi_seq.ROIName
            for name_pattern in config.GTV_NAMES:
                if name_pattern.lower() in roi_name.lower():
                    gtv_name = roi_name
                    break
            if gtv_name:
                break

        if gtv_name is None:
            print("No GTV found in RT structure.")
            print(f"Available ROIs: {[r.ROIName for r in ds.StructureSetROISequence]}")
            return None, None

        # GTV name found but mask cannot be built without rt-utils
        raise RuntimeError(
            f"GTV contour '{gtv_name}' found but mask extraction requires rt-utils.\n"
            f"Install with: pip install rt-utils==1.2.7\n"
            f"Then re-run the pipeline."
        )
    
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
            val = patient_row[config.FOLLOWUP_TIME]
            # Return 0 for NaN/missing so patient is correctly excluded
            # by the follow-up filter (Bug fix: was returning 999)
            return float(val) if pd.notna(val) else 0.0
        else:
            return 0.0  # No follow-up data → exclude from analysis
    
    def filter_patients_by_followup(self, min_followup_months=24):
        """
        Filter patients with adequate follow-up.

        Vectorised — single DataFrame operation instead of per-patient loop.

        Parameters
        ----------
        min_followup_months : int

        Returns
        -------
        valid_patients : list[str]
        """
        all_patients = self.get_patient_list()
        n_total      = len(all_patients)

        if config.FOLLOWUP_TIME not in self.clinical_data.columns:
            # No follow-up column — return all patients with a warning
            print(f"Warning: '{config.FOLLOWUP_TIME}' column not found. "
                  f"Returning all {n_total} patients.")
            return all_patients

        # Vectorised filter — one pass over the DataFrame
        mask = (
            self.clinical_data[config.FOLLOWUP_TIME]
            .fillna(0)                          # Missing → 0 → excluded (Bug 4 fix)
            >= min_followup_months
        )
        valid_ids = self.clinical_data.loc[
            mask, config.PATIENT_ID_COL
        ].tolist()

        print(f"Patients with >={min_followup_months} months follow-up: "
              f"{len(valid_ids)}/{n_total}")

        return valid_ids
    
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
