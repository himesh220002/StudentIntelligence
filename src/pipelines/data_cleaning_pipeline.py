import pandas as pd
import sys
import os
import logging
from typing import List, Dict, Optional, Literal

# Ensure imports work regardless of execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.data_ingestion import DataIngestor
from preprocessing.issue_identifier import IssueIdentifier
from preprocessing.missing_value_handler import MissingValueHandler
from preprocessing.duplicate_handler import DuplicateHandler
from preprocessing.inconsistency_fixer import InconsistencyFixer
from preprocessing.outlier_detector import OutlierDetector
from preprocessing.feature_scaler import FeatureScaler

# Configure basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaningPipeline:
    """
    End-to-end automated Data Cleaning Pipeline.
    Orchestrates ingestion, identification, cleaning, and scaling
    using the production-grade preprocessing modules.
    """
    
    def __init__(self, raw_data_dir: str = "../data/raw", processed_data_dir: str = "../data/processed"):
        self.ingestor = DataIngestor(raw_data_dir=raw_data_dir)
        self.processed_data_dir = processed_data_dir
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
            
    def run_pipeline(self, 
                     filename: str, 
                     categorical_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                     date_columns: Optional[List[str]] = None,
                     scale_method: Literal['minmax', 'standard'] = 'standard',
                     exclude_scale_columns: Optional[List[str]] = None,
                     output_filename: str = 'cleaned_data.csv') -> Dict[str, pd.DataFrame]:
        """
        Executes the full cleaning pipeline on a given file and returns branched ML/Analyst datasets.
        """
        logger.info(f"Starting pipeline for file: {filename}")
        
        # 1. Ingestion
        df = self.ingestor.load_data(filename)
        logger.info(f"Data ingested successfully. Shape: {df.shape}")
        
        # 2. Identify Initial Issues
        logger.info("Generating initial issue report...")
        initial_issues = IssueIdentifier(df).generate_report()
        # In a real app, you might halt here if issues exceed a threshold
        
        # 3. Handle Duplicates
        logger.info("Dropping exact duplicates...")
        dup_handler = DuplicateHandler(df)
        df = dup_handler.drop_exact_duplicates()
        
        # 4. Handle Missing Values
        logger.info("Imputing missing values...")
        miss_handler = MissingValueHandler(df)
        miss_handler.impute_numerical(strategy='median')
        # Tag: VT_missing_value_handling
        miss_handler.impute_categorical(strategy='constant', fill_value='Unknown')
        df = miss_handler.df
        
        # 5. Fix Inconsistencies
        logger.info("Fixing inconsistencies and formatting...")
        fixer = InconsistencyFixer(df)
        fixer.standardize_strings(keep_original=True)
        if date_columns:
            fixer.standardize_dates(date_columns)
        if categorical_mappings:
            for col, mapping in categorical_mappings.items():
                fixer.map_categories(col, mapping)
        df = fixer.df
        
        # 6. Handle Outliers
        logger.info("Capping outliers using IQR...")
        outlier_detector = OutlierDetector(df)
        outlier_detector.handle_outliers_iqr(action='cap')
        df = outlier_detector.df
        
        # 7. Feature Scaling
        logger.info(f"Scaling numeric features using {scale_method} method...")
        scaler = FeatureScaler(df)
        df = scaler.scale_features(method=scale_method, keep_original=True, exclude_columns=exclude_scale_columns)
        
        # 8. Branch Outputs (Tag: UAT_dual_output_validation)
        logger.info("Branching outputs into ML-ready and Analyst-ready datasets...")
        
        # ML-Ready: Drop _raw and _original
        ml_cols = [c for c in df.columns if not c.endswith('_raw') and not c.endswith('_original')]
        df_ml = df[ml_cols].copy()
        df_ml.columns = [c.replace('_clean', '').replace('_scaled', '') for c in df_ml.columns]
        
        # Analyst-Ready: Drop _clean and _scaled
        analyst_cols = [c for c in df.columns if not c.endswith('_clean') and not c.endswith('_scaled')]
        df_analyst = df[analyst_cols].copy()
        df_analyst.columns = [c.replace('_original', '').replace('_raw', '') for c in df_analyst.columns]
        
        # 9. Smart-Ready Branch
        logger.info("Building Smart-Ready dataset with anomaly fixing...")
        df_smart = df_analyst.copy()
        
        for col in df_smart.select_dtypes(include=['object', 'string']).columns.tolist(): # type: ignore
            # 1. Impute missing with mode instead of Unknown
            valid_vals = df_smart[col][df_smart[col] != 'Unknown']
            if not valid_vals.empty:
                mode_val = valid_vals.mode()[0]
                df_smart[col] = df_smart[col].replace('Unknown', mode_val)
            
            # 2. Strip Noise (#, _)
            df_smart[col] = df_smart[col].astype(str).str.replace(r'[#_]+', '', regex=True).str.strip()
            
            # 3. Fix Typo Anomalies (Rare categories matching common ones)
            val_counts = df_smart[col].value_counts()
            rares = val_counts[val_counts < 3].index
            commons = val_counts[val_counts >= 3].index
            for rare in rares:
                for common in commons:
                    if str(rare).startswith(str(common)) or str(common).startswith(str(rare)):
                        df_smart[col] = df_smart[col].replace(rare, common)
                        break

        # 4. Rearrange columns logically
        cols = list(df_smart.columns)
        id_cols = [c for c in cols if 'id' in c.lower()]
        obj_cols = [c for c in cols if df_smart[c].dtype == 'object' or df_smart[c].dtype.name == 'string']
        num_cols = [c for c in cols if c not in id_cols and c not in obj_cols]
        obj_cols = [c for c in obj_cols if c not in id_cols]
        new_order = id_cols + obj_cols + num_cols
        df_smart = df_smart[new_order]

        # 10. Save Processed Data
        ml_output_path = os.path.join(self.processed_data_dir, f"ml_{output_filename}")
        analyst_output_path = os.path.join(self.processed_data_dir, f"analyst_{output_filename}")
        smart_output_path = os.path.join(self.processed_data_dir, f"smart_{output_filename}")
        
        df_ml.to_csv(ml_output_path, index=False)
        df_analyst.to_csv(analyst_output_path, index=False)
        df_smart.to_csv(smart_output_path, index=False)
        logger.info(f"Pipeline complete! Saved to processed directory.")
        
        return {"ml_ready": df_ml, "analyst_ready": df_analyst, "smart_ready": df_smart}

if __name__ == "__main__":
    # Provides a simple CLI execution path if run directly
    print("DataCleaningPipeline module is ready. Import this class into your notebooks or scripts to run.")
