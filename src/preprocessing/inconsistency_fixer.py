import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class InconsistencyFixer:
    """
    Production-grade utility class for standardizing data formats,
    resolving categorical inconsistencies, and parsing datetimes.
    Optimized for robustness and error handling.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def standardize_strings(self, columns: Optional[List[str]] = None, keep_original: bool = False) -> pd.DataFrame:
        """
        Standardizes string columns by converting to lowercase,
        stripping leading/trailing whitespace, and replacing multiple interior spaces with a single space.
        
        Args:
            columns: List of column names to process. If None, targets all 'object' and 'string' dtypes.
            keep_original: If True, maintains the original values in '{col}_original' and puts cleaned in '{col}_clean'.
        """
        # Tag: ST_categorical_dual_representation
        cols = columns if columns is not None else self.df.select_dtypes(include=['object', 'string']).columns.tolist() # type: ignore
        for col in cols:
            if col in self.df.columns:
                if keep_original:
                    self.df[f"{col}_original"] = self.df[col]
                    clean_col_name = f"{col}_clean"
                else:
                    clean_col_name = col
                
                # Use robust vectorization for string manipulation
                self.df[clean_col_name] = (
                    self.df[col]
                    .astype(str)
                    .str.lower()
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                )
                # 'nan' strings resulting from astype(str) on actual NaNs are converted back to np.nan
                self.df[clean_col_name] = self.df[clean_col_name].replace('nan', np.nan)
                
                if keep_original:
                    self.df.drop(columns=[col], inplace=True)
        return self.df

    def standardize_dates(self, columns: List[str], datetime_format: Optional[str] = None) -> pd.DataFrame:
        """
        Attempts to parse given columns into pandas datetime objects.
        Coerces parsing errors to NaT to prevent pipeline crashes on invalid noisy data.
        
        Args:
            columns: List of column names containing date strings.
            datetime_format: Optional specific format string (e.g. '%Y-%m-%d'). 
                             If None, pandas attempts to infer the format automatically.
        """
        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], format=datetime_format, errors='coerce')
        return self.df

    def map_categories(self, column: str, mapping_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Maps inconsistent categorical values to standard values based on a dictionary mapping.
        
        Args:
            column: The categorical column to target.
            mapping_dict: Dictionary where keys are inconsistent values (e.g. "ny") 
                          and values are the standardized value (e.g. "New York").
        """
        if column in self.df.columns:
            self.df[column] = self.df[column].replace(mapping_dict)
        return self.df

    def convert_units(self, column: str, conversion_rate: float) -> pd.DataFrame:
        """
        Converts a numeric column by a conversion rate (M32).
        For example, cm to m (conversion_rate = 0.01).
        """
        if column in self.df.columns:
            # If the column contains strings with units (like '100 cm'), we would need regex extraction first.
            # Assuming the column is numeric for simplicity of the rate conversion.
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce') * conversion_rate
        return self.df
