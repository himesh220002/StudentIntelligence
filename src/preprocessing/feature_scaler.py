import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Literal, Optional

class FeatureScaler:
    """
    Production-grade utility class to scale numerical features 
    using Min-Max or Standardization techniques while securely 
    preserving the pandas DataFrame structure (index and columns).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def scale_features(self, 
                       columns: Optional[List[str]] = None, 
                       method: Literal['minmax', 'standard'] = 'standard',
                       keep_original: bool = False,
                       exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scales the targeted numeric columns and updates the DataFrame.
        
        Args:
            columns: List of numeric columns to scale. If None, targets all numeric columns.
            method: 'minmax' scales values to [0, 1]. 
                    'standard' applies Z-score scaling (mean=0, std=1).
            keep_original: If True, maintains the original values in '{col}_raw' and puts scaled in '{col}_scaled'.
            exclude_columns: List of columns to explicitly skip scaling (e.g., identifiers).
        """
        # Tag: IT_scaling_pipeline_selective
        cols_to_scale = columns if columns is not None else self.df.select_dtypes(include='number').columns.tolist()
        
        # Heuristically exclude ID columns if not explicitly provided
        if exclude_columns is None and columns is None:
            exclude_columns = [col for col in cols_to_scale if col.lower() == 'id' or col.lower().endswith('id')]
        elif exclude_columns is None:
            exclude_columns = []
            
        # Verify columns exist and exclude appropriately
        cols_to_scale = [col for col in cols_to_scale if col in self.df.columns and col not in exclude_columns]
        
        if not cols_to_scale:
            return self.df
            
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Method must be 'minmax' or 'standard'")
            
        # Fit and transform safely
        scaled_array = scaler.fit_transform(self.df[cols_to_scale])
        
        # Update the original dataframe with scaled values using index preservation
        scaled_df = pd.DataFrame(scaled_array, columns=cols_to_scale, index=self.df.index)
        
        # Tag: UT_dual_column_validation
        for col in cols_to_scale:
            if keep_original:
                self.df[f"{col}_raw"] = self.df[col]
                self.df[f"{col}_scaled"] = scaled_df[col]
                self.df.drop(columns=[col], inplace=True)
            else:
                self.df[col] = scaled_df[col]
            
        return self.df
