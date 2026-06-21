import pandas as pd
import numpy as np
from typing import List, Optional, Literal

class OutlierDetector:
    """
    Production-grade utility class to detect and handle statistical outliers
    using standard robust methodologies (IQR, Z-Score).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_outliers_iqr(self, 
                            columns: Optional[List[str]] = None, 
                            action: Literal['drop', 'cap'] = 'cap', 
                            factor: float = 1.5) -> pd.DataFrame:
        """
        Handles outliers using the Interquartile Range (IQR) method.
        Highly robust for skewed distributions.
        
        Args:
            columns: List of numeric columns to target. If None, targets all numeric columns.
            action: 'cap' clamps values to the IQR bounds. 'drop' removes the entire row.
            factor: IQR multiplier to establish bounds (1.5 is standard, 3.0 is extreme).
        """
        cols = columns if columns is not None else self.df.select_dtypes(include='number').columns
        for col in cols:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if action == 'cap':
                self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
            elif action == 'drop':
                # Create mask for valid values OR missing values (we don't want to drop NaNs here, they are handled separately)
                valid_mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                valid_mask = valid_mask | self.df[col].isna()
                self.df = self.df[valid_mask]
                
        return self.df

    def handle_outliers_zscore(self, 
                               columns: Optional[List[str]] = None, 
                               action: Literal['drop', 'cap'] = 'cap', 
                               threshold: float = 3.0) -> pd.DataFrame:
        """
        Handles outliers using the Z-Score method.
        Assumes normal distribution.
        
        Args:
            columns: List of numeric columns to target. If None, targets all numeric columns.
            action: 'cap' clamps values to the threshold bounds. 'drop' removes the entire row.
            threshold: Standard deviation threshold (3.0 is standard).
        """
        cols = columns if columns is not None else self.df.select_dtypes(include='number').columns
        for col in cols:
            if col not in self.df.columns:
                continue
                
            mean = self.df[col].mean()
            std = self.df[col].std()
            
            if std == 0 or pd.isna(std):
                continue  # Avoid division by zero or NaN issues if variance is 0
                
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            if action == 'cap':
                self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
            elif action == 'drop':
                valid_mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                valid_mask = valid_mask | self.df[col].isna()
                self.df = self.df[valid_mask]
                
        return self.df

    def handle_outliers_clustering(self, columns=None, action='drop', contamination=0.05) -> pd.DataFrame:
        """
        Detects and handles outliers using Isolation Forest clustering (A31).
        """
        from sklearn.ensemble import IsolationForest
        
        cols = columns if columns is not None else self.df.select_dtypes(include='number').columns
        cols = [c for c in cols if c in self.df.columns]
        
        if not cols:
            return self.df
            
        # Isolation Forest cannot handle NaNs natively, fill temporarily with median
        temp_df = self.df[cols].fillna(self.df[cols].median())
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(temp_df)
        
        # preds: 1 for inliers, -1 for outliers
        if action == 'drop':
            self.df = self.df[preds == 1]
        elif action == 'cap':
            # Capping is complex in multivariate space, so we replace outlier values with median
            for col in cols:
                median_val = self.df[col].median()
                self.df.loc[preds == -1, col] = median_val
                
        return self.df
