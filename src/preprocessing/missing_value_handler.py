import pandas as pd

class MissingValueHandler:
    """
    A class to handle missing values using various strategies like dropping or imputation.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def drop_missing(self, axis=0, thresh=None, subset=None) -> pd.DataFrame:
        """
        Drops rows (axis=0) or columns (axis=1) with missing values.
        """
        self.df = self.df.dropna(axis=axis, thresh=thresh, subset=subset)
        return self.df
        
    def impute_numerical(self, strategy='mean', columns=None) -> pd.DataFrame:
        """
        Imputes missing values in numerical columns with 'mean' or 'median'.
        """
        cols = columns if columns else self.df.select_dtypes(include='number').columns
        for col in cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                else:
                    raise ValueError("Strategy must be 'mean' or 'median'.")
                self.df[col] = self.df[col].fillna(fill_val)
        return self.df
        
    def impute_categorical(self, strategy='mode', fill_value='Unknown', columns=None) -> pd.DataFrame:
        """
        Imputes missing values in categorical/object columns with 'mode' or a 'constant' value.
        """
        cols = columns if columns else self.df.select_dtypes(exclude='number').columns
        for col in cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant':
                    fill_val = fill_value
                else:
                    raise ValueError("Strategy must be 'mode' or 'constant'.")
                self.df[col] = self.df[col].fillna(fill_val)
        return self.df

    def impute_time_series(self, strategy='ffill', columns=None) -> pd.DataFrame:
        """
        Imputes missing values using forward or backward fill for sequential data (M12).
        """
        cols = columns if columns else self.df.columns
        for col in cols:
            if self.df[col].isnull().any():
                if strategy == 'ffill':
                    self.df[col] = self.df[col].ffill()
                elif strategy == 'bfill':
                    self.df[col] = self.df[col].bfill()
                else:
                    raise ValueError("Strategy must be 'ffill' or 'bfill'.")
        return self.df
