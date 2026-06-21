import pandas as pd
import numpy as np

class LabelCleaner:
    """
    Utility class for detecting and correcting label errors in classification datasets (A32).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_label_errors(self, feature_columns: list, label_column: str, action='drop') -> pd.DataFrame:
        """
        Uses cross-validation with a Random Forest to detect samples where 
        the provided label highly conflicts with the model's prediction (A32).
        This is a common method for finding mislabeled samples.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        
        # Verify columns exist
        missing_cols = [c for c in feature_columns + [label_column] if c not in self.df.columns]
        if missing_cols:
            return self.df
            
        # Drop rows with NaNs in features/labels temporarily as sklearn cannot handle them
        temp_df = self.df.dropna(subset=feature_columns + [label_column]).copy()
        
        if temp_df.empty:
            return self.df
            
        X = temp_df[feature_columns]
        y = temp_df[label_column]
        
        # If there's only 1 class or too few samples, return
        if len(y.unique()) <= 1 or len(y) < 5:
            return self.df

        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        
        try:
            preds = cross_val_predict(clf, X, y, cv=3)
        except ValueError:
            # Fallback if a class has fewer than 3 members
            return self.df
            
        # Flag errors where the CV prediction conflicts with the given label
        temp_df['is_error'] = (preds != y)
        
        # Join back to original df
        self.df['label_error_flag'] = False
        self.df.loc[temp_df.index, 'label_error_flag'] = temp_df['is_error']
        
        if action == 'drop':
            self.df = self.df[~self.df['label_error_flag']]
            self.df = self.df.drop(columns=['label_error_flag'])
            
        return self.df
