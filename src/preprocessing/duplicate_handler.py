import pandas as pd

class DuplicateHandler:
    """
    A class to handle exact and key-based duplicate rows in a DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def drop_exact_duplicates(self, keep='first') -> pd.DataFrame:
        """
        Drops completely identical rows. 
        keep='first' (keeps first occurrence), 'last' (keeps last), or False (drops all copies).
        """
        self.df = self.df.drop_duplicates(keep=keep)
        return self.df
        
    def drop_key_duplicates(self, subset, keep='first') -> pd.DataFrame:
        """
        Drops rows that have identical values in the specific column(s) defined in 'subset'.
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self.df

    def drop_fuzzy_duplicates(self, column: str, threshold: int = 90) -> pd.DataFrame:
        """
        Drops rows using Fuzzy Matching (M22) on a specific text column.
        """
        try:
            from thefuzz import fuzz, process
        except ImportError:
            raise ImportError("Please install `thefuzz` to use fuzzy matching.")

        if column not in self.df.columns:
            return self.df

        temp_df = self.df.dropna(subset=[column]).copy()
        temp_df[column] = temp_df[column].astype(str)
        
        indices_to_drop = []
        unique_texts = []
        
        for idx, row in temp_df.iterrows():
            text = row[column]
            if not unique_texts:
                unique_texts.append((idx, text))
                continue
                
            match = process.extractOne(text, [t[1] for t in unique_texts], scorer=fuzz.ratio)
            
            if match and match[1] >= threshold:
                indices_to_drop.append(idx)
            else:
                unique_texts.append((idx, text))

        self.df = self.df.drop(index=indices_to_drop)
        return self.df
