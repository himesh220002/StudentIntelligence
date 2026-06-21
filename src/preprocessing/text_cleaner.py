import pandas as pd
import string

class TextCleaner:
    """
    Production-grade NLP utility class for Tokenization (A41) and Stopword Removal (A42).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def tokenize(self, column: str) -> pd.DataFrame:
        """
        Tokenizes a text column using NLTK (A41).
        Converts strings into lists of words.
        """
        try:
            from nltk.tokenize import word_tokenize
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)
        except ImportError:
            raise ImportError("Please install `nltk` to use tokenization.")

        if column in self.df.columns:
            # Handle NaNs and ensure string type before tokenization
            self.df[column] = self.df[column].fillna('').astype(str).apply(word_tokenize)
        return self.df

    def remove_stopwords(self, column: str, language: str = 'english') -> pd.DataFrame:
        """
        Removes stopwords from a tokenized text column using NLTK (A42).
        Can process either a list of words or a single string.
        """
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
        except ImportError:
            raise ImportError("Please install `nltk` to use stopword removal.")

        if column in self.df.columns:
            stop_words = set(stopwords.words(language))
            
            def filter_stops(val):
                if isinstance(val, list):
                    return [w for w in val if w.lower() not in stop_words and w not in string.punctuation]
                elif isinstance(val, str):
                    words = val.split()
                    return ' '.join([w for w in words if w.lower() not in stop_words and w not in string.punctuation])
                return val

            self.df[column] = self.df[column].apply(filter_stops)
            
        return self.df
