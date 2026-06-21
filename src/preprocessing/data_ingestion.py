import pandas as pd
import os

class DataIngestor:
    """
    A class to handle data ingestion from various file formats.
    Currently supports tabular data: CSV, Excel, and JSON.
    """
    
    def __init__(self, raw_data_dir: str = "../data/raw"):
        self.raw_data_dir = raw_data_dir
        
    def _get_full_path(self, filepath: str) -> str:
        """Helper to resolve the full path if only a filename is given."""
        if os.path.exists(filepath):
            return filepath
        full_path = os.path.join(self.raw_data_dir, filepath)
        if os.path.exists(full_path):
            return full_path
        raise FileNotFoundError(f"File not found: {filepath}. Please ensure it exists in {self.raw_data_dir}")

    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame."""
        path = self._get_full_path(filepath)
        print(f"Loading CSV from: {path}")
        return pd.read_csv(path, **kwargs)

    def load_excel(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load an Excel file into a pandas DataFrame."""
        path = self._get_full_path(filepath)
        print(f"Loading Excel from: {path}")
        return pd.read_excel(path, **kwargs)

    def load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a JSON file into a pandas DataFrame."""
        path = self._get_full_path(filepath)
        print(f"Loading JSON from: {path}")
        return pd.read_json(path, **kwargs)

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Auto-detect file extension and load data into a DataFrame.
        Use this as the primary entry point for tabular data.
        """
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext == '.csv':
            return self.load_csv(filepath, **kwargs)
        elif ext in ['.xls', '.xlsx']:
            return self.load_excel(filepath, **kwargs)
        elif ext == '.json':
            return self.load_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only CSV, Excel, and JSON are currently supported.")

if __name__ == "__main__":
    # Basic sanity check with a temporary file
    import tempfile
    
    test_data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    df = pd.DataFrame(test_data)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        test_path = f.name
        
    print(f"Created temporary dummy file at {test_path}")
    
    ingestor = DataIngestor(raw_data_dir=".")
    try:
        loaded_df = ingestor.load_data(test_path)
        print("\n✅ Successfully loaded dummy data using DataIngestor:")
        print(loaded_df)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
    finally:
        os.remove(test_path)
        print(f"\nCleaned up temporary file: {test_path}")
