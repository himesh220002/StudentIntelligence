import pandas as pd
import numpy as np

class IssueIdentifier:
    """
    A class to automatically identify data quality issues in a pandas DataFrame.
    This serves as the foundation for deciding which cleaning steps are necessary.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def check_missing_values(self) -> pd.DataFrame:
        """Returns a dataframe with counts and percentages of missing values per column."""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        result = pd.DataFrame({'Missing_Count': missing, 'Missing_Percent': missing_percent})
        return result[result['Missing_Count'] > 0].sort_values(by='Missing_Percent', ascending=False)
        
    def check_duplicates(self) -> dict:
        """Returns the number and percentage of completely duplicated rows."""
        duplicates = self.df.duplicated().sum()
        dup_percent = (duplicates / len(self.df)) * 100
        return {
            "Duplicate_Rows": int(duplicates), 
            "Duplicate_Percent": round(float(dup_percent), 2)
        }
        
    def check_dtypes(self) -> dict:
        """Returns the data types of columns as a dictionary."""
        return self.df.dtypes.astype(str).to_dict()
        
    def generate_report(self) -> dict:
        """Generates a comprehensive summary report of common data issues."""
        print("Generating Data Quality Report...")
        report = {
            "Total_Rows": len(self.df),
            "Total_Columns": len(self.df.columns),
            "Data_Types": self.check_dtypes(),
            "Duplicates": self.check_duplicates(),
            "Missing_Values": self.check_missing_values().to_dict(orient='index')
        }
        return report

if __name__ == "__main__":
    # Basic sanity check with dummy data
    test_data = {
        "id": [1, 2, 2, 4, 5], 
        "name": ["Alice", "Bob", "Bob", None, "Eve"], 
        "age": [25, np.nan, np.nan, 22, 29]
    }
    df = pd.DataFrame(test_data)
    
    print("--- Test DataFrame ---")
    print(df)
    print("\n--- Identifying Issues ---")
    
    identifier = IssueIdentifier(df)
    report = identifier.generate_report()
    
    print(f"\n📊 Summary Report:")
    print(f"Total Rows: {report['Total_Rows']}")
    print(f"Total Columns: {report['Total_Columns']}")
    
    print(f"\nDuplicates:")
    print(f"  - {report['Duplicates']['Duplicate_Rows']} row(s) ({report['Duplicates']['Duplicate_Percent']}%)")
    
    print("\nMissing Values:")
    if not report['Missing_Values']:
        print("  - None")
    else:
        for col, stats in report['Missing_Values'].items():
            print(f"  - Column '{col}': {stats['Missing_Count']} missing ({stats['Missing_Percent']}%)")
