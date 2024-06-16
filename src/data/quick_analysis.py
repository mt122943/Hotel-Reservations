import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def quick_analysis():
    """
    Perform a quick analysis of the raw data, including basic statistics and info.
    """
    raw_data_path = os.getenv('RAW_DATA_PATH')
    
    # Load raw data
    data = pd.read_csv(raw_data_path, sep=',')
    
    # Display basic statistics
    print("Basic Statistics:")
    print(data.describe())
    
    # Display information about the dataset
    print("\nData Info:")
    data.info()
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"Total number of rows: {data.shape[0]}")
    print(f"Total number of columns: {data.shape[1]}")
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])

if __name__ == "__main__":
    quick_analysis()
