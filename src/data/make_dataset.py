import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def load_raw_data():
    """
    Load raw data from a CSV file specified in the .env file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    raw_data_path = os.getenv('RAW_DATA_PATH')
    data = pd.read_csv(raw_data_path, sep=',')
    return data

if __name__ == "__main__":
    # Example usage
    raw_data = load_raw_data()
    print(raw_data.head())
