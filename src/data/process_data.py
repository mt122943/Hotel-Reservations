import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def process_data():
    """
    Process raw data and save the processed data to the specified path.
    """
    raw_data_path = os.getenv('RAW_DATA_PATH')
    processed_data_path = os.getenv('PROCESSED_DATA_PATH')
    
    # Load raw data
    data = pd.read_csv(raw_data_path, sep=',')
    
    # Example data processing steps
    # 1. Handle missing values
    data = data.dropna()
    
    # 2. Convert data types if necessary
    data['booking_status'] = data['booking_status'].astype('category')
    
    # 3. Add any additional processing steps here
    # ...

    # Save processed data
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    process_data()
