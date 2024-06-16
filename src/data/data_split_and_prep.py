import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def preprocess_data():
    """
    Preprocess raw data to prepare for model training.
    """
    # Load processed data
    processed_data_path = os.getenv('PROCESSED_DATA_PATH')
    data = pd.read_csv(processed_data_path)
    
    # Convert booking_status to binary values
    data['booking_status'] = data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    
    # One-hot encoding for categorical variables
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Drop unnecessary columns
    X = data.drop(columns=['Booking_ID', 'booking_status', 'arrival_year', 'arrival_month', 'arrival_date'])
    y = data['booking_status']  # Target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)
    
    # Save the splits
    X_train.to_csv(os.path.join(os.path.dirname(processed_data_path), 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(os.path.dirname(processed_data_path), 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(os.path.dirname(processed_data_path), 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(os.path.dirname(processed_data_path), 'y_test.csv'), index=False)

if __name__ == "__main__":
    preprocess_data()
