import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def visualize_correlations(data):
    """
    Visualize the correlation matrix of the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    """
    # Map booking status to numerical values
    data['booking_status_num'] = data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    
    # Calculate the correlation matrix
    correlation_matrix = data.corr(numeric_only=True)
    
    # Visualize the correlation matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, center=0, 
                linewidths=.5, cbar_kws={"shrink": .75})
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    # Remove the additional column
    data.drop(columns=['booking_status_num'], inplace=True)

if __name__ == "__main__":
    # Example usage
    processed_data_path = os.getenv('PROCESSED_DATA_PATH')
    data = pd.read_csv(processed_data_path)
    visualize_correlations(data)
