import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_numerical_features(data, numerical_features):
    """
    Visualize numerical features using histograms.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    numerical_features (list): A list of numerical feature names to visualize.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(numerical_features):
        plt.subplot(3, 3, i + 1)
        sns.histplot(data[feature], bins=30)
        plt.title(f'{feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('data/processed/processed_data.csv')
    numerical_features = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
        'no_of_week_nights', 'lead_time', 'no_of_previous_cancellations', 
        'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 
        'no_of_special_requests'
    ]
    visualize_numerical_features(data, numerical_features)
