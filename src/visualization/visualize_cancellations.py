import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_cancellation_rate(data, categorical_features):
    """
    Visualize cancellation rate by categorical features using bar plots.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    categorical_features (list): A list of categorical feature names to analyze cancellation rates.
    """
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(categorical_features):
        plt.subplot(2, 3, i + 1)
        sns.barplot(x=feature, y=data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0), data=data, errorbar=None)
        plt.title(f'Cancellation Rate by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Cancellation Rate')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('data/processed/processed_data.csv')
    categorical_features = [
        'type_of_meal_plan', 'room_type_reserved', 
        'market_segment_type', 'repeated_guest', 
        'required_car_parking_space'
    ]
    visualize_cancellation_rate(data, categorical_features)
