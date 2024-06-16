import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_categorical_features(data, categorical_features):
    """
    Visualize categorical features using bar plots.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    categorical_features (list): A list of categorical feature names to visualize.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(categorical_features):
        plt.subplot(2, 3, i + 1)
        sns.countplot(data=data, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
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
    visualize_categorical_features(data, categorical_features)
