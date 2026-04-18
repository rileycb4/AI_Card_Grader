import pandas as pd
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_grading_model(data_path, model_output_path):
    # 1. Load the dataset
    df = pd.read_csv(data_path)
    
    # 8 Predictors
    features = [
        'front_corners', 'front_edges', 'front_centering', 'front_surface',
        'back_corners', 'back_edges', 'back_centering', 'back_surface'
    ]
    
    X = df[features]
    y = df['target_score_1000'] # The 1-1000 score
    
    # 2. Split into training and validation sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Validate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error on test set: {mae:.2f} points (out of 1000)")
    
    # 5. Save the trained model for the analysis program
    joblib.dump(model, model_output_path)
    print(f"Model saved successfully to {model_output_path}")

if __name__ == "__main__":
    # Example execution:
    train_grading_model('card_training_data_dummy.csv', 'card_grader_model.pkl')
    pass