from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


def clean_test_data(cleaned_path: str, test_path: str) -> pd.DataFrame:
    # Read both datasets
    cleaned_df = pd.read_csv(cleaned_path)
    test_df = pd.read_csv(test_path)
    
    # Get columns from cleaned data (excluding 'sii' if present)
    reference_columns = cleaned_df.columns.tolist()
    if 'sii' in reference_columns:
        reference_columns.remove('sii')
    
    # Keep only columns that exist in cleaned data
    test_df = test_df[['id'] + [col for col in reference_columns if col in test_df.columns]]
    
    # Create a copy to avoid modifying the original DataFrame
    df_encoded = test_df.copy()
    
    # Function to check if a column contains strings
    def contains_strings(column):
        # Convert to string first to handle mixed types
        return column.astype(str).apply(lambda x: not x.replace('.', '').isdigit() if pd.notna(x) and x != '' else False).any()
    
    # Get means from cleaned data for numeric columns
    numeric_means = {}
    string_columns = []
    
    for column in reference_columns:
        if column in df_encoded.columns and column != 'id':
            if contains_strings(cleaned_df[column]):
                string_columns.append(column)
            else:
                numeric_means[column] = cleaned_df[column].mean()
    
    # Fill numeric missing values with means from cleaned data
    for column, mean_value in numeric_means.items():
        if column in df_encoded.columns:
            df_encoded[column] = pd.to_numeric(df_encoded[column], errors='coerce')
            df_encoded[column] = df_encoded[column].fillna(mean_value)
    
    # Handle string columns
    le = LabelEncoder()
    for column in string_columns:
        if column in df_encoded.columns:
            # Convert values to strings and handle NaN values
            cleaned_values = cleaned_df[column].fillna('').astype(str)
            test_values = df_encoded[column].fillna('').astype(str)
            
            # Combine unique values from both datasets
            all_values = pd.concat([cleaned_values, test_values]).unique()
            
            # Fit encoder on all unique values
            le.fit(all_values)
            
            # Transform values in test set
            df_encoded[column] = le.transform(test_values)
            
            # Convert column to numeric
            df_encoded[column] = pd.to_numeric(df_encoded[column])
    
    return df_encoded

class KNNModel:
    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN model with specified parameters.
        
        Args:
            n_neighbors: Number of neighbors to use for KNN (default=5)
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def create_and_train_model(self, X, y):
        """
        Create and train a KNN model with the given parameters.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Save feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Initialize and fit the scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train the model
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.model.fit(X_scaled, y)
    
    def evaluate_model(self, X, y):
        """
        Evaluate the model using various metrics.
        
        Args:
            X: Feature matrix
            y: True target values
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    
    def predict_test_set(self, test_path):
        """
        Make predictions on a test set and return them in the required format.
        
        Args:
            test_path: Path to the test CSV file
        
        Returns:
            DataFrame with id and sii predictions
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
            
        # Read and prepare test data
        test_data = pd.read_csv(test_path)
        test_ids = test_data['id'].copy()
        
        # Prepare the test data using the same preparation function
        prepared_test=clean_test_data("data/cleaned.csv",test_path)
        
        # Ensure we have all the necessary features in the same order
        missing_cols = set(self.feature_columns) - set(prepared_test.columns)
        for col in missing_cols:
            prepared_test[col] = 0  # Fill with 0 for missing columns
            
        # Select and order columns to match training data
        X_test = prepared_test[self.feature_columns]
        
        # Scale the features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Round predictions to nearest integer and clip to valid range [0, 3]
        predictions = np.clip(np.round(predictions), 0, 3)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': test_ids,
            'sii': predictions
        })
        
        return submission

def main():
    # Read and prepare the training data
    cleaned_data = pd.read_csv('data/cleaned.csv')

    
    # Separate features and target
    X = cleaned_data.drop(['sii'], axis=1)
    y = cleaned_data['sii']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42
    )
    
    # Create and train the model
    knn = KNNModel(n_neighbors=5)
    knn.create_and_train_model(X_train, y_train)
    
    # Evaluate on training data
    train_metrics = knn.evaluate_model(X_train, y_train)
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate on test data
    test_metrics = knn.evaluate_model(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Demonstrate test set prediction (using sample test path)
    try:
        submission = knn.predict_test_set('data/test.csv')
        print("\nSample of predictions:")
        print(submission.head())
        
        # Save predictions to file
        submission.to_csv('knn_predictions.csv', index=False)
        print("\nPredictions saved to 'knn_predictions.csv'")
    except FileNotFoundError:
        print("\nTest file not found. Skipping prediction demonstration.")
    
    return knn

if __name__ == "__main__":
    model=main()