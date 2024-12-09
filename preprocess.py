
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_data(df, is_train=True):
    """
    Prepare dataset by first dropping rows without SII (for training), handling missing values 
    and converting categorical data to integers.
    
    Args:
        df: pandas DataFrame with the raw data
        is_train: Boolean indicating if this is training data (if True, drops rows without SII)
    
    Returns:
        Cleaned pandas DataFrame ready for modeling with all categorical variables encoded
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # For training data, drop rows without SII values
    if is_train:
        data = data.dropna(subset=['sii'])
        print(f"Dropped {len(df) - len(data)} rows without SII values")
    
    # First identify columns with too many missing values (>70%)
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > 0.7].index.tolist()
    data = data.drop(columns=columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns with >70% missing values: {columns_to_drop}")
    
    # Create label encoder
    le = LabelEncoder()
    
    # Handle each column based on its type and content
    for column in data.columns:
        if column == 'id':  # Skip id column
            continue
            
        # Convert column to string temporarily to handle mixed types
        data[column] = data[column].astype(str)
        
        # Replace empty strings and 'nan' with None
        data[column] = data[column].replace({'': None, 'nan': None})
        
        # Check if column contains only numeric values
        numeric_mask = data[column].str.match(r'^-?\d*\.?\d*$').fillna(False)
        all_numeric = numeric_mask.all()
        
        if all_numeric:
            # Convert to float and fill NaN with median
            data[column] = pd.to_numeric(data[column], errors='coerce')
            median_val = data[column].median()
            data[column] = data[column].fillna(median_val)
        else:
            # Encode categorical values
            # First fill NaN with a placeholder
            data[column] = data[column].fillna('missing')
            
            # Apply label encoding
            data[column] = le.fit_transform(data[column])
    
    # Handle specific columns that should be numeric
    numeric_columns = [
        'Basic_Demos-Age',
        'Basic_Demos-Sex',
        'CGAS-CGAS_Score',
        'Physical-BMI',
        'Physical-Height',
        'Physical-Weight',
        'Physical-Waist_Circumference',
        'Physical-Diastolic_BP',
        'Physical-HeartRate',
        'Physical-Systolic_BP',
        'sii'  # target variable
    ]
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill NaN with median
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
    
    # Convert season columns to integers (0-3)
    season_columns = [col for col in data.columns if col.endswith('-Season')]
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    
    for col in season_columns:
        if col in data.columns:
            # First convert any numeric-like values to strings
            data[col] = data[col].astype(str)
            # Map seasons to integers
            data[col] = data[col].map(lambda x: season_mapping.get(x, 4))  # 4 for unknown values
            # Fill any NaN with mode
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Final verification that all columns are numeric
    for column in data.columns:
        if column != 'id':  # Skip id column
            data[column] = pd.to_numeric(data[column], errors='coerce')
            # Fill any remaining NaN with column median
            data[column] = data[column].fillna(data[column].median())
    
    # Verify no missing values remain
    assert data.isnull().sum().sum() == 0, "There are still missing values in the dataset"
    
    return data

def main():
    # Read the data
    train_data = pd.read_csv('data/train.csv')
    
    # Prepare the training data
    cleaned_data = prepare_data(train_data, is_train=True)

    cleaned_data.to_csv('test.csv')
    
    # Print some information about the cleaned dataset
    print("\nShape of cleaned dataset:", cleaned_data.shape)
    print("\nColumns in cleaned dataset:", cleaned_data.columns.tolist())
    print("\nSample of cleaned data:\n", cleaned_data.head())
    
    # Verify all columns (except id) are numeric
    non_numeric_cols = [col for col in cleaned_data.columns 
                       if col != 'id' and not np.issubdtype(cleaned_data[col].dtype, np.number)]
    if non_numeric_cols:
        print("\nWarning: The following columns are not numeric:", non_numeric_cols)
    else:
        print("\nAll columns (except id) are successfully converted to numeric values")
    
    # Print summary of SII values
    print("\nSII value distribution:")
    print(cleaned_data['sii'].value_counts().sort_index())
    
    return cleaned_data

if __name__ == "__main__":
    cleaned_data = main()
