# Select exact attributes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def load_and_preprocess(path='../data/california_housing.csv', test_size=0.2, random_state=42, scale=True):
    """Load the (processed) CSV and return X_train, X_test, y_train, y_test as numpy arrays.


    Expects the target column to be named 'MedHouseVal' (sklearn's default) or 'medianHouseValue'.
    """
    df = pd.read_csv(path)


    # Detect common target column names
    if 'MedHouseVal' in df.columns:
        target_col = 'MedHouseVal'
    elif 'medianHouseValue' in df.columns:
        target_col = 'medianHouseValue'
    elif 'median_house_value' in df.columns:
        target_col = 'median_house_value'
    else:
    # Try to guess last column
        target_col = df.columns[-1]


    X = df.drop(columns=[target_col])
    y = df[target_col].values


    # Drop non-numeric or ID columns if present
    X = X.select_dtypes(include=[np.number])


    # Simple NA handling
    X = X.fillna(X.mean())


    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=random_state)


    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    return X_train, X_test, y_train, y_test
