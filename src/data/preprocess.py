import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from src.data.load_data import load_data


# =========================================================
# 1. SPLIT DATA
# =========================================================

def split_data(df, target_column, drop_columns=None, stratify=False):

    if drop_columns is None:
        drop_columns = []

    # Create X by removing target and unwanted columns
    X = df.drop(columns=drop_columns + [target_column])

    # Create target variable
    y = df[target_column]

    # Stratification
    stratify_value = y if stratify else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# 2. SPLIT X DATA
# =========================================================

def split_x_data(df, drop_columns=None):

    if drop_columns is None:
        drop_columns = []

    # Remove unwanted columns
    X = df.drop(columns=drop_columns)

    return X


# =========================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# =========================================================

def identify_features(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return numerical_features, categorical_features


# =========================================================
# 4. HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(
        X_train,
        X_test,
        numerical_features
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    X_train[numerical_features] = imputer.fit_transform(
        X_train[numerical_features]
    )

    # Transform test data
    X_test[numerical_features] = imputer.transform(
        X_test[numerical_features]
    )

    return X_train, X_test, imputer


# =========================================================
# 5. STANDARDIZATION
# =========================================================

def standardize(
        X_train,
        X_test,
        numerical_features
):

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit and transform training data
    X_train[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )

    # Transform test data
    X_test[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )

    return X_train, X_test, scaler


# =========================================================
# 6. STANDARDIZATION ALIAS
# =========================================================
# This allows other files to use either:
# standardize()
# or
# standardize_data()

def standardize_data(
        X_train,
        X_test,
        numerical_features
):

    return standardize(
        X_train,
        X_test,
        numerical_features
    )


# =========================================================
# 7. ONE-HOT ENCODING
# =========================================================

def one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
):

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[one_hot_features]
    )

    # Transform test data
    test_encoded = encoder.transform(
        X_test[one_hot_features]
    )

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(
        one_hot_features
    )

    # Convert encoded arrays to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # Remove original categorical columns
    X_train = X_train.drop(
        columns=one_hot_features
    )

    X_test = X_test.drop(
        columns=one_hot_features
    )

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


# =========================================================
# 8. ORDINAL ENCODING
# =========================================================

def ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
):

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[ordinal_features]
    )

    # Transform test data
    test_encoded = encoder.transform(
        X_test[ordinal_features]
    )

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(
        ordinal_features
    )

    # Convert encoded arrays to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # Remove original ordinal columns
    X_train = X_train.drop(
        columns=ordinal_features
    )

    X_test = X_test.drop(
        columns=ordinal_features
    )

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_data()

    print("Original dataset shape:")
    print(df.shape)

    print("\nOriginal columns:")
    print(df.columns.tolist())


    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[],
        stratify=False
    )

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)


    # -----------------------------------------------------
    # IDENTIFY FEATURES
    # -----------------------------------------------------

    numerical_features, categorical_features = identify_features(
        X_train
    )

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)


    # -----------------------------------------------------
    # REMOVE STUDENT ID
    # -----------------------------------------------------

    if "StudentID" in X_train.columns:

        X_train = X_train.drop(
            columns=["StudentID"]
        )

        X_test = X_test.drop(
            columns=["StudentID"]
        )

        if "StudentID" in numerical_features:
            numerical_features.remove("StudentID")


    # -----------------------------------------------------
    # DEFINE ENCODING FEATURES
    # -----------------------------------------------------

    # Nominal categorical features
    one_hot_features = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    # Ordinal categorical features
    # Low < Medium < High
    ordinal_features = [
        "CGPA_Tier"
    ]


    # -----------------------------------------------------
    # HANDLE MISSING VALUES
    # -----------------------------------------------------

    X_train, X_test, imputer = handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print("\nMissing values after imputation:")

    print(
        X_train[numerical_features]
        .isnull()
        .sum()
    )

    print("\nMissing values handling completed.")


    # -----------------------------------------------------
    # STANDARDIZE NUMERICAL FEATURES
    # -----------------------------------------------------

    X_train, X_test, scaler = standardize(
        X_train,
        X_test,
        numerical_features
    )

    print("\nStandardizing completed.")


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )

    print("\nOne-hot encoding completed.")


    # -----------------------------------------------------
    # ORDINAL ENCODING
    # -----------------------------------------------------

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )

    print("\nOrdinal encoding completed.")


    # -----------------------------------------------------
    # ADD TARGET COLUMN
    # -----------------------------------------------------

    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test


    # -----------------------------------------------------
    # SAVE PREPROCESSED DATA
    # -----------------------------------------------------

    train_path = (
        r"C:\Users\hp\PycharmProjects"
        r"\PlacementPredictionSystem\src\data"
        r"\preprocessed_train.csv"
    )

    test_path = (
        r"C:\Users\hp\PycharmProjects"
        r"\PlacementPredictionSystem\src\data"
        r"\preprocessed_test.csv"
    )


    X_train.to_csv(
        train_path,
        index=False
    )

    X_test.to_csv(
        test_path,
        index=False
    )


    # -----------------------------------------------------
    # FINAL INFORMATION
    # -----------------------------------------------------

    print("\nFinal training shape:")
    print(X_train.shape)

    print("\nFinal testing shape:")
    print(X_test.shape)

    print("\nPreprocessing completed successfully.")

    print("\nScaler:")
    print(scaler)

    print("\nImputer:")
    print(imputer)