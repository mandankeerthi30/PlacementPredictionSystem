from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_data

from src.data.preprocess import (
    split_data,
    handle_missing_values,
    identify_features,
    standardize,
    one_hot_encode_data,
    ordinal_encode_data
)


# ==================================================
# CREATE RANDOM FOREST MODEL
# ==================================================

def create_model():

    model = RandomForestClassifier(

        # Number of trees
        n_estimators=30,

        # Maximum depth
        max_depth=10,

        # Features considered at each split
        max_features="sqrt",

        # Same result every time
        random_state=42,

        # Out-of-Bag score
        oob_score=True,

        # Use all CPU cores
        n_jobs=-1
    )

    print("\nRandom Forest Model Created")

    return model


# ==================================================
# TRAIN MODEL
# ==================================================

def train_model(model, X_train, y_train):

    print("\nTraining Random Forest...")
    print("Please wait...")

    model.fit(X_train, y_train)

    print("\nRandom Forest trained successfully!")

    return model


# ==================================================
# EVALUATE MODEL
# ==================================================

def evaluate_model(model, X_test, y_test):

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print("\nTest Accuracy:", accuracy)

    # OOB Score
    print("OOB Score:", model.oob_score_)

    # Classification Report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return y_pred


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    # ----------------------------------------------
    # 1. LOAD DATASET
    # ----------------------------------------------

    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)


    # ----------------------------------------------
    # 2. SPLIT DATASET
    # ----------------------------------------------

    X_train, X_test, y_train, y_test = split_data(

        df,

        target_column="PlacementStatus",

        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)


    # ----------------------------------------------
    # 3. IDENTIFY FEATURES
    # ----------------------------------------------

    numerical_features, categorical_features = identify_features(
        X_train
    )

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)


    # ----------------------------------------------
    # 4. DEFINE ENCODING FEATURES
    # ----------------------------------------------

    # One-Hot Encoding
    one_hot_features = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    # Ordinal Encoding
    ordinal_features = [
        "CGPA_Tier"
    ]


    # ----------------------------------------------
    # 5. HANDLE MISSING VALUES
    # ----------------------------------------------

    X_train, X_test, imputer = handle_missing_values(

        X_train,
        X_test,
        numerical_features
    )

    print("\nMissing Value Handling Completed")


    # ----------------------------------------------
    # 6. STANDARDIZATION
    # ----------------------------------------------

    X_train, X_test, scaler = standardize(

        X_train,
        X_test,
        numerical_features
    )

    print("\nStandardization Completed")


    # ----------------------------------------------
    # 7. ONE-HOT ENCODING
    # ----------------------------------------------

    X_train, X_test, one_hot_encoder = one_hot_encode_data(

        X_train,
        X_test,
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed")


    # ----------------------------------------------
    # 8. ORDINAL ENCODING
    # ----------------------------------------------

    X_train, X_test, ordinal_encoder = ordinal_encode_data(

        X_train,
        X_test,
        ordinal_features
    )

    print("\nOrdinal Encoding Completed")


    # ----------------------------------------------
    # 9. FINAL DATASET SHAPE
    # ----------------------------------------------

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)


    # ----------------------------------------------
    # 10. CREATE MODEL
    # ----------------------------------------------

    model = create_model()


    # ----------------------------------------------
    # 11. TRAIN MODEL
    # ----------------------------------------------

    model = train_model(

        model,
        X_train,
        y_train
    )


    # ----------------------------------------------
    # 12. EVALUATE MODEL
    # ----------------------------------------------

    y_pred = evaluate_model(

        model,
        X_test,
        y_test
    )


    # ----------------------------------------------
    # 13. SAVE PREPROCESSED TRAIN DATA
    # ----------------------------------------------

    train_output = X_train.copy()

    train_output["PlacementStatus"] = y_train

    train_output.to_csv(

        r"C:\Users\hp\PycharmProjects\PlacementPredictionSystem\src\data\preprocessed_train.csv",

        index=False
    )


    # ----------------------------------------------
    # 14. SAVE PREPROCESSED TEST DATA
    # ----------------------------------------------

    test_output = X_test.copy()

    test_output["PlacementStatus"] = y_test

    test_output.to_csv(

        r"C:\Users\hp\PycharmProjects\PlacementPredictionSystem\src\data\preprocessed_test.csv",

        index=False
    )


    print("\nPreprocessed files saved successfully.")


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":

    main()