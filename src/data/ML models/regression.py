from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.data.load_data import load_data

from src.data.preprocess import (
    split_data,
    handle_missing_values,
    standardize,
    one_hot_encode_data,
    ordinal_encode_data
)


# =========================================================
# 1. CREATE REGRESSION MODELS
# =========================================================

def create_models():

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "Lasso": Lasso(
            alpha=1.0
        ),

        "ElasticNet": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5
        )
    }

    return models


# =========================================================
# 2. TRAIN MODEL
# =========================================================

def train_model(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# 3. MAKE PREDICTIONS
# =========================================================

def predict_model(model, X_test):

    y_pred = model.predict(
        X_test
    )

    return y_pred


# =========================================================
# 4. EVALUATE MODEL
# =========================================================

def evaluate_model(y_test, y_pred):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# 5. MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_data()

    print("Original dataset shape:")
    print(df.shape)


    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "IsAnomaly"
        ]
    )

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nTesting shape:")
    print(X_test.shape)


    # -----------------------------------------------------
    # DEFINE FEATURES
    # -----------------------------------------------------

    numerical_features = [
        "Age",
        "CGPA",
        "Internships",
        "Projects",
        "Workshops",
        "AptitudeTestScore"
    ]


    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

    # Nominal categorical features
    # → One-Hot Encoding

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
    # → Ordinal Encoding

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

    print("\nMissing value handling completed.")


    # -----------------------------------------------------
    # STANDARDIZE NUMERICAL FEATURES
    # -----------------------------------------------------

    X_train, X_test, scaler = standardize(
        X_train,
        X_test,
        numerical_features
    )

    print("Standardization completed.")


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )

    print("One-hot encoding completed.")


    # -----------------------------------------------------
    # ORDINAL ENCODING
    # -----------------------------------------------------

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )

    print("Ordinal encoding completed.")


    # -----------------------------------------------------
    # FINAL FEATURE INFORMATION
    # -----------------------------------------------------

    print("\nFinal training shape:")
    print(X_train.shape)

    print("\nFinal testing shape:")
    print(X_test.shape)


    # -----------------------------------------------------
    # CREATE MODELS
    # -----------------------------------------------------

    models = create_models()


    # -----------------------------------------------------
    # TRAIN AND EVALUATE MODELS
    # -----------------------------------------------------

    results = {}

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)

        # Train
        model = train_model(
            model,
            X_train,
            y_train
        )

        # Predict
        y_pred = predict_model(
            model,
            X_test
        )

        # Evaluate
        metrics = evaluate_model(
            y_test,
            y_pred
        )

        results[name] = metrics

        print("MAE :", metrics["mae"])
        print("MSE :", metrics["mse"])
        print("RMSE:", metrics["rmse"])
        print("R²  :", metrics["r2"])


    # -----------------------------------------------------
    # COMPARE MODELS
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, metrics in results.items():

        print(f"\n{name}")

        print(
            f"MAE  : {metrics['mae']:.4f}"
        )

        print(
            f"MSE  : {metrics['mse']:.4f}"
        )

        print(
            f"RMSE : {metrics['rmse']:.4f}"
        )

        print(
            f"R²   : {metrics['r2']:.4f}"
        )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()