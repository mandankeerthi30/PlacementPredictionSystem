import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import (
    split_x_data,
    identify_features,
    handle_missing_values,
    standardize,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# Find Optimal K - Elbow Method
# --------------------------------------------------
def find_optimal_k(X):

    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        model.fit(X)

        wcss.append(model.inertia_)

    print("\n----- WCSS Values -----")

    for k, value in zip(range(1, 11), wcss):
        print(f"K = {k} : {value:.2f}")

    # Elbow graph
    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, 11),
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(range(1, 11))
    plt.grid(True)

    plt.show()

    return wcss


# --------------------------------------------------
# Create K-Means Model
# --------------------------------------------------
def create_model(k):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------
def train_model(model, X):

    labels = model.fit_predict(X)

    return model, labels


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
def evaluate_model(model, X, labels):

    print("\n----- Model Evaluation -----")

    print("Inertia (WCSS):")
    print(f"{model.inertia_:.2f}")

    print("\nNumber of iterations:")
    print(model.n_iter_)

    # Silhouette score needs at least 2 clusters
    silhouette = silhouette_score(X, labels)

    print("\nSilhouette Score:")
    print(f"{silhouette:.4f}")

    return silhouette


# --------------------------------------------------
# Display Clusters
# --------------------------------------------------
def display_clusters(X, labels, model):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="viridis"
    )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering")

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    plt.legend()

    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    df = load_data()

    print("Original shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


    # --------------------------------------------------
    # 2. Separate input features
    # --------------------------------------------------
    X = split_x_data(
        df,
        drop_columns=[
            "StudentID",
            "IsAnomaly",
            "Salary Package",
            "PlacementStatus"
        ]
    )

    print("\nX shape:")
    print(X.shape)


    # --------------------------------------------------
    # 3. Identify features
    # --------------------------------------------------
    numerical_features, categorical_features = identify_features(X)

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)


    # --------------------------------------------------
    # 4. Define encoding features
    # --------------------------------------------------
    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]


    # --------------------------------------------------
    # 5. Handle missing values
    # --------------------------------------------------
    X, _, imputer = handle_missing_values(
        X,
        X.copy(),
        numerical_features
    )

    print("\nMissing values handled")


    # --------------------------------------------------
    # 6. Standardize numerical features
    # --------------------------------------------------
    X, _, scaler = standardize(
        X,
        X.copy(),
        numerical_features
    )

    print("Standardization completed")


    # --------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------
    X, _, one_hot_encoder = one_hot_encode_data(
        X,
        X.copy(),
        one_hot_features
    )

    print("One-hot encoding completed")


    # --------------------------------------------------
    # 8. Ordinal Encoding
    # --------------------------------------------------
    X, _, ordinal_encoder = ordinal_encode_data(
        X,
        X.copy(),
        ordinal_features
    )

    print("Ordinal encoding completed")


    # --------------------------------------------------
    # 9. Keep only numerical columns
    # --------------------------------------------------
    X = X.select_dtypes(include=["number"])

    print("\nFinal X shape:")
    print(X.shape)

    print("\nFinal features:")
    print(X.columns.tolist())


    # --------------------------------------------------
    # 10. Check missing values
    # --------------------------------------------------
    print("\nTotal missing values:")
    print(X.isnull().sum().sum())


    # --------------------------------------------------
    # 11. Find Optimal K
    # --------------------------------------------------
    find_optimal_k(X)


    # --------------------------------------------------
    # 12. Select K
    # --------------------------------------------------
    # Change this value after observing the elbow graph
    k = 3

    print("\nSelected K:")
    print(k)


    # --------------------------------------------------
    # 13. Create Model
    # --------------------------------------------------
    model = create_model(k)


    # --------------------------------------------------
    # 14. Train Model
    # --------------------------------------------------
    model, labels = train_model(
        model,
        X
    )

    print("\nK-Means training completed")


    # --------------------------------------------------
    # 15. Add cluster labels
    # --------------------------------------------------
    df["Cluster"] = labels

    print("\nCluster Counts:")
    print(
        df["Cluster"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------
    # 16. Evaluate Model
    # --------------------------------------------------
    evaluate_model(
        model,
        X,
        labels
    )


    # --------------------------------------------------
    # 17. Display Clusters
    # --------------------------------------------------
    display_clusters(
        X,
        labels,
        model
    )


# --------------------------------------------------
# Run Program
# --------------------------------------------------
if __name__ == "__main__":
    main()