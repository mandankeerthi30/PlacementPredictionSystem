from src.data.load_data import load_data

from src.data.preprocess import (
    split_data,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from sklearn import tree


# =========================================================
# 1. CREATE DECISION TREE MODEL
# =========================================================

def create_model():

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )

    return model


# =========================================================
# 2. TRAIN MODEL
# =========================================================

def train_model(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    print("\nTrained Model")

    return model


# =========================================================
# 3. EVALUATE MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    # Make predictions
    y_pred = model.predict(
        X_test
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:")
    print(accuracy)

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return y_pred


# =========================================================
# 4. DISPLAY DECISION TREE
# =========================================================

def display_tree(model, feature_names):

    plt.figure(
        figsize=(20, 12)
    )

    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title(
        "Decision Tree Model"
    )

    plt.show()


# =========================================================
# 5. MAIN
# =========================================================

def main():

    # =====================================================
    # LOAD DATA
    # =====================================================

    data = load_data()

    print("Original Dataset Shape:")
    print(data.shape)

    print("\nOriginal Dataset Columns:")
    print(data.columns.tolist())


    # =====================================================
    # SPLIT DATA
    # =====================================================

    X_train, X_test, y_train, y_test = split_data(
        data,
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

    print("\nTraining Columns:")
    print(X_train.columns.tolist())


    # =====================================================
    # NUMERICAL FEATURES
    # =====================================================

    numerical_features = [

        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "SGPA_Sem6",
        "SGPA_Sem7",
        "SGPA_Sem8",

        "CGPA",

        "AttendancePercent",

        "Internships",

        "Projects",

        "Workshops",

        "Certifications",

        "Publications",

        "AptitudeTestScore",

        "SoftSkillsRating",

        "CodingTestScore",

        "MockInterviewScore"
    ]


    # =====================================================
    # NOMINAL CATEGORICAL FEATURES
    # =====================================================

    one_hot_features = [

        "Gender",

        "City",

        "CollegeTier",

        "Stream",

        "Specialisation",

        "Hostel",

        "HistoryOfBacklogs",

        "ExtraCurricular"
    ]


    # =====================================================
    # ORDINAL FEATURES
    # =====================================================

    ordinal_features = [

        "CGPA_Tier"
    ]


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    X_train, X_test, imputer = handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print(
        "\nMissing value handling completed."
    )


    # =====================================================
    # ONE-HOT ENCODING
    # =====================================================

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )

    print(
        "One-hot encoding completed."
    )


    # =====================================================
    # ORDINAL ENCODING
    # =====================================================

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )

    print(
        "Ordinal encoding completed."
    )


    # =====================================================
    # FINAL DATA INFORMATION
    # =====================================================

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)


    print("\nFinal Feature Names:")

    feature_names = X_train.columns.tolist()

    print(feature_names)


    # =====================================================
    # CREATE MODEL
    # =====================================================

    model = create_model()


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model = train_model(
        model,
        X_train,
        y_train
    )


    # =====================================================
    # EVALUATE MODEL
    # =====================================================

    y_pred = evaluate_model(
        model,
        X_test,
        y_test
    )


    # =====================================================
    # DISPLAY DECISION TREE
    # =====================================================

    display_tree(
        model,
        feature_names
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    main()