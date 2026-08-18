from src.data.load_data import load_data
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- BASIC EDA ---------------- #

def basic_eda(df):
    print("First 5 Rows")
    print(df.head())

    print("\nLast 5 Rows")
    print(df.tail())

    print("\nRows 25 to 35")
    print(df.iloc[25:35])

    print("\nColumn Names")
    print(df.columns)

    print("\nData Types")
    print(df.dtypes)

    print("\nDataset Information")
    df.info()

    print("\nMinimum Values")
    print(df.min())

    print("\nMaximum Values")
    print(df.max())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nNull Values")
    print(df.isnull().sum())

    count = df["PlacementStatus"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.bar(count.index, count.values)
    plt.title("Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\placement_status_barchart.png")
    plt.show()
    plt.close()


# ---------------- UNIVARIATE ---------------- #

def univariate(df):

    # Histogram
    plt.figure(figsize=(6, 5))
    plt.hist(df["CGPA"], bins=10)
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\Histogram.png")
    plt.show()
    plt.close()

    # Pie Chart
    gender_count = df["Gender"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.pie(
        gender_count,
        labels=gender_count.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Gender Distribution")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\Pie Chart.png")
    plt.show()
    plt.close()


# ---------------- BIVARIATE ---------------- #

def bivariate(df):

    # Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"])
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\scatterplot.png")
    plt.show()
    plt.close()

    # Box Plot
    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [placed, not_placed],
        tick_labels=["Placed", "Not Placed"]
    )
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\boxplot.png")
    plt.show()
    plt.close()


# ---------------- MULTIVARIATE ---------------- #

def multivariate(df):

    # Correlation of selected columns
    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]

    correlation = data.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\heatmap.png")
    plt.show()
    plt.close()

    # Complete Correlation Matrix
    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Complete Correlation Matrix")
    plt.savefig(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\app\static\charts\results\heatmap2.png")
    plt.show()
    plt.close()


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    df = load_data()

    # Uncomment the function you want to execute

    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)