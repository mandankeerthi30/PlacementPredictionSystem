import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv(r"C:\Users\YASASWINI\PycharmProjects\Placementpredictionsystem\src\data\IRIS.csv")
    return df

def eda(df):
    print("First 5 rows")
    print(df.head())

    print("Last 5 rows")
    print(df.tail())

    print("25 to 35 rows")
    print(df.iloc[25:35])

    print("Column names")
    print(df.columns)

    print("Data types")
    print(df.dtypes)

    print("Complete information")
    df.info()

    print("Min")
    print(df.min())

    print("Max")
    print(df.max())

    print("Duplicates")
    print(df.duplicated().sum())

    print("Null values")
    print(df.isnull().sum())

    count = df["species"].value_counts()

    plt.figure(figsize=(6,5))
    plt.bar(count.index, count.values)
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.title("Number of Samples in Each Species")
    plt.show()


if __name__ == '__main__':
    df = load_data()
    eda(df)