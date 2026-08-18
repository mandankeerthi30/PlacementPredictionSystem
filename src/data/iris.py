import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv(r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\src\data\IRIS.csv")
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

def bargraph(df):
    count=df["species"].value_counts()
    plt.bar(count.index, count.values)
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.title("Number of Samples in Each Species")
    plt.show();
def histogram(df):
    plt.hist(df["sepal_length"],bins=50)
    plt.xlabel("Sepal Length")
    plt.ylabel("Number of Samples")
    plt.title("Histogram of Sepal Length")
    plt.show()
def piechart(df):
    count=df["species"].value_counts()
    plt.pie(count,labels=count.index,autopct="%1.1f%%",startangle=90)

    plt.title("Pie Chart of Sepal Length")
    plt.show()
def scatterplot(df):
    plt.scatter(df["sepal_length"], df["sepal_width"])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Scatter Plot of Sepal Length")
    plt.show()


if __name__ == '__main__':
    df = load_data()
    eda(df)
    bargraph(df)
    histogram(df)
    piechart(df)
    scatterplot(df)