import pandas as pd
def load_data():
 df=pd.read_csv(r'C:\Users\YASASWINI\PycharmProjects\Placementpredictionsystem\data\placement_data (1).csv')
 return df
def get_summary(df):
 return {
     "rows":df.shape[0],
     "columns":df.shape[1],
     "target":df.columns[-3]
 }
if __name__=="__main__":
    df=load_data()
    print(get_summary(df))
    print(df.head(5))