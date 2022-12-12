import pandas as pd
import matplotlib.pyplot as plt
#import src.data_proc as data_proc
import plotly.express as px

filename ="C:/NEU/Machine Learning/Project - zillow/1000sample2016.xlsx"
filename2 = "C:/NEU/Machine Learning/Project - zillow/properties_2016/properties_2016.csv"
df = pd.read_excel("C:/NEU/Machine Learning/Project - zillow/1000sample2016.xlsx",dtype={
        'propertycountylandusecode': str})

def load_training_data(file_name):
    return pd.read_csv(file_name)


"""
    Load house properties data
"""
def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, dtype={
        'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str
    })

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop

df.dropna(subset=['calculatedfinishedsquarefeet'], inplace = True)
#df=df.dropna()
print(df.info())

prop_2016 = load_properties_data(filename2)
# print("Number of properties: {}".format(len(prop_2016)))
# print(prop_2016.poolcnt)
# df = prop_2016#[prop_2016.poolcnt >= 0]
# plt.scatter(df.latitude, df.longitude, s=1)
# plt.show()
training_2016 = load_training_data("C:/NEU/Machine Learning/Project - zillow/train_2016_v2/train_2016_v2.csv")
print("\n", training_2016.head())
train_2016 = training_2016.merge(how='left', right=prop_2016, on='parcelid')
train_2016.head(10)

print(train_2016['yearbuilt'].describe())
train_2016.loc[abs(train_2016['yearbuilt']) < 300000, 'yearbuilt'].hist(bins=70)
plt.show()
threshold = 0.55
print("{} training examples in total".format(len(train_2016)))
print("{} with abs(logerror) > {}".format((abs(train_2016['logerror']) > threshold).sum(), threshold))

train_2016 = train_2016[abs(train_2016.logerror) <= threshold]

datetime = pd.to_datetime(train_2016.transactiondate).dt
train_2016['month'] = datetime.month
train_2016['quarter'] = datetime.quarter
print(train_2016.groupby('month')['month', 'logerror'].median())
print(train_2016.groupby('quarter')['quarter', 'logerror'].median())

missing_df = train_2016.isnull().sum()
missing_df_more_than_30 = missing_df[missing_df.values>(0.3*len(train_2016 ))].sort_values().reset_index().rename(columns={"index":"Features",0:"Total_NaN_values"})
print('we are having about {} columns with more than 30% of missing data'.format(len(missing_df_more_than_30)))
fig = px.bar(missing_df_more_than_30,x=missing_df_more_than_30.Features,y=missing_df_more_than_30.Total_NaN_values,title='Columns with than 30% missing values')
fig.show()