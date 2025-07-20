# @title WISDM me
!wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
!tar -xzf WISDM_ar_latest.tar.gz
path = '/content/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

with open(path, 'r') as f:
    processedList = f.read().replace(' ', '').replace(',\n', ';\n').replace(',;', ';').replace('\n', '').replace(';','\n')
processedList = [p.split(',') for p in processedList.split('\n')]

# print(df.isna().sum())

import pandas as pd
columns = ['subject', 'activity','time','x','y','z']
dataset = pd.DataFrame(data = processedList, columns = columns)

y = dataset['subject'].unique()
y.sort()

df_train = dataset[dataset['subject'].isin(y[:int(.7*len(y))])]
df_test = dataset[dataset['subject'].isin(y[-int(.3*len(y)):])]

def make_Xy(dataset):
    ans = [y for _, y in dataset.groupby(['subject', 'activity'])]
    y_train = [df['activity'].iloc[0] for df in ans]
    # y_train = [df['subject'].iloc[0] for df in ans]
    X_train = [df.drop(['subject', 'activity','time'], axis=1) for df in ans]
    X_train = [df.apply(pd.to_numeric, errors='coerce') for df in X_train] # Convert non-numeric data in dataset to numeric. errors='coerce': replace all non-numeric values with NaN.
    X_train = [df.interpolate(method='index', axis=0, limit_direction='both') for df in X_train] # replace NaN by interpolating
    # X_train = [df.interpolate(method='index', axis=0, limit_direction='both') for df in ans]
    return X_train, y_train

X_train, y_train = make_Xy(df_train)
X_test, y_test = make_Xy(df_test)
