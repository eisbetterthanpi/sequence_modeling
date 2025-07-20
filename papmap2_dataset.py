# @title papmap2 me
# https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
# https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
!wget https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip -O pamap2.zip
!unzip pamap2.zip
!unzip PAMAP2_Dataset.zip

import os
import numpy as np
import pandas as pd
# https://github.com/EdnaEze/Physical-Activity-Monitoring/blob/main/DSRM-Edna.ipynb

activities = {0:'transient', 1:'lying', 2:'sitting', 3:'standing', 4:'walking', 5:'running', 6:'cycling', 7:'Nordic_walking', 9:'watching_TV', 10:'computer_work', 11:'car driving', 12:'ascending_stairs', 13:'descending_stairs', 16:'vacuum_cleaning', 17:'ironing', 18:'folding_laundry', 19:'house_cleaning', 20:'playing_soccer', 24:'rope_jumping'}
all_columns = ["time", "activity", "heartrate", 'handTemperature', 'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3', 'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4', 'chestTemperature', 'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 'chestGyro1', 'chestGyro2', 'chestGyro3', 'chestMagne1', 'chestMagne2', 'chestMagne3', 'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4', 'ankleTemperature', 'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 'ankleMagne1', 'ankleMagne2', 'ankleMagne3', 'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

dataset = pd.DataFrame()

# path = '/content/OpportunityUCIDataset/dataset'
path = '/content/PAMAP2_Dataset/Protocol/'

usr_lst = os.listdir(path)
for file in os.listdir(path):
# for file, subject_id in zip(file_names, subject_id):
    df = pd.read_table(path+file, header=None, sep='\s+')
    df.columns = all_columns
    df['subject'] = file
    dataset = pd.concat([dataset, df], ignore_index=True)

y = dataset['subject'].unique()
y.sort()

df_train = dataset[dataset['subject'].isin(y[:int(.7*len(y))])]
df_test = dataset[dataset['subject'].isin(y[-int(.3*len(y)):])]

def make_Xy(dataset):
    anss = [y for _, y in dataset.groupby(['subject', 'activity'])]
    ans = []
    for x in anss:
        if len(x) > 1000: # only keep sequences with more than 1000 samples
            ans.append(x)
    y_train = [df['activity'].iloc[0] for df in ans]
    # y_train = [df['subject'].iloc[0] for df in ans]
    # X_train = [df.drop(['subject', 'activity','time'], axis=1) for df in X_train]
    X_train = [df.drop(['subject', 'activity','time'], axis=1) for df in ans]
    # X_train = [df.interpolate(method='index', axis=0, limit_direction='both') for df in ans]

    X_train = [df.apply(pd.to_numeric, errors='coerce') for df in X_train] # Convert non-numeric data in dataset to numeric. errors='coerce': replace all non-numeric values with NaN.
    X_train = [df.interpolate(method='index', axis=0, limit_direction='both') for df in X_train] # replace NaN by interpolating

    # X_train = [df.interpolate(method='values', axis=0, limit_direction='both') for df in ans]
    # data.reset_index(drop=True, inplace=True) # make row ind start from 0
    return X_train, y_train

X_train, y_train = make_Xy(df_train)
X_test, y_test = make_Xy(df_test)
