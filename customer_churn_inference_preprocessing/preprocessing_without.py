import boto3
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import argparse
import os
import warnings
warnings.simplefilter(action='ignore')
import json

print("import your necessary libraries in here") 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


print("enter your own functions in the bellow space")
def change_format(df):
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    
    return df

def missing_value(df):
    print("count of missing values: (before treatment)", df.isnull().sum())
    
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    print("count of missing values: (before treatment)", df.isnull().sum())
    print("missing values successfully replaced")
    return df

def data_manipulation(df):
    df = df.drop(['customerID'], axis = 1)
    
    return df

def cat_encoder(df, variable_list):
    dummy = pd.get_dummies(df[variable_list], drop_first = True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(df[cat_var], axis = 1, inplace = True)
    
    print("encoded successfully")
    return df

def scaling(X):  
    min_max=MinMaxScaler()
    X=pd.DataFrame(min_max.fit_transform(X),columns=X.columns)
    
    return X

print("successfully loaded our own defined functions")


if __name__ == "__main__":

    input_data_path = os.path.join("/opt/ml/processing/input", "data_inference.csv")

    print("reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)

    print("rename the columns in the dataset")
    df.columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                             'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                             'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    
    #################################### Enter your own script in here ###########################################################################
    
    print("defining the list of categorical variables")
    cat_var = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod']
    
    print("calling our own defined function")
    df = missing_value(change_format(df))
    df = cat_encoder(df, cat_var)
    
    df1=df['customerID']
    df=df.drop(columns=['customerID'])

    
    
    print("scaling the dataset")
    df = scaling(df)
    idx=0
    df.insert(loc=idx, column='customerID', value=df1)

    
    #################################### End of the code #########################################################################################
    
    print("saving the outputs")
    X_output_path = os.path.join("/opt/ml/processing/output1", "data_output.csv.gz")
        
    print("saving output to {}".format(X_output_path))
    pd.DataFrame(df).to_csv(X_output_path, header=False,index=False)
    
    
    print("successfully completed the preprocessing job for inference")
