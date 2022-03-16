#import only the necessary libraries

import glob 
import pyarrow.parquet as pq
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import argparse
import os
import boto3
import json

print("import your necessary libraries in here") 
import pandas as pd 
import numpy as np


if __name__ == "__main__":
    
    input_data_path = os.path.join("/opt/ml/processing/input", "data_output.csv.gz.out")
    
    input_data_path1 = os.path.join("/opt/ml/processing/input1", "data_output.csv.gz")

    print("reading input data from {}".format(input_data_path))
    
    df1 = pd.read_csv(input_data_path,header=None)
    
    print('loading the preprocess dataset')
    df= pd.read_csv(input_data_path1,header=None)
    
    ################### Enter your own script in here #######################
      

    print('renaming the prediction output dataset')      
    df1.rename(columns={0:'churn_prop'}, inplace=True)
    
    print('columns selecting')
    columns = ['customerID','SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    
    print('renaming the colums')
    df.columns = columns
    
    print('add churn probabilty variable to preprocess dataset')
    df['Churn_probability']=df1['churn_prop']

    data_cmp_df = df[['customerID','Churn_probability']]
    
    print('sorting the first five hundred customers')
    data_cmp_df=data_cmp_df.sort_values("Churn_probability", ascending=False).head(500)
          
    #################### End of the code ##############################   
    print("saving the dataframe")

    # Saving outputs.(must be header = False)
    inf_features_output_path = os.path.join("/opt/ml/processing/post_output1", "data_final_output.csv.gz")
    
    
    print("saving training features to {}".format(inf_features_output_path))
    pd.DataFrame(data_cmp_df).to_csv(inf_features_output_path, compression='gzip',index=False)

    print("successfully completed the post-processing job")
