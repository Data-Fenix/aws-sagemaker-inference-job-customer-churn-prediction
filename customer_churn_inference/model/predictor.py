# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback

import flask
import boto3
import pandas as pd
import numpy as np
#import statsmodels.api as sm

#Temporarly remove this path and giving the hardcode path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
s3_client = boto3.client('s3')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class CustomerChurn(object):
    model = None  # Where we keep the model when it's loaded
    complete_model = None
    model_flag = False

    # @classmethod
    # def get_model(cls):
    #     """Get the model object for this instance, loading it if it's not already loaded."""
    #     if cls.complete_model == None:
    #         with open(os.path.join(model_path, 'model.pickle'), 'rb') as inp:
    #             cls.complete_model = pickle.load(inp)
    #     return cls.complete_model
    '''

    print("loading the stage config data from s3")
    
    @classmethod
    def getJsonData(cls, bucket_name, key_name):
        print("[LOG]", bucket_name,'---------')
        print("[LOG]", key_name,'--------------')

        s3_client = boto3.client('s3')
        csv_obj = s3_client.get_object(Bucket=bucket_name, Key=key_name)

        body = csv_obj['Body']

        json_string = body.read().decode('utf-8')
        json_content = json.loads(json_string)

        return cls.json_content

    '''
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
#        model_path = "s3://dlk-cloud-tier-9-training-ml-dev/prepaid-churn/2021-10-13/pipelines-xb88ksi5tynp-PrepaidChurn-trainin-YhUdrqjuke/output/"
#        if cls.model == None:
#            with open(os.path.join(model_path, "model.tar.gz"), "rb") as inp:
#                cls.model = pickle.load(inp)
#        return cls.model
        if cls.model == None:
            with open(os.path.join(model_path, "temp_dict.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model
    
#        if not cls.model_flag:
#            with open(os.path.join(model_path, 'temp_dict.pkl'), 'rb') as inp:
#                cls.complete_model = pickle.load(inp)
#                cls.model_flag = True
#        return cls.model_flag
   

    @classmethod
    def predict_proba(cls, df_prediction):
        
        print("Print Entered...")
        #print(input.head())
        #print(input.dtypes)
        '''
        print("enter your project name and enviornment here")
        project_name = "aws_workshop"
        env = 'dev'
        '''
        
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
        """
        print("setting the parameters for getJsonData function")
        config_bucket = f"dlk-cloud-tier-8-code-ml-{env}"
        config_s3_prefix_stage = f'config_files/stage_config/{project_name}/stage_config.json'

        print("calling the getJsonData function")
        config = cls.getJsonData(config_bucket,config_s3_prefix_stage)

        print("json script loaded successfully")
        """
        print("rename the columns of the dataset")
    
        df_prediction.columns = columns
        
        print(df_prediction.shape)
        print(df_prediction.columns)
        print(df_prediction.head())
        
        feature_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
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
        
        print("Assign the dataframe into X variable")
        X = df_prediction[feature_cols]
        print('number of rows', X.shape[0])
                
        print('model predictions starting')
        
        print("Assign the model into model variable")

        model = cls.get_model()
        
        print("Start the prediction")
        pred_y = model.predict_proba(X)[:,1]
        
        return pred_y


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = CustomerChurn.get_model()  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
        ##data = pd.read_csv(s, header=None, error_bad_lines=False)
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    predictions = CustomerChurn.predict_proba(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")