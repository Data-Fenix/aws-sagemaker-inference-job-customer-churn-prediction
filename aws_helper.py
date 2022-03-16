import json
import boto3


def s3_put(bucket_name, file_path, file_name, payload):
    """
    Use for Uploading a file as StringIO object
    :param str bucket_name: s3 bucket name
    :param str file_path: string name, sub folders are allowed
    :param str file_name: name that you need to be in destination S3 Bucket location
    :param StringIO payload: CSV to be uploaded as stringIO Object
    :return:
    """
    s3r = boto3.resource('s3')
    response = s3r.Object(bucket_name, file_name).put(Body=payload.getvalue())

    s3c = boto3.client('s3')

    response = s3c.put_object(Bucket=bucket_name, Key=file_path + file_name, Body=payload)
    status_code = response.get("ResponseMetadata").get("HTTPStatusCode")
    if status_code == 200:
        return True, status_code
    else:
        return False, status_code

###
def s3_download(bucket_name, file_path, file_name):
    """
    Download a file from an S3 Bucket

    :param str bucket_name: Pass only the bucket name without any "s3://" or "s3a://", just the name
    :param str file_path: Path of your file. Make sure to include the name of your file not just the folder path
    :param str file_name: Name of the file you need it to be, with it's extension ex. my_file.txt
    :return bool:
    """
    from urllib.parse import unquote_plus, unquote
    from os import listdir

    s3c = boto3.client('s3')
    file_path = unquote_plus(unquote(file_path))
    file_extension = file_path.split(".")[-1]
    try:
        s3c.download_file(bucket_name, file_path, file_name)
        print("[mLOG] Files in /tmp/ \n", listdir("/tmp/"))
        return True
    except Exception as e:
        print(f"[ERR] {type(e)} MORE: {str(e)}")
        print("[mLOG] Files in /tmp/ \n", listdir("/tmp/"))
        return False


def s3_upload(bucket_name, s3_path, file_to_upload):
    """

    :param str bucket_name: Pass only the bucket name without any "s3://" or "s3a://", just the name
    :param str s3_path: Path of your s3 folder where you want your file dumped
    :param str file_to_upload: local file path
    :return:
    """
    s3c = boto3.client('s3')

    try:
        s3c.upload_file(file_to_upload, bucket_name, s3_path)
        return True
    except Exception as e:
        print(f"[ERR] File upload failed! TYPE: {type(e)} MORE: {str(e)}")
        return False


def s3_list_objects(bucket_name, prefix=False):
    s3c = boto3.client('s3')
    response = []

    try:
        if prefix:
            response = s3c.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        else:
            response = s3c.list_objects_v2(Bucket=bucket_name)
        return response
    except Exception as e:
        print(f"[ERR] Something went wrong there. More:{type(e)} : {str(e)}")
        return response


def send_sns(region, topic_arn, subject, message):
    sns_client = boto3.client("sns", region_name=region)

    print("[mLOG] Sending SNS")

    response = sns_client.publish(
        TopicArn=topic_arn,
        Message=message,
        Subject=subject
    )

    return response


def get_secret(secret_name, region_name):
    import base64
    from botocore.exceptions import ClientError

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    secret = None

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']

    return secret


def read_and_print_metrics():
    import json
    train_accuracy = 0.68
    train_recall = 0.27

    internal_metrics = {"train_accuracy": train_accuracy, "train_recall": train_recall}

    with open("metrics.json") as fp:
        metrics = json.load(fp)

    for k, v in internal_metrics.items():
        print(metrics.get(k).format(v))

    return True

# ------------------------- Parameterization of Training SQL statements -----------------

# tables_src = {
#     "table_name": "s3",
#     "table_name2": "snowflake"
# }


# from train_oracle_sql import train_queries

# # mapping_1 = dict(EDW_TGT="NEW_SCHEMA", DIM_CUSTOMER_ACCOUNT="NEW_TABLENAME", FACT_ACCT_ACTIVITY="NEW_TABLE2")

# for query in train_queries.list_of_all_query_vars:
#     print(query.format(**training_substitution_map))

# with open("train_oracle_sql/train_queries.py", 'r') as fp:
#     file_string = fp.readlines()[0]
#     print(file_string)

# # ---------------- Scan all Train SQLs for Schemas and Tables --------------------

# import os

# directories = [obj for obj in os.listdir(".") if os.path.isdir(obj) and obj.startswith('pred')]

# for dir in directories:
#     all_files = [file for file in os.listdir(dir) if file.endswith(".sql")]
#     print(all_files)

#     for file in all_files:
#         with open(dir + "/" + file, 'r') as fp:
#             print(file.center(50, "-"))
#             for no, line in enumerate(fp.readlines()):
#                 clean_sql = line.split("--")[0].strip()
#                 if "from" in clean_sql.lower():
#                     print(no, clean_sql)
#                 if "join" in clean_sql.lower():
#                     print(no, clean_sql)
#     break

# # ----------------- SPARK ---------------------

# from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName("Sample PySpark App1").getOrCreate()
# my_df = spark.read.csv("/home/mister-t/PycharmProjects/LearnMachineLearning/DataKit/50_Startups.csv", header=True)
# my_df.show(5)
# my_df.write.csv(path="new.csv", mode="overwrite", sep="|", header=True)


# # ======================

# my_df.coalesce(1).write.csv("file2.csv", mode="overwrite", sep=",", header=True)
# my_df.createOrReplaceGlobalTempView("sample_table")

# spark.sql("select * from global_temp.sample_table")
# df2 = spark.sql("select * from global_temp.sample_table")
# df2.show(5)

# my_df.createOrReplaceTempView("sample_temp_table")
# df3 = spark.sql("select * from sample_temp_table")
# df3.show(5)

# # =====================

# with open("/home/mister-t/Downloads/DDL_29042021.sql", 'r') as fp:
#     for no, line in enumerate(fp.readline()):
#         print(no)
#         if "CREATE TABLE" in line:
#             print(no, line)

