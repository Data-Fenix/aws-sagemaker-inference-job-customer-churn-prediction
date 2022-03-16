FROM ubuntu:20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update OS
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1


RUN apt-get update && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-dev \
     python3-setuptools \
     nginx \
     && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get -y install gcc g++ make 
    
ARG PIP=pip3

WORKDIR /
    
# Installing lightgbm, tqdm
RUN ${PIP} install --no-cache --upgrade \
            pandas flask gunicorn \
            numpy==1.19.5 \
            statsmodels \
            scikit-learn \
            xgboost \
            boto3 \
            scipy
            
# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache --upgrade \
    multi-model-server sagemaker-inference


COPY model/* /opt/ml/code/
WORKDIR /opt/ml/code

RUN chmod +x /opt/ml/code/serve.py

# Defines train.py as script entry point

ENTRYPOINT ["python3", "/opt/ml/code/serve.py"]