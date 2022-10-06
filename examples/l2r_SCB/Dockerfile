FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3


WORKDIR /l2r

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY features ./features
COPY foerstner.py .
COPY inference_paired_test.py .
COPY inference_test.py .
COPY train_registration_paired.py .
COPY train_registration.py .
COPY train_segment.py .
COPY utils.py .

COPY test.sh .
COPY train.sh .

RUN chmod +x test.sh
RUN chmod +x train.sh




