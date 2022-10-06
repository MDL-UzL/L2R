FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3

WORKDIR /l2r

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY test.sh .
RUN chmod +x test.sh
COPY zerofield.py .
RUN chmod +x zerofield.py

ENTRYPOINT ./test.sh $1 -x