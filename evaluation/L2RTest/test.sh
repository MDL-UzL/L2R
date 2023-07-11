#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create l2rtest-output

docker run --platform linux/amd64 --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v l2rtest-output:/output/ \
        l2rtest

docker run --platform linux/amd64 --rm \
        -v l2rtest-output:/output/ \
        python:3.9-slim cat /output/metrics.json | python -m json.tool

docker volume rm l2rtest-output
