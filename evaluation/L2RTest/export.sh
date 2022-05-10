#!/usr/bin/env bash

./build.sh

docker save l2rtest | gzip -c > L2RTest.tar.gz
