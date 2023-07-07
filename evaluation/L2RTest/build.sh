#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo "Building docker image..."
docker build --platform linux/amd64 -t l2rtest "$SCRIPTPATH"
