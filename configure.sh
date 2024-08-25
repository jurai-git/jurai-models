#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <host> <user> <password> <name> <table>"
    exit 1
fi

touch .env

echo "DB_HOST=$1" >> .env
echo "DB_USER=$2" >> .env
echo "DB_PASSWORD=$3" >> .env
echo "DB_NAME=$4" >> .env
echo "DB_TABLE=$5" >> .env

PROJECT_PATH=$(pwd)
echo "PROJECT_PATH=$PROJECT_PATH" >> .env
echo "DATASET_PATH=$PROJECT_PATH/datasets" >> .env

echo ".env file created"
