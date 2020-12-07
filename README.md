# mlflow-databricks-container

This repository provides a MLflow Plugin that provides backend implementation for running MLFlow project on Databricks Container without uploading project files to DBFS.

## Installation 

User can simply install this plugin by running `pip install mlflow-databricks-container` inside mlflow environment. MLflow will register this plugin as an entrypoint with `databricks-container` backend.

## Example usage

To run an MLflow project, use the command:

`mlflow run <uri> -b databricks-container --backend-config <json-new-cluster-spec>`