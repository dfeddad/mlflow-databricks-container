from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mlflow-databricks-container",
    version="0.1.0",
    description="Plugin that provides backend implementation for running MLFlow project on Databricks Container without uploading project files to DBFS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Djamel Feddad",
    author_email="djameleddine.feddad@farfetch.com",
    url="https://github.com/dfeddad/mlflow-databricks-container",
    packages=find_packages(),
    install_requires=[
        "mlflow",
    ],
    entry_points={
        "mlflow.project_backend": [
            "databricks-container=mlflow_databricks_container.databricks_container_backend:databricks_container_backend_builder"
        ]
    },
)