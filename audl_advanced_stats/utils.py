"""General functions that are used in different classes."""

import boto3
import botocore
import time
from google.cloud import storage
from os import environ, makedirs
from os.path import join, exists, dirname, relpath
from pathlib import Path
from .constants import *


def get_data_path(data_path="data"):
    """Return the file path to the data folder."""
    return data_path


def get_json_path(database_path, folder):
    """Return the file path to the JSON game data."""
    return join(database_path, folder)


def get_games_path(database_path, folder):
    """Return the file path to the file that contains all game data."""
    return join(database_path, folder)


def upload_to_bucket(file_path, platform="GCP"):
    """Upload a file to either GCP or AWS."""
    assert platform in ["GCP", "AWS"], "Must use a platform of AWS or GCP."

    if platform == "GCP":
        upload_to_bucket_gcp(file_path)
    elif platform == "AWS":
        upload_to_bucket_aws(file_path)


def download_from_bucket(file_path, platform="GCP"):
    """Download a file from either GCP or AWS."""
    assert platform in ["GCP", "AWS"], "Must use a platform of AWS or GCP."

    if platform == "GCP":
        download_from_bucket_gcp(file_path)
    elif platform == "AWS":
        download_from_bucket_aws(file_path)


def download_folder_from_bucket(file_path, platform="GCP"):
    """Download contents of a folder from either GCP or AWS."""
    assert platform in ["GCP", "AWS"], "Must use a platform of AWS or GCP."

    if platform == "GCP":
        download_folder_from_bucket_gcp(file_path)
    elif platform == "AWS":
        download_folder_from_bucket_aws(file_path)


def upload_to_bucket_gcp(file_path):
    """Copy a local file to GCP bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)


def download_from_bucket_gcp(file_path):
    """Copy a file from GCP bucket to local directory."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    blob.download_to_filename(file_path)


def download_folder_from_bucket_gcp(file_path):
    """Download the contents of a folder directory from GCP bucket.

    Args:
        file_path: a relative or absolute directory path in the local file system

    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=file_path)
    for blob in blobs:
        filename = blob.name
        blob.download_to_filename(filename)


def upload_to_bucket_aws(file_path):
    """Copy a local file to AWS bucket."""
    try:
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=environ["AWS_SECRET_ACCESS_KEY"],
        )
        s3.Bucket(BUCKET_NAME).upload_file(
            file_path.replace("\\", "/"),
            file_path.replace("\\", "/"),
        )
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # If we end up here, it means the file does not exist on AWS
            return False
        else:
            return None


def download_from_bucket_aws(file_path):
    """Copy a file from AWS bucket to local directory."""
    try:
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=environ["AWS_SECRET_ACCESS_KEY"],
        )
        try:
            s3.Bucket(BUCKET_NAME).download_file(
                file_path.replace("\\", "/"),
                file_path.replace("\\", "/"),
            )
        except (PermissionError, FileExistsError) as e:
            if not Path(file_path).is_file():
                time.sleep(1)
                s3.Bucket(BUCKET_NAME).download_file(
                    file_path.replace("\\", "/"),
                    file_path.replace("\\", "/"),
                )
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # If we end up here, it means the file does not exist on AWS
            return False
        else:
            return None


def download_folder_from_bucket_aws(file_path):
    """Download the contents of a folder directory from AWS bucket.

    Args:
        file_path: a relative or absolute directory path in the local file system

    """
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=environ["AWS_SECRET_ACCESS_KEY"],
    )
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=file_path):
        target = (
            obj.key
            if file_path is None
            else join(file_path, relpath(obj.key, file_path))
        )
        if not exists(dirname(target)):
            makedirs(dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)
