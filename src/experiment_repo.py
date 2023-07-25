import io
import logging
import os
import re
import shutil
import sys
import threading
from typing import List, Dict, Tuple, Union

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

logger = logging.getLogger('ray')


def get_all_s3_objects(s3: BaseClient, **base_kwargs):
    """
    Paginate through all objects in S3 bucket.

    Args:
        s3: S3 client
        **base_kwargs:

    Returns:
        generator over objects.
    """
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')


def extract_experiment_data(s) -> Tuple[str, int] | None:
    """
    Get experiment data from file name.

    Args:
        s: filename

    Returns:
        (experiment name, version)
    """
    match = re.match(r"^experiment-(.*?)-v(\d+)", s)
    if match:
        experiment_name = match.group(1).replace('-', '_')
        version = int(match.group(2))
        return experiment_name, version
    else:
        return None


def test_extract_experiment_data():
    assert extract_experiment_data("experiment-test_experiment-v3") == ('test_experiment', 3)
    assert extract_experiment_data("experiment-other_experiment-v10") == ('other_experiment', 10)
    assert extract_experiment_data("experiments-other_experiment-v10") is None


class NoFile(Exception):
    pass


class S3Interface:
    def __init__(self, repo_name: str, aws_access_key_id: str, aws_secret_access_key: str, region: str | None = None):
        """
        Args:
            repo_name: name of repo, e.g. 'recipe-improvement'
            aws_access_key_id: AWS key id
            aws_secret_access_key: AWS key
            region: String region to create bucket in, e.g., 'us-west-2'
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.repo_name = repo_name
        self.region = region
        self.s3_client: BaseClient = self._s3_client()
        if self.bucket_name not in self._all_buckets():
            self._create_bucket(self.bucket_name)

    def _create_bucket(self, bucket_name: str) -> bool:
        """
        Create an S3 bucket in the region of model repo instance.

        Args:
            bucket_name: str

        Returns:
            True iff creation successful
        """

        # Create bucket
        try:
            if self.region is None:
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                location = {'LocationConstraint': self.region}
                self.s3_client.create_bucket(Bucket=bucket_name,
                                             CreateBucketConfiguration=location)
        except ClientError as e:
            logger.error(f"Got error: {e} when creating model bucket: {bucket_name}.")
            return False
        return True

    def _all_buckets(self) -> List[str]:
        """
        Get all buckets in repo.

        Returns:
            list of all buckets.
        """
        response = self.s3_client.list_buckets()
        buckets: List[str] = []
        for bucket in response['Buckets']:
            buckets.append(bucket)
        return buckets

    def _s3_client(self) -> BaseClient:
        """
        Get S3 client.

        Returns:
            base S3 client.
        """
        if self.region is not None:
            return boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        else:
            return boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )

    def delete_object(self, object_name: str) -> bool:
        """
        Delete object from bucket.

        Args:
            object_name:

        Returns:
            True iff object deleted
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
        except ClientError as e:
            logger.error(f"Got error: {e} when deleting {object_name} from experiment bucket: {self.bucket_name}.")
            return False
        return True

    def get_all_bucket_files(self) -> List[str]:
        """
        Get the names of all files in bucket.

        Returns:
            list of file names in bucket.
        """
        files = []
        for obj in get_all_s3_objects(self.s3_client, Bucket=self.bucket_name):
            files.append(obj['Key'])
        return files

    def download_file(self, file_name_or_obj: Union[str, io.IOBase], object_name: str):
        """
        Download a file from bucket.

        Args:
            file_name_or_obj: File name or file-like object
            object_name: S3 object name

        Raises:
            ClientError if file not found
        """
        if isinstance(file_name_or_obj, str):
            with open(file_name_or_obj, 'wb') as f:
                self.s3_client.download_fileobj(self.bucket_name, object_name, f,
                                                Callback=ProgressPercentage(file_name_or_obj, percentage=False))
        elif isinstance(file_name_or_obj, io.IOBase):
            self.s3_client.download_fileobj(self.bucket_name, object_name, file_name_or_obj)
        else:
            raise ValueError("file_name_or_obj must be a string or a file-like object")

    def upload_file(self, file_name_or_obj: Union[str, io.IOBase], object_name: str) -> bool:
        """
        Upload to an S3 bucket.

        Args:
            file_name_or_obj: File name or file-like object
            object_name: S3 object name

        Returns:
            True if successfully uploaded
        """
        try:
            if isinstance(file_name_or_obj, str):
                response = self.s3_client.upload_file(file_name_or_obj, self.bucket_name, object_name,
                                                      Callback=ProgressPercentage(file_name_or_obj))
            elif isinstance(file_name_or_obj, io.IOBase):
                file_name_or_obj.seek(0)
                self.s3_client.upload_fileobj(file_name_or_obj, self.bucket_name, object_name)
            else:
                raise ValueError("file_name_or_obj must be a string or a file-like object")
        except ClientError as e:
            logger.error(f"Got error: {e} when uploading {object_name} to experiment bucket: {self.bucket_name}.")
            return False
        return True

    @property
    def bucket_name(self):
        return f"bayesian-optimiser-experiment-repo-{self.repo_name}".replace("_", "-")


class S3File:
    def __init__(self, s3_interface: S3Interface, path: str):
        self.s3_interface = s3_interface
        self.path = path

    def read(self) -> str:
        try:
            with io.BytesIO() as buffer:
                self.s3_interface.download_file(buffer, self.path)
                buffer.seek(0)
                return buffer.read().decode('utf-8')
        except ClientError as e:
            raise FileNotFoundError(f"Could not read file at path {self.path}") from e

    def write(self, data: str):
        with io.BytesIO(data.encode('utf-8')) as buffer:
            self.s3_interface.upload_file(buffer, self.path)

    def delete(self):
        self.s3_interface.delete_object(object_name=self.path)

    def exists(self) -> bool:
        return self.path in self.s3_interface.get_all_bucket_files()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class S3Bucket:
    def __init__(self, s3_interface: S3Interface):
        self.s3_interface = s3_interface

    def __getitem__(self, path: str) -> S3File:
        return S3File(self.s3_interface, path)


class ExperimentRepo(S3Interface):

    def get_all_experiments(self) -> Dict[str, List[int]]:
        """
        Get map of all experiments.

        Returns:
            (experiment_name) -> List[version: int]
        """
        experiment_data = dict()
        for name, version in filter(lambda x: x is not None,
                                    map(extract_experiment_data, self.get_all_bucket_files())):
            if name not in experiment_data:
                experiment_data[name] = []
            experiment_data[name].append(int(version))
        return experiment_data

    def store_experiment(self, experiment_directory: str, experiment_name: str, version: int):
        """
        Store a directory containing an experiment.

        Args:
            experiment_directory: experiment folder
            experiment_name: name of experiment
            version: version of experiment
        """
        file_to_upload = shutil.make_archive(
            base_name=self._experiment_object(experiment_name=experiment_name, version=version),
            format='bztar',
            root_dir=experiment_directory
        )
        if not self.upload_file(file_name_or_obj=file_to_upload,
                                object_name=self._experiment_object(experiment_name=experiment_name, version=version)):
            raise RuntimeError(f"Failed to store experiment {file_to_upload}.")

    def delete_experiment(self, experiment_name: str, version: int):
        """
        Deletes a given experiment.

        Args:
            experiment_name: experiment name
            version: version
        """
        object_name = self._experiment_object(experiment_name=experiment_name, version=version)
        if not self.delete_object(object_name=object_name):
            raise RuntimeError(f"Failed to delete experiment {object_name} from {self.bucket_name}.")

    def get_experiment(self, experiment_directory: str, experiment_name: str, version: int):
        """
        Get an experiment from S3 writing to the provided folder name.

        Args:
            experiment_directory: folder to open as.
            experiment_name: name of experiment ot look for.
            version: version to get.
        """
        object_name = self._experiment_object(experiment_name=experiment_name, version=version)
        file_name = f"{object_name}.tar.bz2"
        if not self.download_file(file_name_or_obj=file_name,
                                  object_name=object_name):
            raise NoFile(f"Failed to download {object_name}.")
        shutil.unpack_archive(filename=file_name, extract_dir=experiment_directory, format='bztar')

    def _experiment_object(self, experiment_name: str, version: int) -> str:
        """
        Get standard experiment name.

        Args:
            experiment_name: experiment name
            version: version

        Returns:
            str name
        """
        return f"experiment-{experiment_name.replace('_', '-')}-v{version}"


class ProgressPercentage(object):

    def __init__(self, filename, percentage=True):
        self._filename = filename
        if percentage:
            self._size = float(os.path.getsize(filename))
        else:
            self._size = None
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            if self._size is not None:
                percentage = (self._seen_so_far / self._size) * 100
                sys.stdout.write(
                    "\r%s  %s / %s  (%.2f%%)" % (
                        self._filename, self._seen_so_far, self._size,
                        percentage))
            else:
                sys.stdout.write(
                    "\r%s  %s B" % (
                        self._filename, self._seen_so_far))
            sys.stdout.flush()
