import json
import os

import pytest

from src.experiment_repo import ExperimentRepo, S3Interface, S3Bucket


def _test_experiment_repo():
    m = ExperimentRepo(
        repo_name='test',
        aws_access_key_id='None',
        aws_secret_access_key='None'
    )
    m._all_buckets()
    os.makedirs('test_dir', exist_ok=True)
    with open(os.path.join('test_dir', 'data'), 'w') as f:
        f.write('foobar')

    m.store_experiment('./test_dir', 'test-experiment', version=0)
    m.get_experiment('./test_dir_get', 'test-experiment', version=0)

    assert os.path.exists('./test_dir_get/data')
    with open('./test_dir_get/data', 'r') as f:
        assert f.read() == 'foobar'
    assert 'test_experiment' in m.get_all_experiments()

    # delete using - as - goes to _ automatically.
    m.delete_experiment('test-experiment', version=0)
    assert 'test_experiment' not in m.get_all_experiments()


def _test_bucket_reader():
    s3_interface = S3Interface(
        repo_name='test',
        aws_access_key_id='None',
        aws_secret_access_key='None'
    )
    s3_bucket = S3Bucket(s3_interface)

    data = json.dumps(dict(a=1))
    with s3_bucket['test_obj'] as f:
        f.write(data)
        assert 'test_obj' in s3_interface.get_all_bucket_files()
        data_out = f.read()
        assert data_out == data
        f.delete()

    assert 'test_obj' not in s3_interface.get_all_bucket_files()

    with s3_bucket['test_obj'] as f:
        with pytest.raises(FileNotFoundError):
            f.read()
