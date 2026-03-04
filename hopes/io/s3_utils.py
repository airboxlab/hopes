import subprocess
from io import StringIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")


def s3_prefix_has_objects(s3_prefix: str) -> bool:
    # Returns True if the prefix contains at least one object
    res = subprocess.run(["aws", "s3", "ls", s3_prefix], capture_output=True, text=True)
    if res.returncode != 0:
        # If listing fails (permissions / not found), treat as empty to attempt sync
        return False
    return len(res.stdout.strip()) > 0


def split_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key


def s3_prefix_exists_and_has_objects(s3_prefix_uri: str) -> bool:
    """True if there is at least one object under the given S3 prefix.

    Works for "folder exists" semantics (S3 prefixes).
    """
    bucket, prefix = split_s3_uri(s3_prefix_uri)
    res = subprocess.run(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return False
    return '"Key":' in res.stdout


def s3_key_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists.

    We use this to avoid overwriting the global merged file unless an incremental append is
    needed.
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def read_s3_csv(bucket: str, key: str) -> pd.DataFrame:
    """Load a CSV from S3 into a pandas DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(body))


def write_s3_csv(bucket: str, key: str, df: pd.DataFrame) -> None:
    """Write a pandas DataFrame as CSV to S3."""
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"))


def list_day_csv_keys(day: str, bucket: str, base_prefix: str) -> list[str]:
    """List all per-trajectory CSV keys for a given day folder (YYYY-MM-DD).

    Excludes any *_ALL.csv files to avoid re-ingesting daily aggregates.
    """
    prefix = f"{base_prefix}{day}/"
    csv_keys = []

    # Pagination is required because S3 listings can be truncated for large folders
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            if key.endswith("_ALL.csv"):
                continue
            csv_keys.append(key)

    return sorted(csv_keys)
