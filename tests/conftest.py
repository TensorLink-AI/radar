"""Shared test fixtures."""

import io
import json
import pytest


class MockS3:
    """In-memory S3 client for testing."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes):
        self._store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket: str, Key: str):
        if Key not in self._store:
            raise Exception(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self._store[Key])}

    def head_object(self, Bucket: str, Key: str):
        if Key not in self._store:
            raise Exception(f"NoSuchKey: {Key}")
        return {"ContentLength": len(self._store[Key])}

    def upload_file(self, local_path: str, bucket: str, key: str):
        with open(local_path, "rb") as f:
            self._store[key] = f.read()

    def list_objects_v2(self, Bucket: str, Prefix: str = ""):
        contents = []
        for key in sorted(self._store):
            if key.startswith(Prefix):
                contents.append({"Key": key})
        return {"Contents": contents} if contents else {}


class MockR2:
    """In-memory R2AuditLog for testing (no boto3 required)."""

    def __init__(self):
        self.bucket = "test-bucket"
        self._s3 = MockS3()

    def upload_json(self, key: str, data: dict) -> bool:
        body = json.dumps(data, indent=2).encode()
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=body)
        return True

    def download_json(self, key: str):
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except Exception:
            return None

    def upload_file_from_disk(self, local_path: str, key: str) -> bool:
        self._s3.upload_file(local_path, self.bucket, key)
        return True

    def download_file_to_disk(self, key: str, local_path: str) -> bool:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            with open(local_path, "wb") as f:
                f.write(resp["Body"].read())
            return True
        except Exception:
            return False

    def upload_text(self, key: str, text: str) -> bool:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode())
        return True

    def download_text(self, key: str):
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode()
        except Exception:
            return None

    def key_exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception:
            return []


@pytest.fixture
def mock_r2():
    """Provide a MockR2 instance for tests."""
    return MockR2()
