"""R2-compatible storage for experiment artifacts.

Stores experiment data (code, checkpoints, proposals, dispatch records) in
S3-compatible storage (Cloudflare R2) for cross-validator sharing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2AuditLog:
    """S3-compatible experiment storage (Cloudflare R2)."""

    def __init__(
        self,
        account_id: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
        bucket: str = "",
    ):
        self.bucket = bucket or os.getenv("R2_BUCKET", "radar-experiments")
        account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
        access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID", "")
        secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY", "")

        import boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com" if account_id else None,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def upload_json(self, key: str, data: dict) -> bool:
        """Upload a JSON object to R2."""
        try:
            body = json.dumps(data, indent=2).encode()
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=body)
            return True
        except Exception as e:
            logger.error("Failed to upload JSON %s: %s", key, e)
            return False

    def download_json(self, key: str) -> Optional[dict]:
        """Download and parse a JSON object from R2."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug("Key not found in R2: %s", key)
            else:
                logger.error("Failed to download JSON %s: %s", key, e)
            return None
        except Exception as e:
            logger.error("Failed to download JSON %s: %s", key, e)
            return None

    def list_experiments(self, prefix: str = "") -> list[str]:
        """List all experiment keys matching prefix."""
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            keys = [obj["Key"] for obj in resp.get("Contents", [])]
            # Extract experiment IDs from bundle paths
            experiments = set()
            for key in keys:
                parts = key.split("/")
                if len(parts) >= 2:
                    experiments.add(parts[0])
            return sorted(experiments)
        except Exception as e:
            logger.error("Failed to list experiments with prefix %s: %s", prefix, e)
            return []

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys matching prefix."""
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception as e:
            logger.error("Failed to list keys with prefix %s: %s", prefix, e)
            return []

    def upload_file_from_disk(self, local_path: str, key: str) -> bool:
        """Upload a file from local disk to R2."""
        try:
            self._s3.upload_file(local_path, self.bucket, key)
            logger.info("Uploaded file: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to upload file %s: %s", key, e)
            return False

    def download_file_to_disk(self, key: str, local_path: str) -> bool:
        """Download a file from R2 to local disk."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            with open(local_path, "wb") as f:
                f.write(resp["Body"].read())
            return True
        except Exception as e:
            logger.error("Failed to download file %s: %s", key, e)
            return False

    def upload_text(self, key: str, text: str) -> bool:
        """Upload a text string to R2."""
        try:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode())
            return True
        except Exception as e:
            logger.error("Failed to upload text %s: %s", key, e)
            return False

    def download_text(self, key: str) -> Optional[str]:
        """Download a text string from R2."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode()
        except Exception as e:
            logger.error("Failed to download text %s: %s", key, e)
            return None

    def generate_presigned_put_url(self, key: str, ttl: int = 5400) -> str:
        """Generate a pre-signed PUT URL for uploading to a specific key.

        Args:
            key: The S3 key to generate the URL for.
            ttl: Time-to-live in seconds (default 1 hour).

        Returns:
            Pre-signed URL string, or empty string on failure.
        """
        try:
            url = self._s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=ttl,
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned PUT URL for %s: %s", key, e)
            return ""

    def key_exists(self, key: str) -> bool:
        """Check if a key exists in the bucket."""
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
