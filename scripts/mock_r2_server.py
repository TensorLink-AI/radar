"""Minimal S3-compatible server for localnet testing.

Accepts PUT/GET/HEAD/LIST on any key, stores files on the local filesystem.
Generates "presigned" URLs that are just http://localhost:PORT/BUCKET/KEY.

Usage:
    python scripts/mock_r2_server.py              # port 9000
    python scripts/mock_r2_server.py --port 9001  # custom port

Configure the validator/trainer to use this by setting:
    export R2_ACCOUNT_ID=local
    export R2_ACCESS_KEY_ID=test
    export R2_SECRET_ACCESS_KEY=test
    export R2_BUCKET=radar-test

Then override the R2AuditLog endpoint to point here instead of Cloudflare.
The test scripts handle this via MOCK_R2_ENDPOINT env var.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Mock R2/S3 Server")

# Storage root — all objects stored as files under this dir
STORAGE_DIR = Path(os.environ.get("MOCK_R2_STORAGE", "/tmp/mock_r2_storage"))


def _object_path(bucket: str, key: str) -> Path:
    """Map bucket/key to a filesystem path."""
    # Sanitize to prevent path traversal
    safe_key = key.replace("..", "").lstrip("/")
    return STORAGE_DIR / bucket / safe_key


# ── S3-compatible PUT (upload) ────────────────────────────────────

@app.put("/{bucket}/{key:path}")
async def put_object(bucket: str, key: str, request: Request):
    """Store an object (handles both direct and presigned PUT)."""
    body = await request.body()
    path = _object_path(bucket, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    etag = hashlib.md5(body).hexdigest()
    return Response(
        status_code=200,
        headers={"ETag": f'"{etag}"'},
    )


# ── S3-compatible GET (download) ──────────────────────────────────

@app.get("/{bucket}/{key:path}")
async def get_object(bucket: str, key: str):
    """Retrieve an object."""
    path = _object_path(bucket, key)
    if not path.exists():
        return _s3_error("NoSuchKey", f"The specified key does not exist.", key, 404)
    body = path.read_bytes()
    etag = hashlib.md5(body).hexdigest()
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={"ETag": f'"{etag}"', "Content-Length": str(len(body))},
    )


# ── S3-compatible HEAD ────────────────────────────────────────────

@app.head("/{bucket}/{key:path}")
async def head_object(bucket: str, key: str):
    """Check if object exists."""
    path = _object_path(bucket, key)
    if not path.exists():
        return Response(status_code=404)
    body = path.read_bytes()
    return Response(
        status_code=200,
        headers={
            "Content-Length": str(len(body)),
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
        },
    )


# ── S3-compatible LIST (ListObjectsV2) ───────────────────────────

@app.get("/{bucket}")
async def list_objects(bucket: str, request: Request):
    """S3 ListObjectsV2-compatible listing."""
    prefix = request.query_params.get("prefix", "")
    bucket_dir = STORAGE_DIR / bucket
    if not bucket_dir.exists():
        return Response(
            content=_list_xml(bucket, prefix, []),
            media_type="application/xml",
        )

    contents = []
    for path in bucket_dir.rglob("*"):
        if path.is_file():
            key = str(path.relative_to(bucket_dir))
            if key.startswith(prefix):
                body = path.read_bytes()
                contents.append({
                    "Key": key,
                    "Size": len(body),
                    "ETag": hashlib.md5(body).hexdigest(),
                })

    return Response(
        content=_list_xml(bucket, prefix, contents),
        media_type="application/xml",
    )


def _list_xml(bucket: str, prefix: str, contents: list[dict]) -> str:
    """Build an S3 ListObjectsV2 XML response."""
    items = ""
    for c in contents:
        items += f"""<Contents>
            <Key>{c['Key']}</Key>
            <Size>{c['Size']}</Size>
            <ETag>"{c['ETag']}"</ETag>
        </Contents>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult>
    <Name>{bucket}</Name>
    <Prefix>{prefix}</Prefix>
    <KeyCount>{len(contents)}</KeyCount>
    <MaxKeys>1000</MaxKeys>
    <IsTruncated>false</IsTruncated>
    {items}
</ListBucketResult>"""


def _s3_error(code: str, message: str, key: str, status: int) -> Response:
    """Return an S3-style XML error."""
    return Response(
        status_code=status,
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Error>
    <Code>{code}</Code>
    <Message>{message}</Message>
    <Key>{key}</Key>
</Error>""",
        media_type="application/xml",
    )


@app.get("/")
async def health():
    return {"status": "ok", "storage": str(STORAGE_DIR)}


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Mock R2/S3 server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--storage", type=str, default="/tmp/mock_r2_storage")
    args = parser.parse_args()

    global STORAGE_DIR
    STORAGE_DIR = Path(args.storage)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger.info("Mock R2 server starting on port %d, storage: %s", args.port, STORAGE_DIR)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
