#!/usr/bin/env python3
"""Small urllib3 client for the Unlimited-OCR inference server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

try:
    import urllib3
    from urllib3.exceptions import HTTPError
except ModuleNotFoundError:
    urllib3 = None

    class HTTPError(Exception):
        pass


DEFAULT_BASE_URL = "http://127.0.0.1:3000"
TERMINAL_STATUSES = {"succeeded", "failed"}


class ClientError(RuntimeError):
    """Raised when the server returns an error or invalid response."""


class UnlimitedOcrClient:
    def __init__(
        self,
        base_url: str,
        read_timeout: float,
        connect_timeout: float,
        api_key: str | None,
    ) -> None:
        if urllib3 is None:
            raise ClientError("urllib3 is not installed; run `python3 -m pip install urllib3`")

        self.base_url = base_url.rstrip("/") + "/"
        self.default_headers = {"x-api-key": api_key} if api_key else {}
        self.http = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=connect_timeout, read=read_timeout),
            retries=False,
        )

    def submit_upload(
        self,
        image_path: Path,
        prompt: str | None,
        webhook_url: str | None,
    ) -> dict[str, Any]:
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        fields: dict[str, Any] = {
            "image": (image_path.name, image_path.read_bytes(), content_type),
        }
        if prompt is not None:
            fields["text_input"] = prompt
        if webhook_url is not None:
            fields["webhook_url"] = webhook_url

        return self._request_json("POST", "v1/infer", fields=fields)

    def submit_local_path(
        self,
        image_path: str,
        prompt: str | None,
        webhook_url: str | None,
    ) -> dict[str, Any]:
        payload = {
            "image_path": image_path,
            "text_input": prompt,
            "webhook_url": webhook_url,
        }
        return self._request_json(
            "POST",
            "v1/infer/path",
            body=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"v1/jobs/{job_id}")

    def wait_for_job(self, job_id: str, poll_interval: float) -> dict[str, Any]:
        while True:
            job = self.get_job(job_id)
            status = job.get("status")
            if status in TERMINAL_STATUSES:
                return job
            if not isinstance(status, str):
                raise ClientError(f"job {job_id} response has invalid status: {status!r}")
            time.sleep(poll_interval)

    def wait_for_submission(
        self,
        response: dict[str, Any],
        poll_interval: float,
    ) -> dict[str, Any]:
        jobs = response.get("jobs")
        if not isinstance(jobs, list):
            job_id = response["id"]
            if not isinstance(job_id, str):
                raise ClientError(f"submission response has invalid id: {job_id!r}")
            return self.wait_for_job(job_id, poll_interval)

        completed = []
        for job in jobs:
            if not isinstance(job, dict) or not isinstance(job.get("id"), str):
                raise ClientError(f"PDF submission response has invalid job entry: {job!r}")
            completed.append(self.wait_for_job(job["id"], poll_interval))

        result = dict(response)
        result["jobs"] = completed
        return result

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        fields: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = dict(self.default_headers)
        if headers:
            request_headers.update(headers)

        response = self.http.request(
            method,
            urljoin(self.base_url, path),
            fields=fields,
            body=body,
            headers=request_headers,
        )
        raw_body = response.data.decode("utf-8", errors="replace")
        if response.status >= 400:
            raise ClientError(f"server returned HTTP {response.status}: {raw_body}")

        try:
            value = json.loads(raw_body)
        except json.JSONDecodeError as err:
            raise ClientError(f"server returned invalid JSON: {err}") from err

        if not isinstance(value, dict):
            raise ClientError(f"server returned JSON {type(value).__name__}, expected object")
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", help="optional API key sent as x-api-key")
    parser.add_argument("--prompt", "--text-input", dest="prompt")
    parser.add_argument("--webhook-url", help="optional HTTP(S) URL that receives the final job JSON")
    parser.add_argument("--no-wait", action="store_true", help="print the queued job only")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="submit an image file as multipart data")
    upload_parser.add_argument("image_path", type=Path)

    path_parser = subparsers.add_parser("path", help="submit an image path visible to the server")
    path_parser.add_argument("image_path")

    job_parser = subparsers.add_parser("job", help="fetch an existing job by id")
    job_parser.add_argument("job_id")

    args = parser.parse_args()
    client = UnlimitedOcrClient(args.base_url, args.read_timeout, args.connect_timeout, args.api_key)

    try:
        if args.command == "upload":
            response = client.submit_upload(
                args.image_path,
                args.prompt,
                args.webhook_url,
            )
        elif args.command == "path":
            response = client.submit_local_path(
                args.image_path,
                args.prompt,
                args.webhook_url,
            )
        else:
            response = client.get_job(args.job_id)

        if not args.no_wait and args.command in {"upload", "path"}:
            response = client.wait_for_submission(response, args.poll_interval)

        print(json.dumps(response, indent=2, sort_keys=True))
        jobs = response.get("jobs")
        if isinstance(jobs, list):
            return 1 if any(job.get("status") == "failed" for job in jobs if isinstance(job, dict)) else 0
        return 1 if response.get("status") == "failed" else 0
    except (OSError, KeyError, HTTPError, ClientError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
