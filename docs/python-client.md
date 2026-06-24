# Python Client

A simple `urllib3` client is available at `examples/python_client.py`.

Install the Python dependency:

```bash
python3 -m pip install urllib3
```

Submit an image upload and wait for the final result:

```bash
python3 examples/python_client.py upload /path/to/image.png
```

If API key authentication is enabled:

```bash
python3 examples/python_client.py \
  --api-key replace-with-a-long-random-secret \
  upload /path/to/image.png
```

Submit a server-side image path:

```bash
python3 examples/python_client.py path /path/to/image.png
```

Use another OCR prompt:

```bash
python3 examples/python_client.py \
  --prompt '<image>read this receipt' \
  upload /path/to/image.png
```

Send the final job result to a webhook:

```bash
python3 examples/python_client.py \
  --webhook-url https://example.com/unlimited-ocr-webhook \
  upload /path/to/image.png
```

Fetch an existing job:

```bash
python3 examples/python_client.py job <job-id>
```
