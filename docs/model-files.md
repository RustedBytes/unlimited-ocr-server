# Model Files

By default, the server expects the Unlimited-OCR repository contents under `Unlimited-OCR/`, including:

```text
Unlimited-OCR/onnx/unlimited_ocr.onnx
Unlimited-OCR/tokenizer.json
```

Startup validates the selected ONNX graph and tokenizer before the server accepts traffic. If `model.path` or `MODEL_PATH` points to a custom ONNX file, place `tokenizer.json` either beside the ONNX directory or in its parent model directory.
