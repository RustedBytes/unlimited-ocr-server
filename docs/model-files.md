# Model Files

By default, the server expects the Unlimited-OCR repository contents under `Unlimited-OCR/`.

Repository: https://huggingface.co/Yehor/Unlimited-OCR-KV-Cache-ONNX

Startup validates the selected ONNX graph and tokenizer before the server accepts traffic. If `model.path` or `MODEL_PATH` points to a custom ONNX file, place `tokenizer.json` either beside the ONNX directory or in its parent model directory.

Example run:

```bash
MODEL_PATH=Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx DECODE_MODEL_PATH=Unlimited-OCR/onnx/unlimited_ocr_decode.onnx bash run_podman.sh
```
