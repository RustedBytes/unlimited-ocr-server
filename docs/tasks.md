# Prompt Compatibility

Unlimited-OCR inference uses one prompt string.

- `text_input`: optional prompt override. Empty or missing values use `<image>Free OCR.`.
- `task_type`, `task_prompt`, and `task`: accepted for backward-compatible clients, but ignored by inference.

If the prompt does not contain `<image>`, the server prefixes it before tokenization.
