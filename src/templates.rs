use askama::Template;

#[derive(Template)]
#[template(
    source = r###"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unlimited-OCR Inference</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 760px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 28px;
      box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    p {
      margin: 0 0 24px;
      color: #52606d;
    }
    form {
      display: grid;
      gap: 18px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 650;
      color: #323f4b;
    }
    input[type="file"],
    input[type="text"],
    input[type="url"],
    textarea {
      box-sizing: border-box;
      width: 100%;
      border: 1px solid #cbd2d9;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: #ffffff;
    }
    textarea {
      min-height: 92px;
      resize: vertical;
    }
    button {
      justify-self: start;
      border: 0;
      border-radius: 6px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: #2563eb;
      cursor: pointer;
    }
    button:hover {
      background: #1d4ed8;
    }
    .notice {
      margin-top: 22px;
      padding: 14px 16px;
      border-radius: 6px;
      border: 1px solid #b7d7c0;
      background: #edf8f0;
      color: #1f5130;
    }
    .error {
      margin-top: 22px;
      padding: 14px 16px;
      border-radius: 6px;
      border: 1px solid #f3b5b5;
      background: #fff1f1;
      color: #8a1f1f;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
    }
  </style>
</head>
<body>
  <main>
    <h1>Unlimited-OCR Inference</h1>
    <p>Upload an image or PDF and queue OCR inference on the local ONNX worker pool.</p>

    <form method="post" action="/infer-form" enctype="multipart/form-data">
      <div>
        <label for="image">Input File</label>
        <input id="image" name="image" type="file" accept="image/*,application/pdf" required>
      </div>

      <div>
        <label for="text_input">Prompt (optional)</label>
        <textarea id="text_input" name="text_input" placeholder="<image>Free OCR."></textarea>
      </div>

      <div>
        <label for="webhook_url">Webhook URL (optional)</label>
        <input id="webhook_url" name="webhook_url" type="url" placeholder="https://example.com/unlimited-ocr-webhook">
      </div>

      <button type="submit">Submit</button>
    </form>

    {% if queued %}
    <div class="notice">
      {{ queued_message }}<br>
      {% if has_status_url %}
      Status: <a href="{{ status_url }}">{{ status_url }}</a>
      {% endif %}
    </div>
    {% endif %}

    {% if error != "" %}
    <div class="error">{{ error }}</div>
    {% endif %}
  </main>
</body>
</html>
"###,
    ext = "html"
)]
pub struct IndexTemplate {
    pub queued: bool,
    pub queued_message: String,
    pub status_url: String,
    pub has_status_url: bool,
    pub error: String,
}
