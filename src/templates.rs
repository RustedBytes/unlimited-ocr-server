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
    .links {
      margin: 0 0 22px;
    }
    .links a {
      color: #2563eb;
      font-weight: 650;
    }
  </style>
</head>
<body>
  <main>
    <h1>Unlimited-OCR Inference</h1>
    <p>Upload an image or PDF and queue OCR inference on the local ONNX worker pool.</p>
    <p class="links"><a href="/jobs">View jobs</a></p>

    <form method="post" action="/infer-form" enctype="multipart/form-data">
      <div>
        <label for="image">Input File</label>
        <input id="image" name="image" type="file" accept="image/*,application/pdf" required>
      </div>

      <div>
        <label for="text_input">Prompt (optional)</label>
        <textarea id="text_input" name="text_input" placeholder="<|grounding|><image>Convert the document to markdown."></textarea>
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

#[derive(Debug, Clone)]
pub struct JobsIndexRowView {
    pub id: String,
    pub status: String,
    pub input: String,
    pub kind: String,
    pub page: String,
    pub has_page: bool,
    pub updated_at: String,
    pub created_at: String,
    pub html_url: String,
    pub json_url: String,
    pub has_error: bool,
}

#[derive(Template)]
#[template(
    source = r###"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Jobs</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #1f2933;
    }
    body {
      margin: 0;
      padding: 28px;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 20px;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid #d9dee7;
      padding-bottom: 16px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    p {
      margin: 0;
      color: #52606d;
    }
    a {
      color: #2563eb;
    }
    .panel {
      border: 1px solid #d9dee7;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }
    .table-wrap {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th,
    td {
      border-bottom: 1px solid #e5e9f0;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      background: #f8fafc;
      color: #323f4b;
      font-size: 12px;
      text-transform: uppercase;
    }
    td.id,
    td.input {
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .status {
      font-weight: 700;
    }
    .error {
      color: #8a1f1f;
    }
    .pager {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px;
      color: #52606d;
    }
    .pager-links {
      display: flex;
      gap: 12px;
    }
    .empty {
      padding: 18px;
      color: #52606d;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>OCR Jobs</h1>
        <p>{{ total_jobs }} retained jobs &middot; page {{ page }} of {{ total_pages }}</p>
      </div>
      <a href="/">Submit</a>
    </header>

    <section class="panel">
      {% if has_jobs %}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Job</th>
              <th>Status</th>
              <th>Input</th>
              <th>Kind</th>
              <th>Page</th>
              <th>Updated</th>
              <th>Created</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {% for row in rows %}
            <tr>
              <td class="id"><a href="{{ row.html_url }}">{{ row.id }}</a></td>
              <td class="status{% if row.has_error %} error{% endif %}">{{ row.status }}</td>
              <td class="input">{{ row.input }}</td>
              <td>{{ row.kind }}</td>
              <td>{% if row.has_page %}{{ row.page }}{% endif %}</td>
              <td>{{ row.updated_at }}</td>
              <td>{{ row.created_at }}</td>
              <td><a href="{{ row.html_url }}">HTML</a> &middot; <a href="{{ row.json_url }}">JSON</a></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% else %}
      <div class="empty">No retained jobs yet.</div>
      {% endif %}
      <div class="pager">
        <span>Showing {{ start_item }}-{{ end_item }} of {{ total_jobs }}</span>
        <span class="pager-links">
          {% if has_prev %}<a href="{{ prev_url }}">Previous</a>{% endif %}
          {% if has_next %}<a href="{{ next_url }}">Next</a>{% endif %}
        </span>
      </div>
    </section>
  </main>
</body>
</html>
"###,
    ext = "html"
)]
pub struct JobsIndexTemplate {
    pub rows: Vec<JobsIndexRowView>,
    pub has_jobs: bool,
    pub total_jobs: usize,
    pub page: usize,
    pub total_pages: usize,
    pub start_item: usize,
    pub end_item: usize,
    pub has_prev: bool,
    pub has_next: bool,
    pub prev_url: String,
    pub next_url: String,
}

#[derive(Debug, Clone)]
pub struct JobHtmlDetectionView {
    pub label: String,
    pub bbox: String,
    pub text: String,
    pub tables: Vec<JobHtmlTableView>,
    pub has_tables: bool,
}

#[derive(Debug, Clone)]
pub struct JobHtmlTableView {
    pub rows: Vec<Vec<JobHtmlTableCellView>>,
}

#[derive(Debug, Clone)]
pub struct JobHtmlTableCellView {
    pub text: String,
    pub row_span: usize,
    pub col_span: usize,
    pub has_row_span: bool,
    pub has_col_span: bool,
}

#[derive(Template)]
#[template(
    source = r###"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Job {{ id }}</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #1f2933;
    }
    body {
      margin: 0;
      padding: 28px;
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 24px;
      border-bottom: 1px solid #d9dee7;
      padding-bottom: 18px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 22px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    h2 {
      margin: 28px 0 12px;
      font-size: 17px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px 18px;
      color: #52606d;
      font-size: 14px;
    }
    .meta strong {
      display: block;
      color: #323f4b;
      font-size: 12px;
      text-transform: uppercase;
    }
    .notice,
    .error {
      border: 1px solid #d9dee7;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px 16px;
      color: #52606d;
    }
    .error {
      border-color: #f3b5b5;
      background: #fff1f1;
      color: #8a1f1f;
    }
    .detection {
      margin: 0 0 16px;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }
    .detection-header {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: baseline;
      padding: 10px 12px;
      border-bottom: 1px solid #e5e9f0;
      background: #f9fafb;
    }
    .label {
      font-weight: 750;
      color: #102a43;
    }
    .bbox {
      color: #627d98;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .text {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
    }
    .table-wrap {
      overflow-x: auto;
      padding: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: #ffffff;
    }
    td {
      border: 1px solid #cbd2d9;
      padding: 7px 9px;
      vertical-align: top;
      min-width: 70px;
    }
    tr:first-child td {
      font-weight: 700;
      background: #f3f6fa;
    }
    .raw {
      border: 1px solid #d9dee7;
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
    }
    a {
      color: #2563eb;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>OCR Job {{ id }}</h1>
      <div class="meta">
        <div><strong>Status</strong>{{ status }}</div>
        <div><strong>Updated</strong>{{ updated_at }}</div>
        <div><strong>Input</strong>{{ filename }}</div>
        <div><strong>JSON</strong><a href="/v1/jobs/{{ id }}">/v1/jobs/{{ id }}</a></div>
      </div>
    </header>

    {% if error != "" %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if has_result %}
    <section>
      <div class="meta">
        <div><strong>Prompt</strong>{{ prompt_text }}</div>
        <div><strong>Tokens</strong>{{ generated_tokens }}</div>
        <div><strong>Elapsed</strong>{{ elapsed_ms }} ms</div>
      </div>
    </section>

    {% if has_detections %}
    <section>
      <h2>Recognized Blocks</h2>
      {% for detection in detections %}
      <article class="detection">
        <div class="detection-header">
          <span class="label">{{ detection.label }}</span>
          <span class="bbox">{{ detection.bbox }}</span>
        </div>
        {% if detection.has_tables %}
          {% for table in detection.tables %}
          <div class="table-wrap">
            <table>
              <tbody>
                {% for row in table.rows %}
                <tr>
                  {% for cell in row %}
                  <td{% if cell.has_row_span %} rowspan="{{ cell.row_span }}"{% endif %}{% if cell.has_col_span %} colspan="{{ cell.col_span }}"{% endif %}>{{ cell.text }}</td>
                  {% endfor %}
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          {% endfor %}
        {% else %}
        <pre class="text">{{ detection.text }}</pre>
        {% endif %}
      </article>
      {% endfor %}
    </section>
    {% else %}
    <section>
      <h2>Recognized Text</h2>
      <pre class="raw">{{ raw_text }}</pre>
    </section>
    {% endif %}
    {% else %}
    <div class="notice">No OCR result is available for this job yet.</div>
    {% endif %}
  </main>
</body>
</html>
"###,
    ext = "html"
)]
pub struct JobHtmlTemplate {
    pub id: String,
    pub status: String,
    pub updated_at: String,
    pub filename: String,
    pub has_result: bool,
    pub error: String,
    pub prompt_text: String,
    pub generated_tokens: usize,
    pub elapsed_ms: u128,
    pub detections: Vec<JobHtmlDetectionView>,
    pub has_detections: bool,
    pub raw_text: String,
}
