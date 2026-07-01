use std::{
    io,
    path::{Path, PathBuf},
};

use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct RenderedPdfPage {
    pub page_number: usize,
    pub image_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct PdfRenderOptions {
    pub max_pages: usize,
    pub dpi: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum PdfRenderError {
    #[error(
        "PDF renderer is unavailable; install poppler-utils so `pdfinfo` and `pdftoppm` are on PATH"
    )]
    RendererUnavailable,
    #[error("failed to inspect PDF: {0}")]
    Inspect(String),
    #[error("PDF has {pages} pages, which exceeds configured limit {max_pages}")]
    TooManyPages { pages: usize, max_pages: usize },
    #[error("failed to render PDF: {0}")]
    Render(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub fn is_pdf_content(content_type: Option<&str>, bytes: &[u8]) -> bool {
    matches!(content_type, Some("application/pdf")) || bytes.starts_with(b"%PDF-")
}

pub async fn render_pdf_pages(
    pdf_path: &Path,
    output_dir: &Path,
    output_stem: &str,
    options: PdfRenderOptions,
) -> Result<Vec<RenderedPdfPage>, PdfRenderError> {
    let page_count = inspect_page_count(pdf_path).await?;
    if page_count == 0 {
        return Err(PdfRenderError::Inspect(
            "PDF does not contain any pages".to_string(),
        ));
    }
    if page_count > options.max_pages {
        return Err(PdfRenderError::TooManyPages {
            pages: page_count,
            max_pages: options.max_pages,
        });
    }

    let output_prefix = output_dir.join(output_stem);
    let output = Command::new("pdftoppm")
        .arg("-png")
        .arg("-r")
        .arg(options.dpi.to_string())
        .arg("-f")
        .arg("1")
        .arg("-l")
        .arg(page_count.to_string())
        .arg(pdf_path)
        .arg(&output_prefix)
        .output()
        .await
        .map_err(command_error)?;
    if !output.status.success() {
        return Err(PdfRenderError::Render(command_stderr(&output.stderr)));
    }

    collect_rendered_pages(output_dir, output_stem, page_count).await
}

async fn inspect_page_count(pdf_path: &Path) -> Result<usize, PdfRenderError> {
    let output = Command::new("pdfinfo")
        .arg(pdf_path)
        .output()
        .await
        .map_err(command_error)?;
    if !output.status.success() {
        return Err(PdfRenderError::Inspect(command_stderr(&output.stderr)));
    }

    parse_pdfinfo_pages(&String::from_utf8_lossy(&output.stdout)).ok_or_else(|| {
        PdfRenderError::Inspect("pdfinfo output did not include a page count".to_string())
    })
}

fn command_error(err: io::Error) -> PdfRenderError {
    if err.kind() == io::ErrorKind::NotFound {
        PdfRenderError::RendererUnavailable
    } else {
        PdfRenderError::Io(err)
    }
}

fn command_stderr(stderr: &[u8]) -> String {
    let message = String::from_utf8_lossy(stderr).trim().to_string();
    if message.is_empty() {
        "renderer exited without an error message".to_string()
    } else {
        message
    }
}

fn parse_pdfinfo_pages(output: &str) -> Option<usize> {
    output.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case("Pages")
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

async fn collect_rendered_pages(
    output_dir: &Path,
    output_stem: &str,
    page_count: usize,
) -> Result<Vec<RenderedPdfPage>, PdfRenderError> {
    let mut pages = Vec::with_capacity(page_count);
    let mut entries = tokio::fs::read_dir(output_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(page_number) = rendered_page_number(filename, output_stem) else {
            continue;
        };
        pages.push(RenderedPdfPage {
            page_number,
            image_path: path,
        });
    }

    pages.sort_by_key(|page| page.page_number);
    if pages.len() != page_count {
        return Err(PdfRenderError::Render(format!(
            "expected {page_count} rendered pages, found {}",
            pages.len()
        )));
    }
    Ok(pages)
}

fn rendered_page_number(filename: &str, output_stem: &str) -> Option<usize> {
    filename
        .strip_prefix(output_stem)?
        .strip_prefix('-')?
        .strip_suffix(".png")?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pdf_by_content_type_or_magic() {
        assert!(is_pdf_content(Some("application/pdf"), b"not a pdf"));
        assert!(is_pdf_content(None, b"%PDF-1.7\n"));
        assert!(is_pdf_content(Some("image/png"), b"%PDF-1.7\n"));
    }

    #[test]
    fn parses_pdfinfo_page_count() {
        let output = "Title: example\nPages:          12\nEncrypted: no\n";

        assert_eq!(parse_pdfinfo_pages(output), Some(12));
    }

    #[test]
    fn extracts_rendered_page_number() {
        assert_eq!(rendered_page_number("batch-9.png", "batch"), Some(9));
        assert_eq!(rendered_page_number("other-9.png", "batch"), None);
        assert_eq!(rendered_page_number("batch-preview.png", "batch"), None);
    }
}
