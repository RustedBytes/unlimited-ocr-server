use crate::types::{BoundingBox, OcrDetection, OcrResult};

const DET_START: &str = "<|det|>";
const DET_END: &str = "<|/det|>";

pub(super) fn parse_ocr_result(generated_text: &str) -> OcrResult {
    OcrResult {
        text: generated_text.to_string(),
        detections: parse_detections(generated_text),
    }
}

fn parse_detections(generated_text: &str) -> Vec<OcrDetection> {
    let mut remaining = generated_text;
    let mut detections = Vec::new();

    while let Some(start) = remaining.find(DET_START) {
        remaining = &remaining[start + DET_START.len()..];
        let Some(end) = remaining.find(DET_END) else {
            break;
        };

        let header = &remaining[..end];
        remaining = &remaining[end + DET_END.len()..];

        let Some((label, bbox)) = parse_detection_header(header) else {
            continue;
        };

        let next_start = remaining.find(DET_START).unwrap_or(remaining.len());
        let text = remaining[..next_start].trim().to_string();
        detections.push(OcrDetection { label, bbox, text });
        remaining = &remaining[next_start..];
    }

    detections
}

fn parse_detection_header(header: &str) -> Option<(String, BoundingBox)> {
    let header = header.trim();
    let bbox_start = header.rfind('[')?;
    let bbox_end = header[bbox_start..].find(']')? + bbox_start;
    let label = header[..bbox_start].trim();
    if label.is_empty() {
        return None;
    }

    let coords = header[bbox_start + 1..bbox_end]
        .split(',')
        .map(|value| value.trim().parse::<i64>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    let [x_min, y_min, x_max, y_max]: [i64; 4] = coords.try_into().ok()?;
    Some((
        label.to_string(),
        BoundingBox {
            x_min,
            y_min,
            x_max,
            y_max,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_detection_tags_into_structured_boxes() {
        let got = parse_ocr_result(
            "<|det|>title [443, 67, 554, 87]<|/det|>ВИМОГИ\n\
             <|det|>text [249, 98, 740, 118]<|/det|>до оформления статей",
        );

        assert_eq!(
            got.text,
            "<|det|>title [443, 67, 554, 87]<|/det|>ВИМОГИ\n<|det|>text [249, 98, 740, 118]<|/det|>до оформления статей"
        );
        assert_eq!(got.detections.len(), 2);
        assert_eq!(got.detections[0].label, "title");
        assert_eq!(
            got.detections[0].bbox,
            BoundingBox {
                x_min: 443,
                y_min: 67,
                x_max: 554,
                y_max: 87
            }
        );
        assert_eq!(got.detections[0].text, "ВИМОГИ");
        assert_eq!(got.detections[1].label, "text");
        assert_eq!(got.detections[1].text, "до оформления статей");
    }

    #[test]
    fn leaves_plain_text_without_detections() {
        let got = parse_ocr_result("plain OCR text");

        assert_eq!(got.text, "plain OCR text");
        assert!(got.detections.is_empty());
    }

    #[test]
    fn skips_malformed_detection_headers() {
        let got = parse_ocr_result(
            "<|det|>text [1, 2, 3]<|/det|>bad\n\
             <|det|>text [4, 5, 6, 7]<|/det|>good",
        );

        assert_eq!(got.detections.len(), 1);
        assert_eq!(
            got.detections[0].bbox,
            BoundingBox {
                x_min: 4,
                y_min: 5,
                x_max: 6,
                y_max: 7
            }
        );
        assert_eq!(got.detections[0].text, "good");
    }
}
