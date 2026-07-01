use crate::types::{BoundingBox, OcrDetection, OcrResult, OcrTable, OcrTableCell};

const DET_START: &str = "<|det|>";
const DET_END: &str = "<|/det|>";

pub(super) fn parse_ocr_result(generated_text: &str) -> OcrResult {
    let detections = parse_detections(generated_text);
    let tables = parse_tables(&detections);

    OcrResult {
        text: generated_text.to_string(),
        detections,
        tables,
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

fn parse_tables(detections: &[OcrDetection]) -> Vec<OcrTable> {
    detections
        .iter()
        .filter(|detection| detection.label.eq_ignore_ascii_case("table"))
        .filter_map(parse_table_detection)
        .collect()
}

fn parse_table_detection(detection: &OcrDetection) -> Option<OcrTable> {
    let rows = parse_table_rows(&detection.text);
    if rows.is_empty() {
        return None;
    }

    Some(OcrTable {
        bbox: detection.bbox,
        html: detection.text.clone(),
        rows,
    })
}

fn parse_table_rows(html: &str) -> Vec<Vec<OcrTableCell>> {
    let Some((_, table_content_start, _)) = find_open_tag(html, "table", 0) else {
        return Vec::new();
    };
    let Some(table_content_end) = find_close_tag(html, "table", table_content_start) else {
        return Vec::new();
    };

    let mut rows = Vec::new();
    let mut cursor = table_content_start;
    while cursor < table_content_end {
        let Some((row_start, row_content_start, _)) = find_open_tag(html, "tr", cursor) else {
            break;
        };
        if row_start >= table_content_end {
            break;
        }

        let Some(row_content_end) = find_close_tag(html, "tr", row_content_start) else {
            break;
        };
        if row_content_end > table_content_end {
            break;
        }

        let cells = parse_table_cells(&html[row_content_start..row_content_end]);
        if !cells.is_empty() {
            rows.push(cells);
        }
        cursor = row_content_end + "</tr>".len();
    }

    rows
}

fn parse_table_cells(row_html: &str) -> Vec<OcrTableCell> {
    let mut cells = Vec::new();
    let mut cursor = 0;

    while cursor < row_html.len() {
        let Some((tag, cell_content_start, attrs)) = find_next_cell_tag(row_html, cursor) else {
            break;
        };
        let Some(cell_content_end) = find_close_tag(row_html, tag, cell_content_start) else {
            break;
        };

        cells.push(OcrTableCell {
            text: table_cell_text(&row_html[cell_content_start..cell_content_end]),
            row_span: parse_span_attr(attrs, "rowspan").unwrap_or(1),
            col_span: parse_span_attr(attrs, "colspan").unwrap_or(1),
        });
        cursor = cell_content_end + tag_close_len(tag);
    }

    cells
}

fn find_next_cell_tag(html: &str, cursor: usize) -> Option<(&'static str, usize, &str)> {
    let td = find_open_tag(html, "td", cursor);
    let th = find_open_tag(html, "th", cursor);

    match (td, th) {
        (
            Some((td_start, td_content_start, td_attrs)),
            Some((th_start, th_content_start, th_attrs)),
        ) => {
            if td_start <= th_start {
                Some(("td", td_content_start, td_attrs))
            } else {
                Some(("th", th_content_start, th_attrs))
            }
        }
        (Some((_, content_start, attrs)), None) => Some(("td", content_start, attrs)),
        (None, Some((_, content_start, attrs))) => Some(("th", content_start, attrs)),
        (None, None) => None,
    }
}

fn find_open_tag<'a>(html: &'a str, tag: &str, cursor: usize) -> Option<(usize, usize, &'a str)> {
    let lower = html.to_ascii_lowercase();
    let pattern = format!("<{tag}");
    let mut search_from = cursor;

    while search_from < html.len() {
        let relative_start = lower[search_from..].find(&pattern)?;
        let start = search_from + relative_start;
        let boundary_idx = start + pattern.len();
        let boundary = lower.as_bytes().get(boundary_idx).copied();
        let has_tag_boundary = matches!(boundary, Some(b'>') | Some(b'/'))
            || boundary.is_some_and(|byte| byte.is_ascii_whitespace());
        if !has_tag_boundary {
            search_from = boundary_idx;
            continue;
        }

        let tag_end = html[boundary_idx..].find('>')? + boundary_idx;
        return Some((start, tag_end + 1, html[boundary_idx..tag_end].trim()));
    }

    None
}

fn find_close_tag(html: &str, tag: &str, cursor: usize) -> Option<usize> {
    let lower = html.to_ascii_lowercase();
    lower[cursor..]
        .find(&format!("</{tag}>"))
        .map(|idx| cursor + idx)
}

fn tag_close_len(tag: &str) -> usize {
    tag.len() + "</>".len()
}

fn parse_span_attr(attrs: &str, name: &str) -> Option<usize> {
    let lower = attrs.to_ascii_lowercase();
    let start = lower.find(name)?;
    let mut rest = attrs[start + name.len()..].trim_start();
    rest = rest.strip_prefix('=')?.trim_start();

    let (raw_value, _) = if let Some(value) = rest.strip_prefix('"') {
        value.split_once('"')?
    } else if let Some(value) = rest.strip_prefix('\'') {
        value.split_once('\'')?
    } else {
        let end = rest
            .find(|ch: char| ch.is_ascii_whitespace() || ch == '>')
            .unwrap_or(rest.len());
        rest.split_at(end)
    };

    raw_value.parse::<usize>().ok().filter(|value| *value > 0)
}

fn table_cell_text(html: &str) -> String {
    let without_tags = strip_html_tags(html);
    decode_html_entities(&without_tags).trim().to_string()
}

fn strip_html_tags(html: &str) -> String {
    let mut output = String::new();
    let mut remaining = html;

    while let Some(start) = remaining.find('<') {
        output.push_str(&remaining[..start]);
        let Some(end) = remaining[start..].find('>') else {
            output.push_str(&remaining[start..]);
            return output;
        };

        let tag = remaining[start + 1..start + end].trim();
        if tag
            .split_whitespace()
            .next()
            .is_some_and(|name| name.eq_ignore_ascii_case("br") || name.eq_ignore_ascii_case("br/"))
        {
            output.push('\n');
        }
        remaining = &remaining[start + end + 1..];
    }

    output.push_str(remaining);
    output
}

fn decode_html_entities(value: &str) -> String {
    let mut output = String::new();
    let mut remaining = value;

    while let Some(start) = remaining.find('&') {
        output.push_str(&remaining[..start]);
        let entity_start = start + 1;
        let Some(end) = remaining[entity_start..].find(';') else {
            output.push_str(&remaining[start..]);
            return output;
        };

        let entity = &remaining[entity_start..entity_start + end];
        if let Some(decoded) = decode_html_entity(entity) {
            output.push(decoded);
        } else {
            output.push('&');
            output.push_str(entity);
            output.push(';');
        }
        remaining = &remaining[entity_start + end + 1..];
    }

    output.push_str(remaining);
    output
}

fn decode_html_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        _ => decode_numeric_html_entity(entity),
    }
}

fn decode_numeric_html_entity(entity: &str) -> Option<char> {
    let value = if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        u32::from_str_radix(hex, 16).ok()?
    } else {
        entity.strip_prefix('#')?.parse::<u32>().ok()?
    };

    char::from_u32(value)
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
        assert!(got.tables.is_empty());
    }

    #[test]
    fn leaves_plain_text_without_detections() {
        let got = parse_ocr_result("plain OCR text");

        assert_eq!(got.text, "plain OCR text");
        assert!(got.detections.is_empty());
        assert!(got.tables.is_empty());
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

    #[test]
    fn parses_table_detections_into_structured_rows() {
        let got = parse_ocr_result(
            "<|det|>table [202, 627, 830, 740]<|/det|>\
             <table><tr><td></td><td>Attenuation (dB/25m)</td><td>Max. Temp.</td></tr>\
             <tr><td>POF</td><td>1.1</td><td>90°C</td></tr>\
             <tr><td>PCS</td><td>&lt; 0.5</td><td>125°C</td></tr></table>",
        );

        assert_eq!(got.detections.len(), 1);
        assert_eq!(got.tables.len(), 1);
        assert_eq!(
            got.tables[0].bbox,
            BoundingBox {
                x_min: 202,
                y_min: 627,
                x_max: 830,
                y_max: 740
            }
        );
        assert_eq!(got.tables[0].rows.len(), 3);
        assert_eq!(got.tables[0].rows[0][1].text, "Attenuation (dB/25m)");
        assert_eq!(got.tables[0].rows[2][0].text, "PCS");
        assert_eq!(got.tables[0].rows[2][1].text, "< 0.5");
    }

    #[test]
    fn parses_table_cell_spans_and_headers() {
        let got = parse_ocr_result(
            "<|det|>table [1, 2, 3, 4]<|/det|>\
             <table><tr><th colspan=\"2\">Group</th></tr>\
             <tr><td rowspan='2'>A</td><td>&#8805;4</td></tr></table>",
        );

        let table = &got.tables[0];
        assert_eq!(table.rows[0][0].text, "Group");
        assert_eq!(table.rows[0][0].col_span, 2);
        assert_eq!(table.rows[1][0].text, "A");
        assert_eq!(table.rows[1][0].row_span, 2);
        assert_eq!(table.rows[1][1].text, "≥4");
    }

    #[test]
    fn skips_malformed_table_markup() {
        let got = parse_ocr_result("<|det|>table [1, 2, 3, 4]<|/det|><table><tr><td>open");

        assert_eq!(got.detections.len(), 1);
        assert!(got.tables.is_empty());
    }
}
