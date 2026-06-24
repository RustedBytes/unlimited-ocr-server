#[derive(Debug, Clone, Default)]
pub struct TaskSpec {
    pub text_input: Option<String>,
}

impl TaskSpec {
    pub fn from_text_input(text_input: Option<String>) -> Self {
        Self {
            text_input: text_input.and_then(normalize_optional_text),
        }
    }
}

fn normalize_optional_text(text: String) -> Option<String> {
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
}
