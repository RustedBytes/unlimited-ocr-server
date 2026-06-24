use std::fmt;

#[derive(Debug, Clone)]
pub struct TaskSpec {
    pub task_type: TaskType,
    pub task_prompt: TaskPrompt,
    pub text_input: Option<String>,
}

impl Default for TaskSpec {
    fn default() -> Self {
        Self {
            task_type: TaskType::Single,
            task_prompt: TaskPrompt::Ocr,
            text_input: None,
        }
    }
}

impl TaskSpec {
    pub fn from_strings(
        _task_type: Option<String>,
        _task_prompt: Option<String>,
        text_input: Option<String>,
    ) -> Result<Self, TaskSpecError> {
        Ok(Self {
            text_input: text_input.and_then(normalize_optional_text),
            ..Self::default()
        })
    }

    pub fn task_type_name(&self) -> &'static str {
        self.task_type.as_str()
    }

    pub fn task_prompt_name(&self) -> &'static str {
        self.task_prompt.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Single,
}

impl TaskType {
    pub fn as_str(self) -> &'static str {
        "Single task"
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPrompt {
    Ocr,
}

impl TaskPrompt {
    pub fn as_str(self) -> &'static str {
        "OCR"
    }
}

impl fmt::Display for TaskPrompt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid task specification")]
pub struct TaskSpecError;

fn normalize_optional_text(text: String) -> Option<String> {
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
}
