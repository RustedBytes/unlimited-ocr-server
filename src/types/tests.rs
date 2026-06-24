use super::*;

#[test]
fn task_spec_defaults_to_ocr() {
    let task = TaskSpec::from_strings(None, None, None).unwrap();

    assert_eq!(task.task_type, TaskType::Single);
    assert_eq!(task.task_prompt, TaskPrompt::Ocr);
    assert_eq!(task.text_input, None);
}

#[test]
fn task_spec_keeps_text_input_as_prompt_override() {
    let task = TaskSpec::from_strings(
        Some("legacy task type".to_string()),
        Some("legacy task prompt".to_string()),
        Some("  <image>read this receipt  ".to_string()),
    )
    .unwrap();

    assert_eq!(task.task_type_name(), "Single task");
    assert_eq!(task.task_prompt_name(), "OCR");
    assert_eq!(task.text_input.as_deref(), Some("<image>read this receipt"));
}

#[test]
fn task_spec_ignores_empty_text_input() {
    let task = TaskSpec::from_strings(None, None, Some("   ".to_string())).unwrap();

    assert_eq!(task.text_input, None);
}
