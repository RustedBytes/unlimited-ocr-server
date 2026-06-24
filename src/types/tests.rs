use super::*;

#[test]
fn task_spec_defaults_to_default_prompt() {
    let task = TaskSpec::from_text_input(None);

    assert_eq!(task.text_input, None);
}

#[test]
fn task_spec_keeps_prompt_override() {
    let task = TaskSpec::from_text_input(Some("  <image>read this receipt  ".to_string()));

    assert_eq!(task.text_input.as_deref(), Some("<image>read this receipt"));
}

#[test]
fn task_spec_ignores_empty_prompt_override() {
    let task = TaskSpec::from_text_input(Some("   ".to_string()));

    assert_eq!(task.text_input, None);
}
