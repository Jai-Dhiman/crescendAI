import teacher_model.tool_format as tf


def test_tool_format_module_marked_deprecated():
    doc = (tf.__doc__ or "").lower()
    assert "deprecated" in doc
    assert "stage 1" in doc or "chat-template-native" in doc or "apply_chat_template" in doc
