from langgraph_tools.messages.message_builder import MessageBuilder


def test_build():
    builder = MessageBuilder()
    text = "This is a test"
    message = builder.build({"summarize"}, text)
    assert (
        message
        == 'summarize using summarization tool on this blob of text. "This is a test"'
    )

    text = "This is a test"
    message = builder.build({"word_count", "summarize"}, text)
    assert (
        message
        == 'Perform the following actions: count the number of words, summarize using summarization tool on this blob of text. "This is a test"'  # noqa: E501
    )
