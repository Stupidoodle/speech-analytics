import pytest
from datetime import datetime
from src.conversation.types import Message, Role
from src.conversation.exceptions import ConversationError


@pytest.mark.asyncio
async def test_initialization(conversation_manager):
    """Test conversation manager initialization."""
    assert conversation_manager.model_id == "amazon.titan-text-premier-v1:0"
    assert conversation_manager.inference_config.max_tokens == 100
    assert conversation_manager.inference_config.temperature == 0.7
    assert conversation_manager.inference_config.top_p == 0.9
    assert len(conversation_manager.messages) == 0
    assert len(conversation_manager.system_messages) == 0


@pytest.mark.asyncio
async def test_send_message_success(
    conversation_manager,
    mock_bedrock_client
):
    """Test successful message sending and response streaming."""
    # Create a list of events to be yielded
    """mock_events = [
        {
            'contentBlockDelta': {
                'delta': {'text': 'Hello'}
            }
        },
        {
            'contentBlockDelta': {
                'delta': {'text': ' world!'}
            }
        },
        {
            'messageStop': {
                'stopReason': 'end'
            }
        },
        {
            'metadata': {
                'key': 'value'
            }
        }
    ]"""

    # Configure mock response
    async def mock_converse_stream(*args, **kwargs):
        # return {'stream': MockStreamIterator(responses=mock_events)}
        pass

    mock_bedrock_client.converse_stream = mock_converse_stream

    # Send message and collect responses
    responses = []
    async for response in conversation_manager.send_message("Hi"):
        responses.append(response)

    # Assertions
    assert len(responses) == 4, f"Expected 4 responses, got {len(responses)}"
    assert responses[0].text == 'Hello'
    assert responses[1].text == ' world!'
    assert responses[2].stop_reason == 'end'
    assert responses[3].metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_send_message_error(
    conversation_manager,
    mock_bedrock_client
):
    """Test error handling in message sending."""
    # Create error response
    async def mock_error_response():
        yield {
            'modelStreamErrorException': {
                'message': 'Stream error',
                'originalStatusCode': 400,
                'originalMessage': 'Stream error'
            }
        }

    mock_bedrock_client.converse_stream.return_value = {
        'stream': mock_error_response()
    }

    with pytest.raises(ConversationError):
        async for _ in conversation_manager.send_message("Hi"):
            pass


@pytest.mark.asyncio
async def test_add_system_prompt(conversation_manager):
    """Test adding system prompt."""
    test_prompt = "You are a helpful assistant."
    await conversation_manager.add_system_prompt(test_prompt)

    assert len(conversation_manager.system_messages) == 1
    assert conversation_manager.system_messages[0]['text'] == test_prompt


@pytest.mark.asyncio
async def test_add_document(conversation_manager, sample_document):
    """Test adding document to conversation."""
    await conversation_manager.add_document(
        content=sample_document['content'],
        name=sample_document['name'],
        format=sample_document['format']
    )

    assert len(conversation_manager.system_messages) == 1
    system_msg = conversation_manager.system_messages[0]
    assert "Context from document" in system_msg['text']
    assert system_msg['document']['name'] == sample_document['name']
    assert system_msg['document']['format'] == sample_document['format']


@pytest.mark.asyncio
async def test_send_message_with_files(
    conversation_manager,
    mock_bedrock_client,
    sample_document
):
    """Test sending message with attached files."""
    # Create the mock response
    async def mock_response():
        response = {
            'stream': [
                {
                    'contentBlockDelta': {
                        'delta': {'text': 'Response'}
                    }
                },
                {
                    'messageStop': {
                        'stopReason': 'end'
                    }
                },
                {
                    'metadata': {
                        'key': 'value'
                    }
                }
            ]
        }
        for event in response['stream']:
            yield event

    mock_bedrock_client.converse_stream.return_value = {
        'stream': mock_response()
        }

    files = [{
        'document': {
            'format': sample_document['format'],
            'name': sample_document['name'],
            'source': sample_document['content']
        }
    }]

    responses = []
    async for response in conversation_manager.send_message(
        "Check this document",
        files=files
    ):
        responses.append(response)

    last_message = conversation_manager.messages[-2]
    assert len(last_message.content) == 2  # text + document
    assert 'document' in last_message.content[1]


@pytest.mark.asyncio
async def test_get_conversation_history(
    conversation_manager,
    sample_conversation
):
    """Test retrieving conversation history."""
    for msg in sample_conversation:
        conversation_manager.messages.append(
            Message(
                role=Role(msg['role']),
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp'])
            )
        )

    history = conversation_manager.get_conversation_history()
    assert len(history) == len(sample_conversation)
    for orig, hist in zip(sample_conversation, history):
        assert hist['role'] == orig['role']
        assert hist['content'] == orig['content']
        assert hist['timestamp'] == orig['timestamp']


@pytest.mark.asyncio
async def test_clear_conversation(conversation_manager, sample_conversation):
    """Test clearing conversation history."""
    await conversation_manager.add_system_prompt("System prompt")
    for msg in sample_conversation:
        conversation_manager.messages.append(
            Message(
                role=Role(msg['role']),
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp'])
            )
        )

    await conversation_manager.clear_conversation()
    assert len(conversation_manager.messages) == 0
    assert len(conversation_manager.system_messages) == 0
