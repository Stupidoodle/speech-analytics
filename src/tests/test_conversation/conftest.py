import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from src.conversation.manager import ConversationManager


@pytest_asyncio.fixture
async def conversation_manager(mock_bedrock_client):
    with patch("aioboto3.Session.client", return_value=mock_bedrock_client):
        async with ConversationManager(
            region="us-east-1",
            model_id="amazon.titan-text-premier-v1:0",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            client=mock_bedrock_client
        ) as manager:
            yield manager


@pytest.fixture
def mock_bedrock_client():
    """Mock the Bedrock client."""
    mock_client = AsyncMock()

    # Create a special async_generator_mock for converse_stream
    async def mock_converse_stream(*args, **kwargs):
        return {'stream': MockStreamIterator()}

    mock_client.converse_stream = mock_converse_stream
    return mock_client


class MockStreamIterator:
    """Mock async iterator for stream responses."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        response = self.responses[self.index]
        self.index += 1
        return response


@pytest.fixture
def mock_stream_events():
    """Create mock stream events."""
    def create_events(text_chunks=None, include_error=False):
        responses = []

        if include_error:
            responses.append({
                'modelStreamErrorException': {
                    'message': 'Stream error',
                    'originalStatusCode': 400,
                    'originalMessage': 'Stream error'
                }
            })
            return MockStreamIterator(responses)

        if text_chunks:
            for text in text_chunks:
                responses.append({
                    'contentBlockDelta': {
                        'delta': {'text': text}
                    }
                })

        responses.append({
            'messageStop': {
                'stopReason': 'end'
            }
        })

        responses.append({
            'metadata': {
                'key': 'value'
            }
        })

        return MockStreamIterator(responses)

    return create_events


@pytest.fixture
def sample_document():
    """Fixture for a sample document."""
    return {
        "content": b"Sample document content",
        "name": "test_document.pdf",
        "format": "pdf"
    }


@pytest.fixture
def sample_conversation():
    """Fixture to provide a sample conversation history."""
    return [
        {
            "role": "user",
            "content": [{"text": "Hello"}],
            "timestamp": "2023-11-12T12:00:00"
        },
        {
            "role": "assistant",
            "content": [{"text": "Hi there!"}],
            "timestamp": "2023-11-12T12:00:05"
        }
    ]
