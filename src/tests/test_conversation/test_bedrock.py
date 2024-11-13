"""Tests for Bedrock client integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pytest_asyncio.plugin import pytest_fixture_setup

from src.conversation.bedrock import BedrockClient
from src.conversation.types import BedrockConfig, MessageRole, InferenceConfig
from src.conversation.exceptions import (
    ServiceError,
    ServiceConnectionError,
    ServiceQuotaError
)


@pytest.fixture
def bedrock_config():
    """Create test Bedrock configuration."""
    return BedrockConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        inference_config=InferenceConfig(
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9
        )
    )


@pytest.fixture
def mock_bedrock_stream():
    """Create mock Bedrock stream responses."""

    async def stream_generator():
        yield {
            "messageStart": {
                "role": "assistant"
            }
        }
        yield {
            "contentBlockDelta": {
                "delta": {
                    "text": "Hello"
                },
                "contentBlockIndex": 0
            }
        }
        yield {
            "contentBlockDelta": {
                "delta": {
                    "text": " world!"
                },
                "contentBlockIndex": 0
            }
        }
        yield {
            "messageStop": {
                "stopReason": "end_turn"
            }
        }
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 20
                },
                "metrics": {
                    "latencyMs": 100
                }
            }
        }

    return stream_generator


@pytest.fixture
async def mock_bedrock_client(mock_bedrock_stream):
    """Create mock Bedrock client."""
    client = AsyncMock()
    client.invoke_model_with_response_stream.return_value = {
        "stream": mock_bedrock_stream()
    }
    return client


@pytest.fixture
def mock_bedrock_steam_with_error():
    """Create mock Bedrock stream responses with error."""

    async def stream_generator():
        yield {
            "modelStreamErrorException": {
                "message": "Service error",
                "originalStatusCode": 500,
                "originalMessage": "Internal error"
            }
        }

    return stream_generator


@pytest.fixture
async def mock_bedrock_client_with_error_stream(mock_bedrock_steam_with_error):
    """Create mock Bedrock client with error stream."""
    client = AsyncMock()
    client.invoke_model_with_response_stream.return_value = {
        "stream": mock_bedrock_steam_with_error()
    }
    return client

@pytest.fixture
def mock_bedrock_stream_with_throttling():
    """Create mock Bedrock stream responses with throttling error."""

    async def stream_generator():
        yield {
            "throttlingException": {
                "message": "Rate exceeded"
            }
        }

    return stream_generator


@pytest.fixture
async def mock_bedrock_client_with_throttling(mock_bedrock_stream_with_throttling):
    """Create mock Bedrock client with throttling error."""
    client = AsyncMock()
    client.invoke_model_with_response_stream.return_value = {
        "stream": mock_bedrock_stream_with_throttling()
    }
    return client


@pytest.mark.asyncio
async def test_bedrock_client_initialization(bedrock_config):
    """Test BedrockClient initialization."""

    client = BedrockClient("us-west-2", bedrock_config)
    assert client.region == "us-west-2"
    assert client.config == bedrock_config
    assert client._client is None


@pytest.mark.asyncio
async def test_message_formatting(bedrock_config):
    """Test message formatting for Bedrock API."""

    client = BedrockClient("us-west-2", bedrock_config)

    messages = [
        {
            "role": MessageRole.USER,
            "content": ["Hello assistant!"]
        },
        {
            "role": MessageRole.ASSISTANT,
            "content": [
                "Hi! I can help you.",
                {
                    "tool_use": {
                        "tool_use_id": "123",
                        "name": "calculator",
                        "input": {"operation": "add", "numbers": [1, 2]}
                    }
                }
            ]
        }
    ]

    formatted = client._format_messages(messages)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"][0]["text"] == "Hello assistant!"

    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"][0]["text"] == "Hi! I can help you."
    assert "toolUse" in formatted[1]["content"][1]
    assert formatted[1]["content"][1]["toolUse"]["toolUseId"] == "123"


@pytest.mark.asyncio
async def test_successful_stream_generation(
        bedrock_config,
        mock_bedrock_client,
        mocker,
):
    """Test successful stream generation."""

    mock_client_instance = await mock_bedrock_client

    mock_client_context_manager = AsyncMock()
    mock_client_context_manager.__aenter__.return_value = mock_client_instance
    mocker.patch("aioboto3.Session.client", return_value=mock_client_context_manager)

    client = BedrockClient("us-west-2", bedrock_config)
    await client.__aenter__()

    messages = [
        {
            "role": MessageRole.USER,
            "content": ["Test message"]
        }
    ]

    responses = []
    async for response in client.generate_stream(messages):
        responses.append(response)

    await client.__aexit__(None, None, None)

    assert len(responses) == 4  # Excluding messageStart
    assert responses[0].content.text == "Hello"
    assert responses[1].content.text == " world!"
    assert responses[2].stop_reason == "end_turn"
    assert responses[3].metadata.usage["inputTokens"] == 10


@pytest.mark.asyncio
async def test_service_error_handling(bedrock_config, mock_bedrock_client_with_error_stream, mocker):
    """Test handling of service errors."""

    mock_client_instance = await mock_bedrock_client_with_error_stream

    mock_client_context_manager = AsyncMock()
    mock_client_context_manager.__aenter__.return_value = mock_client_instance
    mocker.patch("aioboto3.Session.client", return_value=mock_client_context_manager)

    client = BedrockClient("us-west-2", bedrock_config)
    await client.__aenter__()


    with pytest.raises(ServiceError) as exc_info:
        async for _ in client.generate_stream([{
            "role": MessageRole.USER,
            "content": ["Test"]
        }]):
            pass

    await client.__aexit__(None, None, None)

    assert exc_info.value.service == "bedrock"
    assert exc_info.value.error_code == "500"


@pytest.mark.asyncio
async def test_throttling_handling(bedrock_config, mock_bedrock_client_with_throttling, mocker):
    """Test handling of throttling errors."""

    mock_client_instance = await mock_bedrock_client_with_throttling

    mock_client_context_manager = AsyncMock()
    mock_client_context_manager.__aenter__.return_value = mock_client_instance
    mocker.patch("aioboto3.Session.client", return_value=mock_client_context_manager)

    client = BedrockClient("us-west-2", bedrock_config)
    await client.__aenter__()

    with pytest.raises(ServiceQuotaError) as exc_info:
        async for _ in client.generate_stream([{
            "role": MessageRole.USER,
            "content": ["Test"]
        }]):
            pass

    await client.__aexit__(None, None, None)

    assert exc_info.value.service == "bedrock"


@pytest.mark.asyncio
async def test_uninitialized_client(bedrock_config):
    """Test error when using uninitialized client."""
    client = BedrockClient("us-west-2", bedrock_config)

    with pytest.raises(ServiceConnectionError) as exc_info:
        async for _ in client.generate_stream([]):
            pass

    assert "not initialized" in str(exc_info.value)