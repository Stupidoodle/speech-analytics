import pytest
import os
import asyncio
import aioboto3
import json

from src.realtime.processor import RealtimeProcessor
from src.assistance.enhanced_assistant import Role
from src.conversation.manager import ConversationManager
from src.transcription.aws_transcribe import TranscribeManager


@pytest.fixture(scope="session")
async def aws_credentials_available():
    # Check if environment variables are set
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        return True

    # Check if credentials file exists and can be loaded by boto3
    try:
        session = aioboto3.Session()
        credentials = await session.get_credentials()
        return credentials is not None
    except Exception:
        return False


@pytest.mark.asyncio
async def test_example(aws_credentials_available):
    if not aws_credentials_available:
        pytest.skip("AWS credentials not found\
            in environment or configuration file")


@pytest.fixture
def test_audio_file():
    """Get path to test audio file."""
    return os.path.join(
        os.path.dirname(__file__),
        'test_data',
        'test_interview.wav'
    )


@pytest.fixture
def test_cv_file():
    """Get path to test CV file."""
    return os.path.join(
        os.path.dirname(__file__),
        'test_data',
        'test_cv.pdf'
    )


@pytest.fixture
async def conversation_manager():
    """Create real conversation manager."""
    async with ConversationManager(
        region="us-east-1",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    ) as manager:
        yield manager


@pytest.mark.asyncio
async def test_real_transcription():
    """Test real transcription with AWS Transcribe."""
    transcribe = TranscribeManager(region="us-east-1")

    # Read test audio file in chunks
    chunk_size = 1024 * 8
    results = []

    with open(test_audio_file, 'rb') as f:
        while chunk := f.read(chunk_size):
            transcript = await transcribe.process_audio(chunk)
            if transcript:
                results.append(transcript)

    # Verify we got some transcription
    assert len(results) > 0
    print(f"Transcription results: {results}")


@pytest.mark.asyncio
async def test_real_conversation(conversation_manager):
    """Test real conversation with Claude."""
    responses = []
    async for response in conversation_manager.send_message(
        "What are good questions to ask in a technical interview?"
    ):
        if response.text:
            responses.append(response.text)

    full_response = ''.join(responses)
    assert len(full_response) > 0
    assert "technical" in full_response.lower()
    print(f"Claude response: {full_response}")


@pytest.mark.asyncio
async def test_real_document_processing(conversation_manager):
    """Test processing a real document."""
    with open(test_cv_file, 'rb') as f:
        cv_content = f.read()

    processor = RealtimeProcessor(
        role=Role.INTERVIEWER,
        context_type="interview"
    )

    # Add the document
    await processor.add_context(cv_content, "cv")

    # Test getting suggestions based on the CV
    responses = []
    async for response in processor.assistant.get_suggestions(
        "The candidate mentioned Python experience"
    ):
        if response.text:
            responses.append(response.text)

    full_response = ''.join(responses)
    assert len(full_response) > 0
    print(f"Suggestions based on CV: {full_response}")


@pytest.mark.asyncio
async def test_full_integration():
    """Test full integration of all components."""
    results = {
        'transcripts': [],
        'suggestions': [],
        'analyses': []
    }

    def on_transcript(t):
        results['transcripts'].append(t)
        print(f"Transcript: {t}")

    def on_suggestion(s):
        results['suggestions'].append(s)
        print(f"Suggestion: {s}")

    def on_analysis(a):
        results['analyses'].append(a)
        print(f"Analysis: {a}")

    processor = RealtimeProcessor(
        role=Role.INTERVIEWER,
        context_type="interview",
        on_transcript=on_transcript,
        on_suggestion=on_suggestion,
        on_analysis=on_analysis
    )

    # Add test CV
    with open(test_cv_file, 'rb') as f:
        await processor.add_context(f.read(), "cv")

    # Start processing
    await processor.start(region="us-east-1")

    # Process test audio file in chunks
    chunk_size = 1024 * 8
    with open(test_audio_file, 'rb') as f:
        while chunk := f.read(chunk_size):
            await asyncio.sleep(0.1)  # Simulate real-time
            await processor._process_chunk(chunk)

    # Wait for processing to complete
    await asyncio.sleep(2)

    # Stop processing
    await processor.stop()

    # Verify results
    assert len(results['transcripts']) > 0
    assert len(results['suggestions']) > 0
    assert len(results['analyses']) > 0

    # Save results for manual inspection
    with open('integration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Full integration test results saved to\
        'integration_test_results.json'")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with real services."""
    processor = RealtimeProcessor(
        role=Role.INTERVIEWER,
        context_type="interview",
        on_error=lambda e: print(f"Error caught: {e}")
    )

    # Test with invalid audio
    invalid_chunk = b'not_audio_data'
    await processor._process_chunk(invalid_chunk)

    # Test with invalid document
    with pytest.raises(Exception):
        await processor.add_context(b'not_a_pdf', "cv")

    # Test with invalid region
    with pytest.raises(Exception):
        await processor.start(region="invalid-region")
