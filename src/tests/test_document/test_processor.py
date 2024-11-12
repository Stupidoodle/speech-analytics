import pytest
from unittest.mock import AsyncMock
import json
from src.document.processor import DocumentProcessor
from src.conversation.manager import ConversationManager


@pytest.fixture
async def mock_ai_analysis():
    return {
        "skills": ["Python", "AWS", "ML"],
        "experience": {
            "years": 5,
            "highlights": ["Led team of 10", "Improved performance by 50%"]
        },
        "education": "BS Computer Science",
        "projects": ["AI chatbot", "Cloud migration"],
        "achievements": ["Best developer 2023"]
    }


@pytest.fixture
async def processor(mock_ai_analysis):
    ai_manager = AsyncMock(spec=ConversationManager)
    ai_manager.send_message.return_value = [
        AsyncMock(text=json.dumps(mock_ai_analysis))
    ]
    return DocumentProcessor(ai_manager)


@pytest.mark.asyncio
async def test_ai_document_processing(processor):
    """Test AI-based document processing."""
    content = "Sample CV content..."
    result = await processor.process_document(content, "cv")

    assert "skills" in result
    assert "experience" in result
    assert isinstance(result["skills"], list)
    assert len(result["skills"]) > 0


@pytest.mark.asyncio
async def test_context_update(processor):
    """Test AI-based context updating."""
    # Initial processing
    await processor.process_document("Initial CV", "cv")

    # Update with new information
    update_result = await processor.update_context(
        "cv",
        "New certification: AWS Solutions Architect"
    )

    assert "skills" in update_result
    assert any("aws" in skill.lower() for skill in update_result["skills"])
