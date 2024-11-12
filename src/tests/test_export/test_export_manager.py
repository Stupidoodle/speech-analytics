import pytest
from unittest.mock import AsyncMock
import json
from datetime import datetime

from src.export.manager import (
    ExportManager,
    ExportFormat,
    Role,
    ConversationContext
)
from src.conversation.manager import ConversationManager


@pytest.fixture
async def mock_conversation_manager():
    manager = AsyncMock(spec=ConversationManager)

    async def mock_send_message(prompt):
        if "summary" in prompt.lower():
            yield AsyncMock(text=json.dumps({
                "key_points": ["Point 1", "Point 2"],
                "action_items": ["Action 1", "Action 2"],
                "insights": ["Insight 1"],
                "follow_ups": ["Follow-up 1"],
                "recommendations": ["Rec 1"]
            }))

    manager.send_message.side_effect = mock_send_message
    return manager


@pytest.fixture
async def mock_conversation_context():
    context = AsyncMock(spec=ConversationContext)
    context.turns = [
        AsyncMock(
            speaker="interviewer",
            content="Tell me about your experience",
            timestamp=datetime.now()
        ),
        AsyncMock(
            speaker="candidate",
            content="I have 5 years of Python experience",
            timestamp=datetime.now()
        )
    ]
    return context


@pytest.fixture
async def export_manager(mock_conversation_manager):
    return ExportManager(
        mock_conversation_manager,
        Role.INTERVIEWER
    )


@pytest.mark.asyncio
async def test_json_export(export_manager, mock_conversation_context):
    """Test JSON export format."""
    result = await export_manager.export_conversation(
        mock_conversation_context,
        ExportFormat.JSON,
        {"purpose": "interview_record"}
    )

    # Parse the bytes result back to JSON
    exported = json.loads(result.decode())

    assert "key_points" in exported
    assert "action_items" in exported
    assert "metadata" in exported
    assert exported["metadata"]["purpose"] == "interview_record"
    assert exported["metadata"]["role"] == "interviewer"
    assert len(exported["conversation"]) == 2


@pytest.mark.asyncio
async def test_role_specific_summary(export_manager,
                                     mock_conversation_context
                                     ):
    """Test role-specific summary generation."""
    # Test interviewer role
    interviewer_summary = await export_manager._generate_summary(
        mock_conversation_context
    )
    assert "key_points" in interviewer_summary
    assert "action_items" in interviewer_summary

    # Test support role
    export_manager.role = Role.SUPPORT_AGENT
    support_summary = await export_manager._generate_summary(
        mock_conversation_context
    )
    assert "insights" in support_summary
    assert "follow_ups" in support_summary


@pytest.mark.asyncio
async def test_summary_validation(export_manager):
    """Test summary validation and feedback incorporation."""
    original_summary = {
        "key_points": ["Original point"],
        "action_items": ["Original action"]
    }

    summary_id = "test_summary"
    export_manager.validated_summaries[summary_id] = original_summary

    updated_summary = await export_manager.validate_summary(
        summary_id,
        "Add point about technical skills",
        Role.INTERVIEWER
    )

    assert updated_summary != original_summary
    assert len(updated_summary["key_points"]) > len(
        original_summary["key_points"]
    )
    assert export_manager.validated_summaries[summary_id] == updated_summary


@pytest.mark.asyncio
async def test_invalid_format_handling(export_manager,
                                       mock_conversation_context
                                       ):
    """Test handling of invalid export formats."""
    with pytest.raises(ValueError) as exc_info:
        await export_manager.export_conversation(
            mock_conversation_context,
            "invalid_format",
            {}
        )
    assert "Unsupported format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_conversation_handling(export_manager):
    """Test handling of empty conversations."""
    empty_context = AsyncMock(spec=ConversationContext)
    empty_context.turns = []

    result = await export_manager.export_conversation(
        empty_context,
        ExportFormat.JSON,
        {}
    )

    exported = json.loads(result.decode())
    assert exported["conversation"] == []
    assert "metadata" in exported
