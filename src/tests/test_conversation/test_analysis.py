import pytest
from datetime import datetime
from src.assistance.types import (
    ConversationAnalysis,
    ActionItem
)


@pytest.fixture
def sample_analysis():
    return ConversationAnalysis(
        key_points=['Point 1', 'Point 2'],
        action_items=[
            ActionItem(
                description='Task 1',
                assignee='John',
                deadline=datetime.now(),
                status='pending',
                priority='high',
                context={},
                timestamp=datetime.now()
            )
        ],
        questions=['Question 1?'],
        follow_up_topics=['Topic 1'],
        context_specific={},
        timestamp=datetime.now()
    )


def test_analysis_validation(sample_analysis):
    """Test conversation analysis validation."""
    assert not sample_analysis.validated
    assert sample_analysis.validation_notes is None

    # Test validation
    sample_analysis.validate(notes='Looks good')
    assert sample_analysis.validated
    assert sample_analysis.validation_notes == 'Looks good'

    # Test invalidation
    sample_analysis.invalidate(notes='Needs revision')
    assert not sample_analysis.validated
    assert sample_analysis.validation_notes == 'Needs revision'


@pytest.mark.asyncio
async def test_conversation_export(conversation_manager, sample_analysis):
    """Test conversation export functionality."""
    # Add some test messages
    await conversation_manager.send_message("Test message 1")
    await conversation_manager.send_message("Test message 2")

    # Set test analysis
    conversation_manager.latest_analysis = sample_analysis

    # Export conversation
    export_data = await conversation_manager.export_conversation()

    # Verify export structure
    assert 'conversation' in export_data
    assert 'analysis' in export_data
    assert 'metadata' in export_data

    # Verify conversation content
    assert len(export_data['conversation']) == 2

    # Verify analysis content
    analysis = export_data['analysis']
    assert len(analysis['key_points']) == 2
    assert len(analysis['action_items']) == 1
    assert not analysis['validated']
