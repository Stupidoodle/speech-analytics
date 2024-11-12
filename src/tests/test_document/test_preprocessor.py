import pytest
from src.document.preprocessor import DocumentPreprocessor, \
    PreprocessedDocument


@pytest.fixture
def preprocessor():
    return DocumentPreprocessor()


@pytest.fixture
def sample_document():
    return """
    Skills
    - Python Programming
    - AWS Cloud Services
    - Machine Learning

    Experience
    Software Engineer at TechCorp
    - Developed Python applications
    - Managed AWS infrastructure

    Education
    BS in Computer Science
    """


@pytest.mark.asyncio
async def test_section_extraction(preprocessor, sample_document):
    """Test document section extraction."""
    sections = await preprocessor._extract_sections(sample_document)

    assert 'skills' in sections
    assert 'experience' in sections
    assert 'education' in sections

    assert 'Python Programming' in sections['skills']
    assert 'Software Engineer' in sections['experience']
    assert 'Computer Science' in sections['education']


@pytest.mark.asyncio
async def test_keyword_extraction(preprocessor, sample_document):
    """Test keyword extraction functionality."""
    keywords = await preprocessor._extract_keywords(sample_document)

    assert 'python' in keywords
    assert 'aws' in keywords
    assert 'software' in keywords

    # Common words should be excluded
    assert 'and' not in keywords
    assert 'the' not in keywords


@pytest.mark.asyncio
async def test_full_preprocessing(preprocessor, sample_document):
    """Test complete document preprocessing."""
    result = await preprocessor.preprocess(
        sample_document,
        'cv',
        {'source': 'test'}
    )

    assert isinstance(result, PreprocessedDocument)
    assert result.metadata['doc_type'] == 'cv'
    assert result.metadata['source'] == 'test'
    assert len(result.keywords) > 0
    assert len(result.sections) > 0

    # Test section access
    skills_section = result.get_section('skills')
    assert skills_section is not None
    assert 'Python' in skills_section

    # Test keyword density
    assert 'python' in result.content['keyword_density']
    assert 0 <= result.content['keyword_density']['python'] <= 1


@pytest.mark.asyncio
async def test_section_keywords(preprocessor, sample_document):
    """Test keyword extraction from specific sections."""
    result = await preprocessor.preprocess(sample_document, 'cv')

    skills_keywords = result.get_keywords_by_section('skills')
    assert 'python' in skills_keywords
    assert 'aws' in skills_keywords

    education_keywords = result.get_keywords_by_section('education')
    assert 'computer' in education_keywords
    assert 'science' in education_keywords
