from typing import Dict, Any, List, Optional
import re
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PreprocessedDocument:
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    keywords: List[str]
    sections: Dict[str, str]
    timestamp: datetime

    def get_section(self, name: str) -> Optional[str]:
        """Get content of a specific section."""
        return self.sections.get(name)

    def get_keywords_by_section(self, section: str) -> List[str]:
        """Get keywords from a specific section."""
        section_content = self.get_section(section)
        if not section_content:
            return []
        return [k for k in self.keywords if k in section_content.lower()]


class DocumentPreprocessor:
    def __init__(self):
        self.section_patterns = {
            'skills': r'(?i)skills?|technical|technologies|expertise',
            'experience': r'(?i)experience|work history|employment',
            'education': r'(?i)education|academic|qualification',
            'projects': r'(?i)projects?|portfolio|works?'
        }

    async def preprocess(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PreprocessedDocument:
        """Preprocess document content."""
        # Extract sections
        sections = await self._extract_sections(content)

        # Extract keywords
        keywords = await self._extract_keywords(content)

        # Process metadata
        processed_metadata = await self._process_metadata(
            metadata or {},
            doc_type
        )

        # Create structured content
        structured_content = await self._structure_content(
            content,
            sections,
            keywords
        )

        return PreprocessedDocument(
            content=structured_content,
            metadata=processed_metadata,
            keywords=keywords,
            sections=sections,
            timestamp=datetime.now()
        )

    async def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract document sections using patterns."""
        sections = {}
        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            # Check if line is a section header
            for section, pattern in self.section_patterns.items():
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    if current_section:
                        sections[current_section] = '\n'.join(section_content)
                    current_section = section
                    section_content = []
                    break
            else:
                if current_section:
                    section_content.append(line)

        # Add last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)

        return sections

    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from content."""
        # Convert to lowercase and split into words
        words = re.findall(r'\w+', content.lower())

        # Remove common words and short terms
        common_words = {'and', 'or', 'the', 'in', 'at', 'to', 'for'}
        keywords = [
            word for word in words
            if word not in common_words and len(word) > 2
        ]

        # Count frequencies
        freq = {}
        for word in keywords:
            freq[word] = freq.get(word, 0) + 1

        # Return top keywords by frequency
        sorted_keywords = sorted(
            freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [k for k, _ in sorted_keywords[:50]]  # Top 50 keywords

    async def _process_metadata(
        self,
        metadata: Dict[str, Any],
        doc_type: str
    ) -> Dict[str, Any]:
        """Process and validate document metadata."""
        processed = metadata.copy()
        processed.update({
            'doc_type': doc_type,
            'processed_at': datetime.now().isoformat(),
            'version': '1.0'
        })
        return processed

    async def _structure_content(
        self,
        content: str,
        sections: Dict[str, str],
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Create structured content representation."""
        return {
            'full_text': content,
            'sections': sections,
            'keyword_density': await self._calculate_keyword_density(
                content,
                keywords
            ),
            'section_summaries': {
                name: await self._summarize_section(content)
                for name, content in sections.items()
            }
        }

    async def _calculate_keyword_density(
        self,
        content: str,
        keywords: List[str]
    ) -> Dict[str, float]:
        """Calculate keyword density in content."""
        total_words = len(content.split())
        density = {}

        for keyword in keywords:
            count = len(re.findall(rf'\b{keyword}\b', content.lower()))
            density[keyword] = count / total_words if total_words > 0 else 0

        return density

    async def _summarize_section(self, content: str) -> str:
        """Create brief summary of section content."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        # Remove empty sentences and trim
        sentences = [s.strip() for s in sentences if s.strip()]

        # Return first 2 sentences if available
        if len(sentences) >= 2:
            return '. '.join(sentences[:2]) + '.'
        return '. '.join(sentences) + '.' if sentences else ''
