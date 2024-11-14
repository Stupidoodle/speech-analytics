"""Utility functions for context operations."""
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import re
from collections import defaultdict

from .types import (
    ContextLevel,
    ContextSource,
    ContextState,
    ContextEntry,
    ContextMetadata
)


class ContextMerger:
    """Handles merging of context entries."""

    @staticmethod
    async def merge_entries(
        entries: List[ContextEntry],
        strategy: str = "latest_wins"
    ) -> ContextEntry:
        """Merge multiple context entries.

        Args:
            entries: Entries to merge
            strategy: Merge strategy

        Returns:
            Merged entry
        """
        if not entries:
            raise ValueError("No entries to merge")
        if len(entries) == 1:
            return entries[0]

        strategies = {
            "latest_wins": ContextMerger._merge_latest_wins,
            "combine_all": ContextMerger._merge_combine_all,
            "priority_based": ContextMerger._merge_priority_based
        }

        merge_func = strategies.get(
            strategy,
            ContextMerger._merge_latest_wins
        )
        return await merge_func(entries)

    @staticmethod
    async def _merge_latest_wins(
        entries: List[ContextEntry]
    ) -> ContextEntry:
        """Merge using latest entry.

        Args:
            entries: Entries to merge

        Returns:
            Merged entry
        """
        # Get latest entry
        latest = max(
            entries,
            key=lambda x: x.metadata.timestamp
        )

        # Combine metadata
        all_tags = set()
        all_refs = set()
        for entry in entries:
            all_tags.update(entry.metadata.tags)
            all_refs.update(entry.metadata.references)

        # Create merged metadata
        merged_metadata = ContextMetadata(
            source=latest.metadata.source,
            level=latest.metadata.level,
            state=latest.metadata.state,
            timestamp=datetime.now(),
            tags=all_tags,
            references=all_refs,
            custom_data=latest.metadata.custom_data
        )

        return ContextEntry(
            id=latest.id,
            content=latest.content,
            metadata=merged_metadata
        )

    @staticmethod
    async def _merge_combine_all(
        entries: List[ContextEntry]
    ) -> ContextEntry:
        """Merge by combining all content.

        Args:
            entries: Entries to merge

        Returns:
            Merged entry
        """
        # Combine all content
        if all(isinstance(e.content, dict) for e in entries):
            # Merge dictionaries
            merged_content = {}
            for entry in entries:
                merged_content.update(entry.content)
        else:
            # Concatenate strings
            merged_content = "\n".join(
                str(e.content) for e in entries
            )

        # Get the highest level
        highest_level = max(
            entries,
            key=lambda x: list(ContextLevel).index(x.metadata.level)
        ).metadata.level

        # Combine metadata
        all_tags = set()
        all_refs = set()
        custom_data = {}
        for entry in entries:
            all_tags.update(entry.metadata.tags)
            all_refs.update(entry.metadata.references)
            custom_data.update(entry.metadata.custom_data)

        merged_metadata = ContextMetadata(
            source=entries[0].metadata.source,
            level=highest_level,
            state=ContextState.ACTIVE,
            timestamp=datetime.now(),
            tags=all_tags,
            references=all_refs,
            custom_data=custom_data
        )

        return ContextEntry(
            id=entries[0].id,
            content=merged_content,
            metadata=merged_metadata
        )

    @staticmethod
    async def _merge_priority_based(
        entries: List[ContextEntry]
    ) -> ContextEntry:
        """Merge based on context priority.

        Args:
            entries: Entries to merge

        Returns:
            Merged entry
        """
        # Sort by level priority
        sorted_entries = sorted(
            entries,
            key=lambda x: list(ContextLevel).index(x.metadata.level),
            reverse=True
        )

        # Take content from the highest priority
        highest_priority = sorted_entries[0]

        # Combine metadata
        all_tags = set()
        all_refs = set()
        for entry in entries:
            all_tags.update(entry.metadata.tags)
            all_refs.update(entry.metadata.references)

        merged_metadata = ContextMetadata(
            source=highest_priority.metadata.source,
            level=highest_priority.metadata.level,
            state=ContextState.ACTIVE,
            timestamp=datetime.now(),
            tags=all_tags,
            references=all_refs,
            custom_data=highest_priority.metadata.custom_data
        )

        return ContextEntry(
            id=highest_priority.id,
            content=highest_priority.content,
            metadata=merged_metadata
        )


class ContextAnalyzer:
    """Analyzes context patterns and relationships."""

    @staticmethod
    def analyze_relationships(
        entries: List[ContextEntry]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze relationships between entries.

        Args:
            entries: Entries to analyze

        Returns:
            Relationship analysis
        """
        relationships = {
            "references": [],
            "tags": [],
            "temporal": []
        }

        # Analyze reference relationships
        ref_map = defaultdict(list)
        for entry in entries:
            for ref in entry.metadata.references:
                ref_map[ref].append(entry.id)

        for ref, entry_ids in ref_map.items():
            if len(entry_ids) > 1:
                relationships["references"].append({
                    "type": "shared_reference",
                    "reference": ref,
                    "entries": entry_ids
                })

        # Analyze tag relationships
        tag_map = defaultdict(list)
        for entry in entries:
            for tag in entry.metadata.tags:
                tag_map[tag].append(entry.id)

        for tag, entry_ids in tag_map.items():
            if len(entry_ids) > 1:
                relationships["tags"].append({
                    "type": "shared_tag",
                    "tag": tag,
                    "entries": entry_ids
                })

        # Analyze temporal relationships
        sorted_entries = sorted(
            entries,
            key=lambda x: x.metadata.timestamp
        )
        for i in range(len(sorted_entries)-1):
            time_diff = (
                sorted_entries[i+1].metadata.timestamp -
                sorted_entries[i].metadata.timestamp
            ).total_seconds()
            if time_diff < 60:  # Within a minute
                relationships["temporal"].append({
                    "type": "temporal_proximity",
                    "entries": [
                        sorted_entries[i].id,
                        sorted_entries[i+1].id
                    ],
                    "time_difference": time_diff
                })

        return relationships


class ContextSearch:
    """Context search functionality."""

    @staticmethod
    def search_content(
        entries: List[ContextEntry],
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Search context content.

        Args:
            entries: Entries to search
            query: Search query
            case_sensitive: Whether to match case

        Returns:
            Search results
        """
        results = []
        pattern = re.compile(
            query if case_sensitive else query.lower()
        )

        for entry in entries:
            content_str = str(entry.content)
            if not case_sensitive:
                content_str = content_str.lower()

            matches = list(pattern.finditer(content_str))
            if matches:
                results.append({
                    "entry_id": entry.id,
                    "matches": [
                        {
                            "text": m.group(),
                            "start": m.start(),
                            "end": m.end()
                        }
                        for m in matches
                    ],
                    "source": entry.metadata.source,
                    "timestamp": entry.metadata.timestamp
                })

        return sorted(
            results,
            key=lambda x: len(x["matches"]),
            reverse=True
        )


class ContextFormatter:
    """Formats context for display/export."""

    @staticmethod
    def format_entry(
        entry: ContextEntry,
        format: str = "text"
    ) -> str:
        """Format context entry.

        Args:
            entry: Entry to format
            format: Output format

        Returns:
            Formatted entry
        """
        formats = {
            "text": ContextFormatter._format_text,
            "html": ContextFormatter._format_html,
            "markdown": ContextFormatter._format_markdown
        }

        format_func = formats.get(format, ContextFormatter._format_text)
        return format_func(entry)

    @staticmethod
    def _format_text(entry: ContextEntry) -> str:
        """Format as plain text.

        Args:
            entry: Entry to format

        Returns:
            Text format
        """
        return (
            f"ID: {entry.id}\n"
            f"Content: {entry.content}\n"
            f"Source: {entry.metadata.source}\n"
            f"Level: {entry.metadata.level}\n"
            f"State: {entry.metadata.state}\n"
            f"Time: {entry.metadata.timestamp}\n"
            f"Tags: {', '.join(entry.metadata.tags)}\n"
            f"References: {', '.join(entry.metadata.references)}"
        )

    @staticmethod
    def _format_html(entry: ContextEntry) -> str:
        """Format as HTML.

        Args:
            entry: Entry to format

        Returns:
            HTML format
        """
        return f"""
        <div class="context-entry">
            <div class="header">
                <span class="id">{entry.id}</span>
                <span class="source">{entry.metadata.source}</span>
                <span class="level">{entry.metadata.level}</span>
                <span class="state">{entry.metadata.state}</span>
            </div>
            <div class="content">{entry.content}</div>
            <div class="metadata">
                <div class="timestamp">{entry.metadata.timestamp}</div>
                <div class="tags">{', '.join(entry.metadata.tags)}</div>
                <div class="refs">{', '.join(entry.metadata.references)}</div>
            </div>
        </div>
        """

    @staticmethod
    def _format_markdown(entry: ContextEntry) -> str:
        """Format as Markdown.

        Args:
            entry: Entry to format

        Returns:
            Markdown format
        """
        return (
            f"## Context Entry: {entry.id}\n\n"
            f"**Source:** {entry.metadata.source}  \n"
            f"**Level:** {entry.metadata.level}  \n"
            f"**State:** {entry.metadata.state}  \n\n"
            f"### Content\n{entry.content}\n\n"
            f"### Metadata\n"
            f"- Time: {entry.metadata.timestamp}\n"
            f"- Tags: {', '.join(entry.metadata.tags)}\n"
            f"- References: {', '.join(entry.metadata.references)}\n"
        )