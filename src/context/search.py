# src/context/search.py
from typing import List, Dict, Any
from src.context.context_manager import ContextManager, ContextEntry

class ContextSearchEngine:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.index: Dict[str, Dict[str, List[str]]] = {}

    async def build_index(self, context_id: str):
        """Build an inverted index for efficient context search."""
        self.index[context_id] = {}
        entries = await self.context_manager.store.get_context(context_id)
        for entry_key, entry in entries.items():
            # Tokenize and normalize the content
            tokens = self._tokenize(entry.content)
            for token in tokens:
                if token not in self.index[context_id]:
                    self.index[context_id][token] = []
                self.index[context_id][token].append(entry_key)

    def _tokenize(self, content: str) -> List[str]:
        """Tokenize and normalize the content for indexing."""
        # Implement tokenization and normalization logic here
        # This can include lowercasing, removing punctuation, stemming, etc.
        # Return a list of normalized tokens
        return content.lower().split()

    async def search(self, context_id: str, query: str) -> List[ContextEntry]:
        """Search for context entries based on a query."""
        if context_id not in self.index:
            await self.build_index(context_id)

        # Tokenize and normalize the query
        query_tokens = self._tokenize(query)

        # Perform the search using the inverted index
        entry_keys = set()
        for token in query_tokens:
            if token in self.index[context_id]:
                entry_keys.update(self.index[context_id][token])

        # Retrieve the context entries based on the matched keys
        entries = []
        for entry_key in entry_keys:
            entry = await self.context_manager.store.get_entry(context_id, entry_key)
            if entry:
                entries.append(entry)

        return entries

    async def filter_context(self, context_id: str, criteria: Dict[str, Any]) -> List[ContextEntry]:
        """Filter context entries based on specified criteria."""
        entries = await self.context_manager.store.get_context(context_id)
        filtered_entries = []
        for entry in entries.values():
            if self._match_criteria(entry, criteria):
                filtered_entries.append(entry)
        return filtered_entries

    def _match_criteria(self, entry: ContextEntry, criteria: Dict[str, Any]) -> bool:
        """Check if a context entry matches the specified criteria."""
        for key, value in criteria.items():
            if key == "timestamp":
                # Handle timestamp-based filtering
                if "min" in value and entry.timestamp < value["min"]:
                    return False
                if "max" in value and entry.timestamp > value["max"]:
                    return False
            elif key == "priority":
                # Handle priority-based filtering
                if "min" in value and entry.priority < value["min"]:
                    return False
                if "max" in value and entry.priority > value["max"]:
                    return False
            elif getattr(entry, key, None) != value:
                return False
        return True