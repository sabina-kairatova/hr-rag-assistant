"""
Test for the caching layer.
"""

import time
from app.cache import ResponseCahce

class TestResponseCahce:
    """Test the response cache."""

    def setup_method(self):
        self.cache = ResponseCahce(ttl_seconds=2)

    def test_cache_miss_returns_none(self):
        assert self.cache.get("unknown query") is None

    def test_cache_hit_returns_response(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("What is Python?")
        assert result == "A programming language."

    def test_case_insensitive_matching(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("what is python?")
        assert result == "A programming language."

    def test_ttl_expiration(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("what is python?")
        assert result == "A programming language."
        time.sleep(2.5)
        result = self.cache.get("what is python?")
        assert result is None

    def test_stats_tracking(self):
        self.cache.get("miss1")
        self.cache.get("miss2")
        self.cache.set("hit1", "value")
        self.cache.get("hit1")

        stats = self.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 1