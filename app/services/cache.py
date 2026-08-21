"""
Cache interface.

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
No caching strategy or layer (query cache, embedding cache, etc.) is
specified in the available architecture source. Generic key/value contract
only — do not select Redis/in-memory/etc. as a frozen choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError
