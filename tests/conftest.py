"""Shared pytest fixtures and lightweight test doubles.

These stand in for LangChain message objects so we can exercise the extraction
helpers in ``main`` without touching the network or the Anthropic API.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the project root importable when tests are run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_message(**attrs):
    """Return a duck-typed stand-in for a LangChain message.

    The extraction helpers only read attributes via ``getattr`` (``name``,
    ``content``, ``usage_metadata``), so a SimpleNamespace is a faithful double.
    """
    attrs.setdefault("name", None)
    attrs.setdefault("content", None)
    return SimpleNamespace(**attrs)


@pytest.fixture
def message_factory():
    return make_message
