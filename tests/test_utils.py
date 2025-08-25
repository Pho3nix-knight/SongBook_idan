import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.utils import chunk_text

def test_chunk_text_normal():
    text = "abcdefghijklmnopqrstuvwxyz"
    expected = [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
        "yz",
    ]
    assert chunk_text(text, chunk_size=10, overlap=2) == expected

def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)
