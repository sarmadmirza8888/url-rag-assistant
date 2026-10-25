import pytest

from url_rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n ") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world")
    assert chunks == [{"chunk_id": 0, "text": "hello world"}]


def test_chunk_ids_are_sequential_and_sizes_bounded():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["text"]) <= 800 for c in chunks)


def test_overlap_is_shared_between_neighbours():
    text = "".join(chr(65 + (i % 26)) for i in range(1000))  # A..Z repeating
    chunks = chunk_text(text, chunk_size=300, overlap=100)
    assert chunks[0]["text"][-100:] == chunks[1]["text"][:100]


def test_all_text_is_covered():
    text = "".join(chr(65 + (i % 26)) for i in range(1234))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert chunks[0]["text"] == text[:300]
    assert chunks[-1]["text"].endswith(text[-20:])


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, overlap=overlap)
