from langchain_core.documents import Document

from src.generation.answer import format_context, unique_sources


def make_document(source: str, section: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "section": section,
        },
    )


def test_format_context_labels_each_document_with_source_and_section() -> None:
    documents = [
        make_document("item1.txt", "Item 1", "Business overview."),
        make_document("item7.txt", "Item 7", "Revenue discussion."),
    ]

    context = format_context(documents)

    assert "Source: item1.txt (Item 1)\nBusiness overview." in context
    assert "Source: item7.txt (Item 7)\nRevenue discussion." in context


def test_unique_sources_deduplicates_while_preserving_order() -> None:
    documents = [
        make_document("item1.txt", "Item 1", "First chunk."),
        make_document("item1.txt", "Item 1", "Second chunk."),
        make_document("item7.txt", "Item 7", "Third chunk."),
    ]

    assert unique_sources(documents) == [
        ("item1.txt", "Item 1"),
        ("item7.txt", "Item 7"),
    ]
