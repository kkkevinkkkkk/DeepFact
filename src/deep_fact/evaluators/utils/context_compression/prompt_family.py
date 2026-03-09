from langchain.schema import Document


class PromptFamily:
    """Minimal prompt family interface used by context compression."""

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join(
            f"Source: {doc.metadata.get('source')}\n"
            f"Title: {doc.metadata.get('title')}\n"
            f"Content: {doc.page_content}\n"
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        )
