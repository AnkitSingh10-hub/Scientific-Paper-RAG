from pydantic import BaseModel


class Result(BaseModel):
    """A homemade stand-in for langchain_core.documents.Document.

    Anything downstream (chunking, ingest, answer, eval) that used to expect
    a LangChain Document only ever touched `.page_content` and `.metadata`,
    so this is a drop-in replacement with no LangChain dependency.
    """

    page_content: str
    metadata: dict
