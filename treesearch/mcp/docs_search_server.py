import pickle
from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("Documentation search")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

VECTOR_STORES_BASE_PTH = Path("./ragEmbeddings")
VECTOR_STORE_NAMES = Literal["omnirec", "lenskit", "recbole"]


def _rebuild_vector_store(name: str) -> FAISS:
    """Rebuild a FAISS vector store from the existing docstore using local embeddings."""
    vector_store_pth = VECTOR_STORES_BASE_PTH / name
    pkl_path = vector_store_pth / "index.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # pkl format: (InMemoryDocstore, {int_index: docstore_id})
    docstore, index_to_docstore_id = data
    docs = []
    for idx in sorted(index_to_docstore_id.keys()):
        doc_id = index_to_docstore_id[idx]
        doc = docstore.search(doc_id)
        docs.append(doc)

    # Rebuild with local embeddings
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    vector_store.save_local(str(vector_store_pth))
    return vector_store


def _get_embedding_dim() -> int:
    """Get the dimension of the local embedding model."""
    return len(embedding_model.embed_query("test"))


EMBEDDING_DIM = _get_embedding_dim()


def load_vector_store(name: str) -> FAISS:
    vector_store_pth = VECTOR_STORES_BASE_PTH / name
    if not vector_store_pth.exists():
        raise FileNotFoundError(
            f"Could not read in store at '{vector_store_pth}'! Did you generate or download the embeddings first?"
        )
    store = FAISS.load_local(
        str(vector_store_pth),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    # If stored index has different dimensions, rebuild with local model
    if store.index.d != EMBEDDING_DIM:
        return _rebuild_vector_store(name)
    return store


VECTOR_STORES: dict[str, FAISS] = {
    name: load_vector_store(name) for name in get_args(VECTOR_STORE_NAMES)
}


@mcp.tool()
def documentation_query(library: VECTOR_STORE_NAMES, query: str, k: int = 4) -> str:
    """Queries the documentation and code of a given library

    Args:
        library (str): Target documentation store to search. Needs to be one of: ["omnirec", "lenskit", "recbole"]
        query (str): Natural-language search query.
        k (int, optional): Number of top matching documents to return. Defaults to 4.

    Returns:
        str: Top-k relevant documentation entries formatted in a string.
    """
    vector_store = VECTOR_STORES.get(library)
    if vector_store is None:
        return f"Invalid library! The library parameter needs to be one of: {VECTOR_STORE_NAMES}"

    results = vector_store.similarity_search_with_score(query, k)

    final_output = f"Found {len(results)} relevant documentation sections:\n\n"
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        # Convert distance to similarity score (lower distance = higher similarity)
        similarity = 1 / (1 + score)  # Simple conversion

        final_output += f"--- Result {i} (Relevance: {similarity:.2%}) ---\n"
        final_output += f"Source: {source}\n"
        final_output += f"Content:\n{doc.page_content}\n\n"

    return final_output


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
