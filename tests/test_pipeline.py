from vector_store import debug_retrieval
import vector_store


def test_retrieval_debug():
    debug_retrieval(vector_store, "refund policy")
