from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (
    Settings,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
import pinecone
import os
import streamlit as st

from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from node_postprocessors.duplicate_postprocessor import (
    DuplicateRemoverNodePostprocessor,
)

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    print("RAG...")

    pinecone_index = pc.Index("llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()
if "chat_engine" not in st.session_state:
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=Settings.embed_model, percentile_cutoff=0.5, threshold_cutoff=0.8
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_manager=ChatMode.CONTEXT,
        verbose=True,
        node_postprocessor=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

st.set_page_config(
    page_title="Chat with LlamaIndex Docs, powered by LlamaIndex",
    page_icon="🦙 ",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with LlamaIndex Docs 💬🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "context": "Ask me a question about LlamaIndex's open source python Library?",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "context": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["context"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node {i+1}: score= {node.score}")
                    st.write(node.text)
            message = {"role": "assistant", "context": response.response}
            st.session_state.messages.append(message)
