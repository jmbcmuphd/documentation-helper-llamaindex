from dotenv import load_dotenv
import os
import nltk
import pinecone
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("llamaindex-documentation-helper")
nltk.download("averaged_perceptron_tagger")

if __name__ == "__main__":
    print("Going to ingest pinecone documentation")
    # UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        file_extractor={".html": UnstructuredReader()},
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    # nodes = node_parser.get_nodes_from_documents(documents=documents)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("finished ingesting...")
