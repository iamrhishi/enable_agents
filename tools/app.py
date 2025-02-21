from flask import Flask
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model

app = Flask(__name__)

# 1. Load: First we need to load our data. This is done with Document Loaders.
# 2. Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and passing it into a model, as large chunks are harder to search over and won't fit in a model's finite context window.
# 3. Store: We need somewhere to store and index our splits, so that they can be searched over later. This is often done using a VectorStore and Embeddings model.
# 4. Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
# 5. Generate: A ChatModel / LLM produces an answer using a prompt that includes both the question with the retrieved data

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def get_credentials():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def init_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm

def init_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def init_vector_store(embeddings):
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

def loader():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()
    return docs

def splitter(docs, vector_store):  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits= text_splitter.split_documents(docs)
    #index_chunks
    _ = vector_store.add_documents(documents = all_splits)

    prompt = hub.pull("rlm/rag-prompt")

# Define application steps
def retrieve(state: State):
    vector_store = init_vector_store(init_embeddings())
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    llm=init_llm();
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def graph_compile():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is task decomposition?"})
    return response


@app.route('/')
def rag_test():
    openai_key = get_credentials();
    embeddings = init_embeddings();
    vector_store = init_vector_store(embeddings);
    docs = loader();
    splitter(docs, vector_store);
    response = graph_compile();
    return response

if __name__ == '__main__':
    app.run(debug=True)
