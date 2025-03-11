from flask import Flask,request
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
import fitz 
from langchain_community.document_loaders import PyPDFLoader
import faiss
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from rake_nltk import Rake
import numpy as np
from scipy.spatial.distance import cosine
import openai
import nltk
import pandas as pd
import pickle
import hashlib


nltk.download('stopwords')
nltk.download('punkt_tab')

app = Flask(__name__)

# 1. Load: First we need to load our data. This is done with Document Loaders.
# 2. Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and passing it into a model, as large chunks are harder to search over and won't fit in a model's finite context window.
# 3. Store: We need somewhere to store and index our splits, so that they can be searched over later. This is often done using a VectorStore and Embeddings model.
# 4. Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
# 5. Generate: A ChatModel / LLM produces an answer using a prompt that includes both the question with the retrieved data

cache = {}

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def get_credentials():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_embeddings(file_hash, index, phrase_embeddings, page_chunks):
    with open(f"{file_hash}_index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(f"{file_hash}_phrase_embeddings.pkl", "wb") as f:
        pickle.dump(phrase_embeddings, f)
    with open(f"{file_hash}_page_chunks.pkl", "wb") as f:
        pickle.dump(page_chunks, f)

def load_embeddings(file_hash):
    with open(f"{file_hash}_index.pkl", "rb") as f:
        index = pickle.load(f)
    with open(f"{file_hash}_phrase_embeddings.pkl", "rb") as f:
        phrase_embeddings = pickle.load(f)
    with open(f"{file_hash}_page_chunks.pkl", "rb") as f:
        page_chunks = pickle.load(f)
    return index, phrase_embeddings, page_chunks

def init_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm

def init_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def init_vector_store(embeddings):
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

def pdf_loader(file_path):
    pdf_document = fitz.open(file_path)
    pdf_text ={}
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pdf_text[page_number + 1] = page.get_text()
    pdf_document.close()
    return pdf_text

def web_loader():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()
    return docs

def pdf_splitter(pdf_text):  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=300)
    page_chunks = {}
    for page, text in pdf_text.items():
        # print(f"Page {page} length: {len(text)}")  # Debug print for text length
        chunks = text_splitter.split_text(text)
        # print(f"Page {page} chunks: {len(chunks)}")  # Debug print for number of chunks
        page_chunks[page] = chunks
    return page_chunks

def extract_keywords_from_pdf(pdf_text):
    rake = Rake()
    page_phrases = {}
    for page, text in pdf_text.items():
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        page_phrases[page] = phrases
    return page_phrases

def extract_keywords_from_chunks(page_chunks):
    rake = Rake()
    chunk_phrases = {}
    for page, chunks in page_chunks.items():
        for chunk_number, chunk in enumerate(chunks, start=1):
            rake.extract_keywords_from_text(chunk)
            phrases = rake.get_ranked_phrases()
            chunk_phrases[(page, chunk_number)] = phrases
    return chunk_phrases

def get_embeddings(phrase):
    client = openai.OpenAI()
    client.api_key = get_credentials()
    response = client.embeddings.create(model="text-embedding-ada-002", input=phrase)
    return response.data[0].embedding

def store_embeddings(page_phrases, chunk_phrases):
    phrase_embeddings = {}
    for (page, chunk_number), phrases in chunk_phrases.items():
        embeddings = [get_embeddings(phrase) for phrase in phrases]
        phrase_embeddings[(page, chunk_number)] = list(zip(phrases, embeddings))

    # Initialise FAISS index
    dimension = len(phrase_embeddings[(1, 1)][0][1])
    index = faiss.IndexFlatIP(dimension)
    # Add all embeddings to the index
    for (page, chunk_number), phrases in phrase_embeddings.items():
       for phrase, embedding in phrases:
           index.add(np.array([embedding], dtype=np.float32))

    return index, phrase_embeddings

def extract_phrases_from_query(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    return rake.get_ranked_phrases()

def get_embeddings_for_query(phrases):
    client = openai.OpenAI()
    client.api_key = get_credentials()
    return [client.embeddings.create(model="text-embedding-ada-002", input=phrase).data[0].embedding for phrase in phrases]

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def store_cosine_similarities(query_embeddings, phrase_embeddings, page_chunks):
    chunk_similarities = {}
    for (page, chunk_number), phrases in phrase_embeddings.items():
        similarities = []
        for phrase, embedding in phrases:
            phrase_similarities = [get_cosine_similarity(embedding, query_embedding) for query_embedding in query_embeddings] 
        similarities.append(max(phrase_similarities)) 
        # Choose the highest similarity for each phrase 
        average_similarity = np.mean(similarities) 
        # Average similarity for the chunk 
        chunk_similarities[(page, chunk_number)] = average_similarity 
    # Get top 5 chunks by similarity 
    top_chunks = sorted(chunk_similarities.items(), key=lambda x: x[1], reverse=True)[:5] 
    # Output top 5 chunks 
    print("Top 5 most relatable chunks:") 
    selected_chunks = []
    for (page, chunk_number), similarity in top_chunks: 
        print(f"Page: {page}, Chunk: {chunk_number}, Similarity: {similarity}") 
        print(f"Chunk text:\n{page_chunks[page][chunk_number-1]}\n")
        selected_chunks.append(page_chunks[page][chunk_number-1])
    return selected_chunks

def retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks):
    query_embeddings_np = np.array(query_embeddings, dtype=np.float32)
    D, I = index.search(query_embeddings_np, k=5)  # Retrieve top 5 similar chunks

    selected_chunks = []
    for i in range(len(I)):
        for j in range(len(I[i])):
            chunk_id = int(I[i][j])
            for (page, chunk_number), phrases in phrase_embeddings.items():
                for phrase, embedding in phrases:
                    if np.array_equal(embedding, index.reconstruct(chunk_id)):
                        selected_chunks.append(page_chunks[page][chunk_number-1])
                        break

    return selected_chunks

def generate(selected_chunks, query):
    client = openai.OpenAI()
    context = "\n\n".join(selected_chunks) 
    prompt = f"Answer the following query based on the provided text:\n\n{context}\n\nQuery: {query}\nAnswer:" 
    response = client.chat.completions.create( 
        model="gpt-4", 
        messages=[ {"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt} ], 
        max_tokens=300, 
        temperature=0.1 ) 
    answer = response.choices[0].message.content 
    return answer

@app.route('/rag_test', methods=['GET'])
def rag_test():
    query = request.args.get('query')
    file_name = request.args.get('file_name')
    
    openai.api_key = get_credentials();

    file_hash = get_file_hash(file_name)
    
    if file_hash in cache:
        index, phrase_embeddings, page_chunks = load_embeddings(file_hash)
    else:
        pdf_doc = pdf_loader(file_name)
        page_chunks = pdf_splitter(pdf_doc)
   
        page_phrases = extract_keywords_from_pdf(pdf_doc)
        chunk_phrases = extract_keywords_from_chunks(page_chunks)
        index, phrase_embeddings = store_embeddings(page_phrases, chunk_phrases)
        cache[file_hash] = (index, phrase_embeddings, page_chunks)
        save_embeddings(file_hash, index, phrase_embeddings, page_chunks)
        
    query_phrases = extract_phrases_from_query(query)
    query_embeddings = get_embeddings_for_query(query_phrases)
    selected_chunks = retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks)

    answer = generate(selected_chunks, query)
    return answer

if __name__ == '__main__':
    app.run(debug=True)
