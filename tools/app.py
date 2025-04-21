from flask import Flask,request, jsonify
from datetime import datetime
from uuid import uuid4
import json
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
import http.client
import json
from flask_cors import CORS  # Import Flask-CORS

nltk.download('stopwords')
nltk.download('punkt_tab')

PROMPTS_FILE = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/prompts.json"

app = Flask(__name__)
CORS(app)

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
    print(page_phrases)
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

        # print(page_chunks)

        page_phrases = extract_keywords_from_pdf(pdf_doc)
        chunk_phrases = extract_keywords_from_chunks(page_chunks)
        
        index, phrase_embeddings = store_embeddings(page_phrases, chunk_phrases)
    
        cache[file_hash] = (index, phrase_embeddings, page_chunks)
        save_embeddings(file_hash, index, phrase_embeddings, page_chunks)
        print(save_embeddings)
        
    query_phrases = extract_phrases_from_query(query)
    query_embeddings = get_embeddings_for_query(query_phrases)
    selected_chunks = retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks)

    answer = generate(selected_chunks, query)
    return answer

@app.route('/save-prompt', methods=['POST'])
def save_prompt():
    data = request.get_json()
    prompt_id = str(uuid4())  # Generate a unique ID for the prompt
    data['id'] = prompt_id
    data['timestamp'] = datetime.now().isoformat()

    # Load existing prompts
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as file:
            prompts = json.load(file)
    else:
        prompts = []

    # Add the new prompt
    prompts.append(data)

    # Save back to the file
    with open(PROMPTS_FILE, 'w') as file:
        json.dump(prompts, file, indent=4)

    return jsonify({"message": "Prompt saved successfully", "id": prompt_id})

@app.route('/previous-prompts', methods=['GET'])
def previous_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as file:
            prompts = json.load(file)
    else:
        prompts = []

    return jsonify({"prompts": prompts})

@app.route('/yfinance', methods=['GET'])
def yfinance_test():
    symbol = request.args.get('stock')
    region = request.args.get('region')

    if not symbol or not region:
        return "Missing required parameters: 'stock' and 'region'", 400

    conn = http.client.HTTPSConnection("yahoo-finance166.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "95cdd43379mshbd9483856442c47p1c2782jsn897449ebefb8",
        'x-rapidapi-host': "yahoo-finance166.p.rapidapi.com"
    }

    endpoint = f"/api/stock/get-financial-data?region={region}&symbol={symbol}"
    print(f"Requesting data from endpoint: {endpoint}")  # Debug statement
    conn.request("GET", endpoint, headers=headers)

    res = conn.getresponse()
    data = res.read()
    json_data = json.loads(data.decode("utf-8"))

    print(json_data)  # Debug statement to print the entire response

    if 'quoteSummary' not in json_data or 'result' not in json_data['quoteSummary'] or not json_data['quoteSummary']['result']:
        return jsonify({"error": "No data found for the given stock symbol and region"}), 404

    current_price = json_data['quoteSummary']['result'][0]['financialData']['currentPrice']['fmt']
    operating_margins = json_data['quoteSummary']['result'][0]['financialData']['operatingMargins']['fmt']
    netprofit_margins = json_data['quoteSummary']['result'][0]['financialData']['profitMargins']['fmt']
    gross_margins = json_data['quoteSummary']['result'][0]['financialData']['grossMargins']['fmt']
    revenue_growth = json_data['quoteSummary']['result'][0]['financialData']['revenueGrowth']['fmt']
    debt_to_equity = json_data['quoteSummary']['result'][0]['financialData']['debtToEquity']['fmt']
    quick_ratio = json_data['quoteSummary']['result'][0]['financialData']['quickRatio']['fmt']
    current_ratio = json_data['quoteSummary']['result'][0]['financialData']['currentRatio']['fmt']
    analyst_recommendation = json_data['quoteSummary']['result'][0]['financialData']['recommendationKey']
    number_of_analysts = json_data['quoteSummary']['result'][0]['financialData']['numberOfAnalystOpinions']['fmt']
    target_high_price = json_data['quoteSummary']['result'][0]['financialData']['targetHighPrice']['fmt']
    target_low_price = json_data['quoteSummary']['result'][0]['financialData']['targetLowPrice']['fmt']
    target_mean_price = json_data['quoteSummary']['result'][0]['financialData']['targetMeanPrice']['fmt']
    target_median_price = json_data['quoteSummary']['result'][0]['financialData']['targetMedianPrice']['fmt']

    financial_KPIs = {
        "current_price": current_price,
        "operating margin": operating_margins,
        "netprofit_margins": netprofit_margins,
        "gross_margins": gross_margins,
        "revenue_growth": revenue_growth,
        "debt_to_equity": debt_to_equity,
        "quick_ratio": quick_ratio,
        "current_ratio": current_ratio,
        "number_of_analysts": number_of_analysts,
        "analyst_recommendation": analyst_recommendation,
        "target_high_price": target_high_price,
        "target_low_price": target_low_price,
        "target_mean_price": target_mean_price,
        "target_median_price": target_median_price
    }

    return jsonify(financial_KPIs)

@app.route('/generate-requirements', methods=['POST'])
def generate_requirements():
    openai.api_key = get_credentials()

    data = request.get_json()
    overview = data.get('overview', '')
    context = data.get('context', '')  # Get the context from the payload
    country = data.get('countries', '')
    industries = data.get('industries', '')
    function = data.get('businessFunction', '')
    frameworks = data.get('frameworks', [])

    format = data.get('responseFormat', '')
    

    prompt = f"""
    Here is an overview of business requirements: {overview}.
    Consider {context} as context, {country} for region specific insights, 
    {industries} for industry focus, {function} for business function and role of the requester,
    research based on these analysis frameworks: {frameworks} for one valuable and rare resource each using the VRIO, market forces for and against the startup using PESTLE, and product readiness using Mckinsey's 3 Horizon and use response format as reference: {format}.
    """

    print(prompt)

    client = openai.OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.6
        )

        answer = response.choices[0].message.content
        return jsonify({"requirements": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-requirements', methods=['POST'])
def save_requirements():
    data = request.get_json()
    requirements = data.get('requirements', [])
    export_option = data.get('exportOption', 'Unknown')  # Get the export option

    if not requirements:
        return jsonify({"error": "No requirements to save"}), 400

    # Define the folder path
    folder_path = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/requirements_versions"
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Create a unique file name with the export option and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"requirements_{export_option}_{timestamp}.txt")

    # Save the requirements to the file
    with open(file_path, "w") as file:
        file.write("\n".join(requirements))

    return jsonify({"message": f"Requirements saved successfully via {export_option}", "file_path": file_path})

# @app.route('/AI_ML', methods=['GET'])
# def yfinance_test()
    
#     return "Hello World!"


# @app.route('/Location', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Transportation', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Business- Enterprise', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Visual Recognition', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Small Tools', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Text Analysis', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Weather', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Messaging', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Logistics', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/News', methods=['GET'])
# def yfinance_test():
    
#     return "Hello World!"

# @app.route('/Jobs', methods=['GET'])
# def yfinance_test():
    
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
