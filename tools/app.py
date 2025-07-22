from flask import Flask,request, jsonify, redirect
import requests
import sqlite3
import shutil
import psutil
import tempfile
from datetime import datetime
from uuid import uuid4
from time import time
import json
import os
import bs4
import fitz 
import faiss
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
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
import glob
from flask_cors import CORS  # Import Flask-CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import openpyxl
from flask_cors import cross_origin
from urllib.parse import urlencode
from langchain import hub
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


load_dotenv()
LINKEDIN_CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID')
LINKEDIN_CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
LINKEDIN_REDIRECT_URI = os.getenv('LINKEDIN_REDIRECT_URI', 'http://localhost:5000/linkedin/callback')


nltk.download('stopwords')
nltk.download('punkt_tab')

PROMPTS_FILE = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/prompts.json"

app = Flask(__name__)
CORS(app)

# MySQL config
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@localhost/enable_agents'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(80))
    last_name = db.Column(db.String(80))
    email = db.Column(db.String(120))
    company = db.Column(db.String(120))

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

def get_chrome_history_path():
    """Get Chrome history file path based on OS"""
    if os.name == 'nt':  # Windows
        return os.path.expanduser('~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\History')
    elif 'darwin' in os.sys.platform.lower():  # macOS
        return os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/History')
    else:  # Linux
        return os.path.expanduser('~/.config/google-chrome/Default/History')

def identify_saas_tools_with_openai(history_data):
    """Use OpenAI to identify which URLs are web applications, tools, SaaS, PaaS, or productivity platforms"""
    try:
        # Extract URLs for analysis (limit to avoid token limits)
        urls_to_analyze = [item['url'] for item in history_data[:30]]  # Analyze top 30 URLs
        urls_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(urls_to_analyze)])
        
        prompt = f"""Analyze the following URLs and identify which ones are web applications, tools, or platforms. This includes:

- SaaS (Software as a Service) tools
- PaaS (Platform as a Service) platforms  
- Web-based productivity tools (Google Workspace, Microsoft 365, etc.)
- Cloud platforms and services
- Development tools and platforms
- Business applications and tools
- Design and creative tools
- Communication and collaboration platforms
- Analytics and monitoring tools
- Project management tools
- CRM and business software
- Educational and learning platforms
- Entertainment and media platforms (if they're tools/apps)

For each URL, return a JSON array with this structure:
[
  {{"url_index": 1, "is_tool": true, "tool_name": "Google Docs", "category": "Productivity", "type": "SaaS", "description": "Document creation and collaboration"}},
  {{"url_index": 2, "is_tool": true, "tool_name": "AWS Console", "category": "Cloud Platform", "type": "PaaS", "description": "Cloud computing services"}},
  {{"url_index": 3, "is_tool": false, "tool_name": null, "category": null, "type": null, "description": "Regular website"}}
]

URLs to analyze:
{urls_text}

Rules:
- Identify ANY web-based tool, application, or platform (not just traditional SaaS)
- Include Google Workspace (Docs, Sheets, Drive, Gmail), Microsoft 365, Slack, Zoom, etc.
- Include development platforms (GitHub, GitLab, Heroku, Vercel)
- Include cloud platforms (AWS, Azure, GCP)  
- Include design tools (Figma, Canva, Adobe Creative Cloud)
- Include business tools (Salesforce, HubSpot, Trello, Asana)
- Include social media platforms if used as business tools
- Categories: Development, Communication, Productivity, Design, Analytics, Cloud Platform, CRM, Project Management, Storage, Entertainment, Education, Social Media, Other
- Types: SaaS, PaaS, Web App, Platform, Tool
- Keep descriptions short (under 60 characters)
- Only mark as "false" if it's clearly a regular informational website
- Return valid JSON only"""

        client = openai.OpenAI()
        client.api_key = os.environ['OPENAI_API_KEY']
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a web application and tool identifier. You identify ALL types of web-based tools, applications, and platforms. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,  # Increased for more detailed analysis
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        try:
            tools_analysis = json.loads(response_text)
            
            # Create a mapping from URL index to tool info
            tools_mapping = {}
            for item in tools_analysis:
                if 'url_index' in item:
                    tools_mapping[item['url_index'] - 1] = {  # Convert to 0-based index
                        'is_tool': item.get('is_tool', False),
                        'tool_name': item.get('tool_name'),
                        'category': item.get('category'),
                        'type': item.get('type'),  # SaaS, PaaS, Web App, etc.
                        'description': item.get('description')
                    }
            
            return {
                'success': True,
                'mapping': tools_mapping
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for tools analysis: {str(e)}")
            return {
                'success': False,
                'error': f'JSON parsing failed: {str(e)}'
            }
            
    except Exception as e:
        print(f"OpenAI tools analysis error: {str(e)}")
        return {
            'success': False,
            'error': f'OpenAI error: {str(e)}'
        }

def read_chrome_history_safe():
    """Read Chrome browser history with comprehensive web tool identification"""
    try:
        history_path = get_chrome_history_path()
        
        if not os.path.exists(history_path):
            return {
                'success': False,
                'error': 'Chrome history file not found. Make sure Chrome is installed.'
            }
        
        # Try multiple approaches
        for attempt in range(3):
            try:
                # Method 1: Direct copy (works if Chrome is closed)
                temp_dir = tempfile.mkdtemp()
                temp_history = os.path.join(temp_dir, 'History')
                
                # Try to copy the file
                shutil.copy2(history_path, temp_history)
                
                # Connect to history database
                conn = sqlite3.connect(temp_history)
                cursor = conn.cursor()
                
                # Get top 60 latest URLs (we'll analyze 30 for tools)
                query = """
                SELECT 
                    url, 
                    title, 
                    visit_count,
                    last_visit_time,
                    datetime(last_visit_time/1000000 + (strftime('%s', '1601-01-01')), 'unixepoch', 'localtime') as visit_date
                FROM urls 
                ORDER BY last_visit_time DESC 
                LIMIT 60
                """
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                history_data = []
                for row in rows:
                    history_data.append({
                        'url': row[0],
                        'title': row[1] if row[1] else 'No Title',
                        'visit_count': row[2],
                        'last_visit_time': row[3],
                        'visit_date': row[4]
                    })
                
                conn.close()
                
                # Clean up temp files
                os.remove(temp_history)
                os.rmdir(temp_dir)
                
                # Analyze URLs with OpenAI to identify web tools and applications
                print("Analyzing URLs for web applications and tools...")
                tools_result = identify_saas_tools_with_openai(history_data)
                
                # Add tool information to history data
                if tools_result['success']:
                    tools_mapping = tools_result['mapping']
                    for i, item in enumerate(history_data):
                        if i in tools_mapping:
                            # Update with new field names
                            item.update({
                                'is_tool': tools_mapping[i]['is_tool'],
                                'tool_name': tools_mapping[i]['tool_name'],
                                'category': tools_mapping[i]['category'],
                                'tool_type': tools_mapping[i]['type'],  # SaaS, PaaS, Web App, etc.
                                'description': tools_mapping[i]['description'],
                                # Keep old fields for backward compatibility
                                'is_saas': tools_mapping[i]['is_tool']  
                            })
                        else:
                            # Default values for URLs not analyzed
                            item.update({
                                'is_tool': False,
                                'tool_name': None,
                                'category': None,
                                'tool_type': None,
                                'description': None,
                                'is_saas': False  # Backward compatibility
                            })
                else:
                    # If analysis fails, add default values
                    print(f"Tools analysis failed: {tools_result.get('error', 'Unknown error')}")
                    for item in history_data:
                        item.update({
                            'is_tool': None,  # null indicates analysis failed
                            'tool_name': None,
                            'category': None,
                            'tool_type': None,
                            'description': 'Tool analysis unavailable',
                            'is_saas': None  # Backward compatibility
                        })
                
                # Count tools found (using new field name)
                web_tools = [item for item in history_data if item.get('is_tool') == True]
                
                # Count by type
                tool_types = {}
                categories = {}
                for item in history_data:
                    if item.get('is_tool'):
                        tool_type = item.get('tool_type', 'Unknown')
                        category = item.get('category', 'Unknown')
                        tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
                        categories[category] = categories.get(category, 0) + 1
                
                return {
                    'success': True,
                    'data': history_data,
                    'total_urls': len(history_data),
                    'web_tools_found': len(web_tools),
                    'saas_tools_found': len(web_tools),  # Backward compatibility
                    'tool_types': tool_types,
                    'categories': categories,
                    'analysis_status': 'completed' if tools_result['success'] else 'failed',
                    'method': 'direct_copy'
                }
                
            except (PermissionError, sqlite3.OperationalError) as e:
                # Method 2: Try reading directly with WAL mode
                try:
                    conn = sqlite3.connect(f'file:{history_path}?mode=ro', uri=True)
                    cursor = conn.cursor()
                    
                    query = """
                    SELECT 
                        url, 
                        title, 
                        visit_count,
                        last_visit_time,
                        datetime(last_visit_time/1000000 + (strftime('%s', '1601-01-01')), 'unixepoch', 'localtime') as visit_date
                    FROM urls 
                    ORDER BY last_visit_time DESC 
                    LIMIT 30
                    """
                    
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    history_data = []
                    for row in rows:
                        history_data.append({
                            'url': row[0],
                            'title': row[1] if row[1] else 'No Title',
                            'visit_count': row[2],
                            'last_visit_time': row[3],
                            'visit_date': row[4]
                        })
                    
                    conn.close()
                    
                    # Analyze URLs with OpenAI to identify web tools
                    print("Analyzing URLs for web applications and tools...")
                    tools_result = identify_saas_tools_with_openai(history_data)
                    
                    # Add tool information to history data
                    if tools_result['success']:
                        tools_mapping = tools_result['mapping']
                        for i, item in enumerate(history_data):
                            if i in tools_mapping:
                                item.update({
                                    'is_tool': tools_mapping[i]['is_tool'],
                                    'tool_name': tools_mapping[i]['tool_name'],
                                    'category': tools_mapping[i]['category'],
                                    'tool_type': tools_mapping[i]['type'],
                                    'description': tools_mapping[i]['description'],
                                    'is_saas': tools_mapping[i]['is_tool']  # Backward compatibility
                                })
                            else:
                                item.update({
                                    'is_tool': False,
                                    'tool_name': None,
                                    'category': None,
                                    'tool_type': None,
                                    'description': None,
                                    'is_saas': False
                                })
                    else:
                        for item in history_data:
                            item.update({
                                'is_tool': None,
                                'tool_name': None,
                                'category': None,
                                'tool_type': None,
                                'description': 'Tool analysis unavailable',
                                'is_saas': None
                            })
                    
                    web_tools = [item for item in history_data if item.get('is_tool') == True]
                    
                    # Count by type and category
                    tool_types = {}
                    categories = {}
                    for item in history_data:
                        if item.get('is_tool'):
                            tool_type = item.get('tool_type', 'Unknown')
                            category = item.get('category', 'Unknown')
                            tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
                            categories[category] = categories.get(category, 0) + 1
                    
                    return {
                        'success': True,
                        'data': history_data,
                        'total_urls': len(history_data),
                        'web_tools_found': len(web_tools),
                        'saas_tools_found': len(web_tools),  # Backward compatibility
                        'tool_types': tool_types,
                        'categories': categories,
                        'analysis_status': 'completed' if tools_result['success'] else 'failed',
                        'method': 'readonly_direct'
                    }
                    
                except Exception as e2:
                    if attempt < 2:  # Not the last attempt
                        import time
                        time.sleep(1)  # Wait 1 second before retry
                        continue
                    else:
                        # Clean up if temp files were created
                        try:
                            if 'temp_history' in locals() and os.path.exists(temp_history):
                                os.remove(temp_history)
                            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                                os.rmdir(temp_dir)
                        except:
                            pass
                            
                        return {
                            'success': False,
                            'error': 'Chrome is currently running. Please close Chrome completely and try again.',
                            'detailed_error': f'Attempt {attempt + 1}: {str(e)} | {str(e2)}',
                            'suggestions': [
                                'Close Chrome completely (check system tray)',
                                'Wait a few seconds after closing Chrome',
                                'Make sure no Chrome processes are running in Task Manager'
                            ]
                        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error reading Chrome history: {str(e)}'
        }


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_dataframe(df):
    """Clean dataframe by handling merged cells and missing values"""
    # Replace NaN values with empty strings
    df = df.fillna('')
    
    # Clean column names (remove extra spaces, special characters)
    df.columns = df.columns.astype(str).str.strip()
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert all values to strings and strip whitespace
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        # Replace 'nan' string with empty string
        df[col] = df[col].replace('nan', '')
    
    return df

def csv_to_json(file_path):
    """Convert CSV file to JSON object"""
    try:
        # Read CSV with error handling for different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        # Convert to JSON
        json_data = df.to_dict('records')
        
        return {
            'success': True,
            'data': json_data,
            'total_records': len(json_data),
            'columns': list(df.columns),
            'message': f'Successfully converted {len(json_data)} records'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error processing CSV file: {str(e)}',
            'data': []
        }

def xlsx_to_json(file_path):
    """Convert XLSX file to JSON object"""
    try:
        # Read Excel file (first sheet by default)
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        # Convert to JSON
        json_data = df.to_dict('records')
        
        return {
            'success': True,
            'data': json_data,
            'total_records': len(json_data),
            'columns': list(df.columns),
            'message': f'Successfully converted {len(json_data)} records'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error processing XLSX file: {str(e)}',
            'data': []
        }
    
def xlsx_to_json_multiple_sheets(file_path):
    """Convert XLSX file with multiple sheets to JSON object"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheets_data = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            df = clean_dataframe(df)
            sheets_data[sheet_name] = df.to_dict('records')
        
        return {
            'success': True,
            'data': sheets_data,
            'sheets': list(excel_file.sheet_names),
            'message': f'Successfully converted {len(excel_file.sheet_names)} sheets'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error processing XLSX file: {str(e)}',
            'data': {}
        }
    
def extract_unique_companies(json_data):
    """Extract unique company names from JSON data"""
    companies = set()
    
    for record in json_data:
        # Check various possible company field names
        company_fields = ['company', 'Company', 'organization', 'Organization', 'employer', 'Employer']
        
        for field in company_fields:
            if field in record and record[field]:
                company_name = str(record[field]).strip()
                if company_name and company_name.lower() not in ['', 'n/a', 'na', 'none', 'null']:
                    companies.add(company_name)
                break
    
    return list(companies)

def get_company_skills_from_openai(company_list):
    """Send company list to OpenAI and get required skills for each company"""
    try:
        companies_text = ", ".join(company_list)
        
        # Simplified prompt with clearer instructions
        prompt = f"""For the following companies, provide required skills in JSON format:

Companies: {companies_text}

Return a JSON array with this exact structure:
[
  {{"company": "Company1", "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5", "skill6"]}},
  {{"company": "Company2", "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5", "skill6"]}}
]

Rules:
- Exactly 6 skills per company
- Mix of technical and soft skills
- Valid JSON only, no explanations
- Keep skills concise (1-3 words each)"""

        client = openai.OpenAI()
        client.api_key=os.environ['OPENAI_API_KEY']
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Return only valid JSON arrays. Keep responses concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Reduced token limit to prevent truncation
            temperature=0.1   # Lower temperature for more consistent output
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # More robust cleaning
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Remove any trailing incomplete content
        if response_text.endswith(','):
            response_text = response_text[:-1]
        if response_text.endswith('}'):
            response_text += ']'
        elif not response_text.endswith(']'):
            # Find the last complete object and truncate there
            last_complete = response_text.rfind('}')
            if last_complete > 0:
                response_text = response_text[:last_complete + 1] + ']'
        
        # Try to fix common JSON issues
        response_text = fix_json_issues(response_text)
        
        try:
            company_skills_data = json.loads(response_text)
            
            # Validate the structure
            if not isinstance(company_skills_data, list):
                raise ValueError("Response is not a list")
            
            for item in company_skills_data:
                if not isinstance(item, dict) or 'company' not in item or 'required_skills' not in item:
                    raise ValueError("Invalid item structure")
                if not isinstance(item['required_skills'], list):
                    raise ValueError("required_skills is not a list")
            
            return {
                'success': True,
                'data': company_skills_data
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing failed: {str(e)}")
            print(f"Raw response: {response_text[:500]}...")
            
            # Fallback: try to extract data manually
            fallback_result = extract_skills_manually(companies_text, response_text)
            if fallback_result['success']:
                return fallback_result
            
            return {
                'success': False,
                'error': f'Invalid JSON response: {str(e)}',
                'raw_response': response_text[:300]
            }
        
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return {
            'success': False,
            'error': f'OpenAI API error: {str(e)}'
        }
    

def fix_json_issues(json_text):
    """Fix common JSON formatting issues"""
    # Remove any text before the first [
    start_idx = json_text.find('[')
    if start_idx > 0:
        json_text = json_text[start_idx:]
    
    # Remove any text after the last ]
    end_idx = json_text.rfind(']')
    if end_idx > 0:
        json_text = json_text[:end_idx + 1]
    
    # Fix common escape issues
    json_text = json_text.replace('\n', ' ').replace('\r', ' ')
    json_text = json_text.replace('"', '"').replace('"', '"')  # Fix smart quotes
    
    return json_text

def extract_skills_manually(companies_text, response_text):
    """Fallback method to extract skills manually if JSON parsing fails"""
    try:
        import re
        
        company_list = [c.strip() for c in companies_text.split(',')]
        result_data = []
        
        # Default skills for different company types
        default_skills = {
            'tech': ['Python', 'JavaScript', 'Problem Solving', 'Communication', 'Teamwork', 'Leadership'],
            'finance': ['Excel', 'Financial Analysis', 'Risk Management', 'Communication', 'Attention to Detail', 'Leadership'],
            'healthcare': ['Patient Care', 'Medical Knowledge', 'Communication', 'Empathy', 'Attention to Detail', 'Teamwork'],
            'default': ['Communication', 'Problem Solving', 'Leadership', 'Teamwork', 'Analytical Thinking', 'Adaptability']
        }
        
        for company in company_list:
            # Try to determine company type and assign appropriate skills
            company_lower = company.lower()
            if any(tech_word in company_lower for tech_word in ['google', 'microsoft', 'apple', 'facebook', 'amazon', 'tech', 'software']):
                skills = default_skills['tech']
            elif any(fin_word in company_lower for fin_word in ['bank', 'finance', 'capital', 'investment', 'goldman', 'morgan']):
                skills = default_skills['finance']
            elif any(health_word in company_lower for health_word in ['hospital', 'medical', 'health', 'pharma', 'clinic']):
                skills = default_skills['healthcare']
            else:
                skills = default_skills['default']
            
            result_data.append({
                'company': company,
                'required_skills': skills
            })
        
        return {
            'success': True,
            'data': result_data
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Manual extraction failed: {str(e)}'
        }

def enrich_json_with_openai(json_data):
    """Enhanced function: Extract companies -> Get skills from OpenAI -> Enrich original data"""
    try:
        # Step 1: Extract unique company names from JSON data
        unique_companies = extract_unique_companies(json_data)
        
        if not unique_companies:
            return {
                'success': False,
                'error': 'No company names found in the data'
            }
        
        print(f"Found {len(unique_companies)} unique companies: {unique_companies}")
        
        # Step 2: Send company list to OpenAI to get required skills
        openai_result = get_company_skills_from_openai(unique_companies)
        
        if not openai_result['success']:
            return {
                'success': False,
                'error': f'Failed to get skills from OpenAI: {openai_result.get("error", "Unknown error")}'
            }
        
        # Step 3: Create company-skills mapping from OpenAI response
        company_skills_map = {}
        for item in openai_result['data']:
            if 'company' in item and 'required_skills' in item:
                company_skills_map[item['company']] = item['required_skills']
        
        print(f"Created skills mapping for {len(company_skills_map)} companies")
        
        # Step 4: Enrich original JSON data by matching company names
        enriched_data = []
        skills_added_count = 0
        
        for record in json_data:
            # Create a copy of the original record
            enriched_record = record.copy()
            
            # Find company name in this record
            company_name = None
            company_fields = ['company', 'Company', 'organization', 'Organization', 'employer', 'Employer']
            
            for field in company_fields:
                if field in record and record[field]:
                    company_name = str(record[field]).strip()
                    break
            
            # Add required_skills based on company match
            if company_name and company_name in company_skills_map:
                enriched_record['required_skills'] = company_skills_map[company_name]
                skills_added_count += 1
            else:
                enriched_record['required_skills'] = []
            
            enriched_data.append(enriched_record)
        
        return {
            'success': True,
            'data': enriched_data,
            'message': f'Successfully enriched {len(enriched_data)} profiles. Added skills to {skills_added_count} records based on {len(unique_companies)} companies.',
            'companies_processed': len(unique_companies),
            'records_enriched': skills_added_count
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Enrichment process error: {str(e)}'
        }

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
    embeddings_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/embeddings"
    os.makedirs(embeddings_folder, exist_ok=True)
    with open(os.path.join(embeddings_folder, f"{file_hash}_index.pkl"), "wb") as f:
        pickle.dump(index, f)
    with open(os.path.join(embeddings_folder, f"{file_hash}_phrase_embeddings.pkl"), "wb") as f:
        pickle.dump(phrase_embeddings, f)
    with open(os.path.join(embeddings_folder, f"{file_hash}_page_chunks.pkl"), "wb") as f:
        pickle.dump(page_chunks, f)

def load_embeddings(file_hash):
    embeddings_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/embeddings"
    with open(os.path.join(embeddings_folder, f"{file_hash}_index.pkl"), "rb") as f:
        index = pickle.load(f)
    with open(os.path.join(embeddings_folder, f"{file_hash}_phrase_embeddings.pkl"), "rb") as f:
        phrase_embeddings = pickle.load(f)
    with open(os.path.join(embeddings_folder, f"{file_hash}_page_chunks.pkl"), "rb") as f:
        page_chunks = pickle.load(f)
    print("Embeddings loaded successfully.")
    return index, phrase_embeddings, page_chunks

def init_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm

def init_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def init_vector_store(embeddings):
    return None

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
    number_of_characters = sum(len(text) for text in pdf_text.values())
    print(f"Total number of characters in PDF: {number_of_characters}")  # Debug print for total characters
    # Set chunk size and overlap based on the total number of characters
    # For example, if the total number of characters is less than 100,000, use smaller chunks
    # Adjust chunk size and overlap based on the total number of characters
    if number_of_characters < 100000:
        chunk_size = 1000
        chunk_overlap = 200
    elif number_of_characters < 500000:
        chunk_size = 800
        chunk_overlap = 200
    elif number_of_characters < 1000000:
        chunk_size = 600
        chunk_overlap = 200
    elif number_of_characters < 2000000:
        chunk_size = 500
        chunk_overlap = 200
    elif number_of_characters < 5000000:
        chunk_size = 400
        chunk_overlap = 200
    elif number_of_characters < 10000000:
        chunk_size = 300
        chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        phrases = rake.get_ranked_phrases()[:5]
        page_phrases[page] = phrases
    return page_phrases

def extract_keywords_from_chunks(page_chunks):
    rake = Rake()
    chunk_phrases = {}
    for page, chunks in page_chunks.items():
        for chunk_number, chunk in enumerate(chunks, start=1):
            rake.extract_keywords_from_text(chunk)
            phrases = rake.get_ranked_phrases()[:5]
            chunk_phrases[(page, chunk_number)] = phrases
    return chunk_phrases

def get_embeddings(phrase):
    client = openai.OpenAI()
    client.api_key = get_credentials()
    response = client.embeddings.create(model="text-embedding-ada-002", input=phrase)
    return response.data[0].embedding

def get_embeddings_batch(phrases):
    client = openai.OpenAI()
    client.api_key = get_credentials()
    response = client.embeddings.create(model="text-embedding-ada-002", input=phrases)
    return [data.embedding for data in response.data]

def store_embeddings(page_phrases, chunk_phrases):
    print(page_phrases)
    phrase_embeddings = {}
    # for (page, chunk_number), phrases in chunk_phrases.items():
    #     embeddings = [get_embeddings(phrase) for phrase in phrases]
    #     phrase_embeddings[(page, chunk_number)] = list(zip(phrases, embeddings))
    for (page, chunk_number), phrases in chunk_phrases.items():
        if phrases:
            embeddings = get_embeddings_batch(phrases)
            phrase_embeddings[(page, chunk_number)] = list(zip(phrases, embeddings))
        else:
            phrase_embeddings[(page, chunk_number)] = []

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
            # Fix: Use the correct key structure (page, chunk_number) instead of (file_hash, page, chunk_number)
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
    # response = client.chat.completions.create( 
    #     model="gpt-4", 
    #     messages=[ {"role": "system", "content": "You are a legal research and reasoning assistant trained in Indian income tax law, especially capital gains exemptions under the Income Tax Act. Your job is to analyze a user's scenario, determine applicability of specific sections (like Section 54F), and generate responses following a clear structure: Start with statutory interpretation — quote the relevant section (e.g., Section 54F) and clearly list the conditions in bullet points. Apply the law to the user’s case — mention whether conditions are satisfied and explain eligibility for exemption. Cite relevant case law in support of the position taken. Choose cases that match the factual scenario and jurisdiction where possible. Include citation (e.g., ITA 4012/Mum/2023 - Abdul Nayab Shaikh). Quote only favourable rulings unless otherwise requested. Prefer recent, relevant, and jurisdictionally appropriate cases. Discuss any common exceptions or judicial deviations — e.g., benefit being allowed even when more than one residential unit is purchased, especially if adjacent or used as a single unit. Quote examples from case law or factual scenarios to support the interpretation or exception. Keep the examples precise and relevant. Format your response in a professional, advisory tone suitable for a tax consultant’s opinion. Do not speculate — rely only on clear statutory provisions, circulars, and judicial precedents."}, {"role": "user", "content": prompt} ], 
    #     max_tokens=400, 
    #     temperature=0.1 ) 

    response = client.chat.completions.create( 
        model="gpt-4", 
        messages=[ {"role": "system", "content": "You are a professional skills extractor"}, {"role": "user", "content": prompt} ], 
        max_tokens=400, 
        temperature=0.1 ) 
    
    print(response)
    answer = response.choices[0].message.content 
    # usage = response.usage
    return answer

@app.route('/upload', methods=['POST'])
def upload_file():
    # Accept folder_name from form data (for file uploads)
    folder_name = request.form.get('folder_name', '').strip()

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Define the base upload directory
    base_upload_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/uploaded_data"

    # If folder_name is provided, create/use subfolder
    if folder_name:
        upload_folder = os.path.join(base_upload_folder, folder_name)
    else:
        upload_folder = base_upload_folder

    os.makedirs(upload_folder, exist_ok=True)  # Create the directory if it doesn't exist

    # Check if the file already exists
    file_path = os.path.join(upload_folder, file.filename)
    if os.path.exists(file_path):
        return jsonify({"message": "File already exists", "file_path": file_path}), 200

    # Save the file
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200


# @app.route('/rag_test', methods=['GET'])
# def rag_test():
#     query = request.args.get('query')
#     file_name = request.args.get('file_name')

#     upload_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data"
#     file_path = os.path.join(upload_folder, file_name)

#     # file_size = os.path.getsize(file_path)  # in bytes
    
#     if not os.path.exists(file_path):
#         return jsonify({"error": f"File '{file_name}' not found"}), 404

#     print(f"Absolute file path: {file_path}")
    
#     openai.api_key = get_credentials();

#     file_hash = get_file_hash(file_path)

#     if file_hash in cache:
#         print(f"Using cached embeddings for file hash: {file_hash}")
#         index, phrase_embeddings, page_chunks = load_embeddings(file_hash)
#     else:
#         print(f"Processing file and saving embeddings for file hash: {file_hash}")
#         pdf_doc = pdf_loader(file_path)
#         page_chunks = pdf_splitter(pdf_doc)

#         # print(page_chunks)

#         page_phrases = extract_keywords_from_pdf(pdf_doc)
#         chunk_phrases = extract_keywords_from_chunks(page_chunks)
        
#         index, phrase_embeddings = store_embeddings(page_phrases, chunk_phrases)
    
#         cache[file_hash] = (index, phrase_embeddings, page_chunks)
#         save_embeddings(file_hash, index, phrase_embeddings, page_chunks)
#         print(save_embeddings)
        
#     query_phrases = extract_phrases_from_query(query)
#     query_embeddings = get_embeddings_for_query(query_phrases)
#     selected_chunks = retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks)

#     max_chunks = 5
#     max_chunk_length = 1000  # characters
#     selected_chunks = [chunk[:max_chunk_length] for chunk in selected_chunks[:max_chunks]]


#     answer = generate(selected_chunks, query)


#     return jsonify({"answer": answer})  # Always return a JSON object

@app.route('/rag_test', methods=['POST'])
def rag_test():
    data = request.get_json()
    query = data.get('query')
    file_name = data.get('file_name')
    data_store = data.get('data_store')  # This is the folder name under uploaded_data

    base_upload_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/uploaded_data"
    if data_store:
        file_path = os.path.join(base_upload_folder, data_store, file_name)
    else:
        file_path = os.path.join(base_upload_folder, file_name)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File '{file_path}' not found"}), 404

    print(f"Absolute file path: {file_path}")
    openai.api_key = get_credentials()
    file_hash = get_file_hash(file_path)

    if file_hash in cache:
        print(f"Using cached embeddings for file hash: {file_hash}")
        index, phrase_embeddings, page_chunks = load_embeddings(file_hash)
    else:
        print(f"Processing file and saving embeddings for file hash: {file_hash}")
        pdf_doc = pdf_loader(file_path)
        page_chunks = pdf_splitter(pdf_doc)
        page_phrases = extract_keywords_from_pdf(pdf_doc)
        chunk_phrases = extract_keywords_from_chunks(page_chunks)
        index, phrase_embeddings = store_embeddings(page_phrases, chunk_phrases)
        cache[file_hash] = (index, phrase_embeddings, page_chunks)
        save_embeddings(file_hash, index, phrase_embeddings, page_chunks)

    query_phrases = extract_phrases_from_query(query)
    query_embeddings = get_embeddings_for_query(query_phrases)
    selected_chunks = retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks)

    max_chunks = 5
    max_chunk_length = 1000  # characters
    selected_chunks = [chunk[:max_chunk_length] for chunk in selected_chunks[:max_chunks]]

    answer = generate(selected_chunks, query)

    return jsonify({"answer": answer})  # Always return a JSON object

@app.route('/chat_api', methods=['POST', 'OPTIONS'])
@cross_origin()  # Allow CORS for this endpoint
def chat_api():
    import glob
    data = request.get_json()
    query = data.get('query')

    embeddings_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/embeddings"
    embedding_files = glob.glob(os.path.join(embeddings_folder, "*_index.pkl"))
    file_hashes = [os.path.basename(f).split('_')[0] for f in embedding_files]

    if not file_hashes:
        return jsonify({"error": "No knowledge base available. Please upload and process at least one file first."}), 400

    # Aggregate all embeddings, phrase mappings, and page chunks
    all_indexes = []
    all_phrase_embeddings = {}
    all_page_chunks = {}

    for file_hash in file_hashes:
        try:
            index, phrase_embeddings, page_chunks = load_embeddings(file_hash)
        except Exception as e:
            print(f"Error loading embeddings for {file_hash}: {e}")
            continue  # Skip files that can't be loaded

        all_indexes.append(index)
        # Update keys to include file_hash for uniqueness
        for (page, chunk_number), phrases in phrase_embeddings.items():
            all_phrase_embeddings[(file_hash, page, chunk_number)] = phrases
        for page, chunks in page_chunks.items():
            all_page_chunks[(file_hash, page)] = chunks

    # For simplicity, use the first index (FAISS) for searching, or you can merge indexes if needed
    index = all_indexes[0] if all_indexes else None
    phrase_embeddings = all_phrase_embeddings
    page_chunks = all_page_chunks

    if not index or not phrase_embeddings or not page_chunks:
        return jsonify({"error": "No valid embeddings found."}), 400

    query_phrases = extract_phrases_from_query(query)
    query_embeddings = get_embeddings_for_query(query_phrases)
    selected_chunks = retrieve_similar_chunks(query_embeddings, index, phrase_embeddings, page_chunks)

    max_chunks = 5
    max_chunk_length = 1000  # characters
    selected_chunks = [chunk[:max_chunk_length] for chunk in selected_chunks[:max_chunks]]

    answer = generate(selected_chunks, query)

    return jsonify({"answer": answer})

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

# @app.route('/yfinance', methods=['GET'])
# def yfinance_test():
#     symbol = request.args.get('stock')
#     region = request.args.get('region')

#     if not symbol or not region:
#         return "Missing required parameters: 'stock' and 'region'", 400

#     conn = http.client.HTTPSConnection("yahoo-finance166.p.rapidapi.com")

#     headers = {
#         'x-rapidapi-key': "95cdd43379mshbd9483856442c47p1c2782jsn897449ebefb8",
#         'x-rapidapi-host': "yahoo-finance166.p.rapidapi.com"
#     }

#     endpoint = f"/api/stock/get-financial-data?region={region}&symbol={symbol}"
#     print(f"Requesting data from endpoint: {endpoint}")  # Debug statement
#     conn.request("GET", endpoint, headers=headers)

#     res = conn.getresponse()
#     data = res.read()
#     json_data = json.loads(data.decode("utf-8"))

#     print(json_data)  # Debug statement to print the entire response

#     if 'quoteSummary' not in json_data or 'result' not in json_data['quoteSummary'] or not json_data['quoteSummary']['result']:
#         return jsonify({"error": "No data found for the given stock symbol and region"}), 404

#     current_price = json_data['quoteSummary']['result'][0]['financialData']['currentPrice']['fmt']
#     operating_margins = json_data['quoteSummary']['result'][0]['financialData']['operatingMargins']['fmt']
#     netprofit_margins = json_data['quoteSummary']['result'][0]['financialData']['profitMargins']['fmt']
#     gross_margins = json_data['quoteSummary']['result'][0]['financialData']['grossMargins']['fmt']
#     revenue_growth = json_data['quoteSummary']['result'][0]['financialData']['revenueGrowth']['fmt']
#     debt_to_equity = json_data['quoteSummary']['result'][0]['financialData']['debtToEquity']['fmt']
#     quick_ratio = json_data['quoteSummary']['result'][0]['financialData']['quickRatio']['fmt']
#     current_ratio = json_data['quoteSummary']['result'][0]['financialData']['currentRatio']['fmt']
#     analyst_recommendation = json_data['quoteSummary']['result'][0]['financialData']['recommendationKey']
#     number_of_analysts = json_data['quoteSummary']['result'][0]['financialData']['numberOfAnalystOpinions']['fmt']
#     target_high_price = json_data['quoteSummary']['result'][0]['financialData']['targetHighPrice']['fmt']
#     target_low_price = json_data['quoteSummary']['result'][0]['financialData']['targetLowPrice']['fmt']
#     target_mean_price = json_data['quoteSummary']['result'][0]['financialData']['targetMeanPrice']['fmt']
#     target_median_price = json_data['quoteSummary']['result'][0]['financialData']['targetMedianPrice']['fmt']

#     financial_KPIs = {
#         "current_price": current_price,
#         "operating margin": operating_margins,
#         "netprofit_margins": netprofit_margins,
#         "gross_margins": gross_margins,
#         "revenue_growth": revenue_growth,
#         "debt_to_equity": debt_to_equity,
#         "quick_ratio": quick_ratio,
#         "current_ratio": current_ratio,
#         "number_of_analysts": number_of_analysts,
#         "analyst_recommendation": analyst_recommendation,
#         "target_high_price": target_high_price,
#         "target_low_price": target_low_price,
#         "target_mean_price": target_mean_price,
#         "target_median_price": target_median_price
#     }

#     return jsonify(financial_KPIs)

@app.route('/yfinance', methods=['POST'])
def yfinance_test():
    data = request.get_json()
    symbol = data.get('stock')
    region = data.get('region')

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
    

    # prompt = f"""
    # Draft requirements based on the requirements {overview} that are specific, measurable, achievable, relevant, and time-bound (SMART).
    # Consider {context} as context for the requirements being asked for, focus on the market in {country} or region, 
    # consider {industries} for industry related insights, consider {function} as the role or business function of the requester,
    # and without mentioning the framework in the final response, conduct research taking into account these analysis frameworks: {frameworks} for one valuable and rare resource each using the VRIO, market forces for and against the startup using PESTLE, and product readiness using Mckinsey's 3 Horizon and use response format as reference: {format}.
    # """

    prompt = f"""
    Draft an research and analysis report based on the requirements {overview} .
    Consider {context} as context for the research being asked for, focus on the market in {country} or region, 
    consider {industries} for industry related insights, consider {function} as the role or business function of the requester,
    and without mentioning the framework in the final response, conduct research taking into account these analysis frameworks: {frameworks} for one valuable and rare resource each using the VRIO, market forces for and against the startup using PESTLE, and product readiness using Mckinsey's 3 Horizon and use response format as reference: {format}.
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
            max_tokens=800,
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

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    email = data.get('email')
    company = data.get('company')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    try:
        hashed_password = generate_password_hash(password)
        user = User(
            username=username,
            password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            email=email,
            company=company
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        print("Registration error:", e)  # Add this line
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        # You can return user info or a token here if you want
        return jsonify({'message': 'Login successful', 'username': user.username}), 200
    else:
        return jsonify({'error': 'Invalid username or password'}), 401
    

@app.route('/file_to_json_convert', methods=['POST'])
def convert_file():
    """Main endpoint to convert CSV/XLSX files to JSON"""
    
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided',
            'data': []
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected',
            'data': []
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'File type not allowed. Please upload CSV or XLSX files only.',
            'data': []
        }), 400
    
    try:
        # Create temporary upload folder for conversion
        temp_folder = "/Users/rhishikeshthakur/Enable/Software Development/enable_agents/data/temp_conversion"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_folder, filename)
        file.save(file_path)
        
        # Get conversion options
        multiple_sheets = request.form.get('multiple_sheets', 'false').lower() == 'true'
        
        # Convert based on file type
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'csv':
            result = csv_to_json(file_path)
        elif file_extension in ['xlsx', 'xls']:
            if multiple_sheets:
                result = xlsx_to_json_multiple_sheets(file_path)
            else:
                result = xlsx_to_json(file_path)
        
        # Clean up - remove uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up file if error occurs
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'data': []
        }), 500
    

@app.route('/enrich_with_openai', methods=['POST'])
@cross_origin()
def enrich_with_openai():
    """API endpoint to enrich JSON data with required skills using OpenAI"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({
                'success': False,
                'error': 'No data provided in request body'
            }), 400
        
        json_data = request_data['data']
        
        if not isinstance(json_data, list) or len(json_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Data must be a non-empty list of objects'
            }), 400
        
        # Enrich data using the new workflow
        result = enrich_json_with_openai(json_data)
        print(result)
        # return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
    

@app.route('/chrome_history', methods=['GET'])
def get_chrome_history():
    """API endpoint to get Chrome browser history with better error handling"""
    try:
        result = read_chrome_history_safe()
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/chrome_status', methods=['GET'])
def check_chrome_status():
    """Check if Chrome is running"""
    try:
        
        chrome_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    chrome_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return jsonify({
            'success': True,
            'chrome_running': len(chrome_processes) > 0,
            'processes': chrome_processes
        })
        
    except ImportError:
        return jsonify({
            'success': False,
            'error': 'psutil not installed. Cannot check Chrome status.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
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
    with app.app_context():
        db.create_all()  # Creates tables if not exist
    app.run(debug=True)
