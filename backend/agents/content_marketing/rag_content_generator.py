"""
RAG Content Generator Module
Generates marketing content using Retrieval-Augmented Generation
Leverages knowledge graphs and document embeddings
"""

from typing import List, Dict, Optional
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

from core.ai_client import get_langchain_llm, log_langchain_usage

load_dotenv()


class RAGContentGenerator:
    """
    Generates marketing content using RAG with knowledge graphs
    Supports multiple channels and content types
    """
    
    CHANNEL_CONFIGS = {
        'linkedin': {
            'max_length': 3000,
            'tone': 'professional',
            'style': 'thought leadership',
            'hashtags': 5,
            'cta': 'professional'
        },
        'email': {
            'max_length': 2000,
            'tone': 'persuasive',
            'style': 'narrative',
            'subject_line': True,
            'cta': 'conversions'
        },
        'social': {
            'max_length': 280,
            'tone': 'casual',
            'style': 'witty',
            'hashtags': 3,
            'cta': 'engagement'
        },
        'google_ads': {
            'max_length': 300,
            'tone': 'direct',
            'style': 'benefit-focused',
            'headline': True,
            'cta': 'clicks'
        }
    }
    
    CONTENT_TYPE_TEMPLATES = {
        'post': 'social media post',
        'article': 'long-form article',
        'ad': 'advertising copy',
        'email_campaign': 'email marketing campaign',
        'case_study': 'case study or success story',
        'whitepaper': 'technical whitepaper',
        'announcement': 'product or company announcement'
    }
    
    def __init__(self, user_id: Optional[str] = None, project_id: Optional[str] = None):
        self.user_id = user_id
        self.project_id = project_id
        self.embeddings = OpenAIEmbeddings()
        self.llm, self._key_source = get_langchain_llm(user_id, project_id, model="gpt-4", temperature=0.7)
        self.retriever = None
        self.vector_store = None
    
    def _setup_rag(self, documents: List[str]):
        """
        Setup RAG pipeline with document embeddings
        """
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        doc_chunks = []
        for i, doc in enumerate(documents):
            chunks = splitter.split_text(doc)
            for chunk in chunks:
                doc_chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": f"document_{i}"}
                ))
        
        if doc_chunks:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(doc_chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def _retrieve_context(self, query: str, num_docs: int = 3) -> List[str]:
        """
        Retrieve relevant documents using semantic search
        """
        if not self.retriever:
            return []
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            return [doc.page_content for doc in docs[:num_docs]]
        except:
            return []
    
    def _get_graph_context(self, 
                          knowledge_graph: Optional[Dict],
                          query: str) -> str:
        """
        Extract relevant context from knowledge graph
        """
        if not knowledge_graph:
            return ""
        
        context_parts = []
        
        # Add key entities
        if 'entities' in knowledge_graph:
            top_entities = knowledge_graph['entities'][:5]
            entities_text = ', '.join([e['entity'] for e in top_entities])
            context_parts.append(f"Key entities: {entities_text}")
        
        # Add key concepts
        if 'concepts' in knowledge_graph:
            top_concepts = knowledge_graph['concepts'][:3]
            concepts_text = ', '.join([c['concept'] for c in top_concepts])
            context_parts.append(f"Key concepts: {concepts_text}")
        
        # Add relationship summary
        if 'relationships' in knowledge_graph and knowledge_graph['relationships']:
            context_parts.append(f"Found {len(knowledge_graph['relationships'])} key relationships")
        
        return '\n'.join(context_parts)
    
    def generate(self,
                documents: List[str],
                knowledge_graph: Optional[Dict],
                channel: str = 'linkedin',
                content_type: str = 'post',
                domain_context: Optional[str] = None,
                user_context: str = '') -> Dict:
        """
        Generate marketing content for specified channel
        
        Args:
            documents: List of source documents
            knowledge_graph: Knowledge graph structure
            channel: Target channel (linkedin, email, social, google_ads)
            content_type: Type of content (post, article, ad, etc.)
            domain_context: Industry/domain information
            user_context: Additional user guidance
            
        Returns:
            Dictionary with generated content and metadata
        """
        # Setup RAG
        self._setup_rag(documents)
        
        # Get channel configuration
        channel_config = self.CHANNEL_CONFIGS.get(channel, self.CHANNEL_CONFIGS['linkedin'])
        content_description = self.CONTENT_TYPE_TEMPLATES.get(content_type, 'marketing content')
        
        # Build prompt
        query = f"Create {content_description} for {channel}"
        
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_context(query)
        context_text = '\n\n'.join(retrieved_docs)
        
        # Get knowledge graph context
        kg_context = self._get_graph_context(knowledge_graph, query)
        
        # Build content generation prompt
        prompt = self._build_generation_prompt(
            channel=channel,
            content_type=content_type,
            channel_config=channel_config,
            domain_context=domain_context,
            user_context=user_context
        )
        
        # Generate content
        chain = prompt | self.llm
        
        response = chain.invoke({
            "context": context_text[:2000],
            "kg_context": kg_context,
            "domain": domain_context or "General",
            "user_guidance": user_context or "Create engaging marketing content"
        })
        log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.rag_generate", "gpt-4", self._key_source)

        # Parse response
        content = response.content
        
        # Generate variations
        variations = self._generate_variations(
            base_content=content,
            channel=channel,
            content_type=content_type,
            num_variations=2
        )
        
        return {
            'content': content,
            'variations': variations,
            'metadata': {
                'channel': channel,
                'content_type': content_type,
                'domain': domain_context,
                'sources_used': len(retrieved_docs),
                'channel_config': channel_config
            }
        }
    
    def _build_generation_prompt(self,
                                 channel: str,
                                 content_type: str,
                                 channel_config: Dict,
                                 domain_context: Optional[str],
                                 user_context: str) -> ChatPromptTemplate:
        """
        Build channel-specific content generation prompt
        """
        base_instructions = f"""
You are an expert marketing copywriter specializing in {domain_context or 'B2B marketing'}.

Generate a {self.CONTENT_TYPE_TEMPLATES.get(content_type, 'marketing piece')} for {channel}.

Requirements:
- Tone: {channel_config.get('tone', 'professional')}
- Style: {channel_config.get('style', 'persuasive')}
- Maximum length: {channel_config['max_length']} characters
"""
        
        if channel_config.get('hashtags'):
            base_instructions += f"- Include {channel_config['hashtags']} relevant hashtags\n"
        
        if channel_config.get('subject_line'):
            base_instructions += "- Include an attention-grabbing subject line\n"
        
        if channel_config.get('headline'):
            base_instructions += "- Include a compelling headline (max 60 characters)\n"
        
        base_instructions += f"""
Source Context:
{{context}}

Knowledge Graph Context:
{{kg_context}}

Domain: {{domain}}

User Guidance: {{user_guidance}}

Generate compelling marketing content that:
1. Uses specific facts and claims from the source documents
2. Highlights key value propositions and benefits
3. Incorporates domain expertise and terminology
4. Is optimized for {channel} platform conventions
5. Includes a clear call-to-action appropriate for {channel}
6. Follows the specified tone and style

Output only the generated content, formatted appropriately for the channel.
"""
        
        return ChatPromptTemplate.from_template(base_instructions)
    
    def _generate_variations(self,
                           base_content: str,
                           channel: str,
                           content_type: str,
                           num_variations: int = 2) -> List[str]:
        """
        Generate alternative versions of the content
        """
        variations = []
        
        variation_prompts = [
            "Create a version that emphasizes the benefits more strongly",
            "Create a version with a different tone (more casual/formal)",
            "Create a version that leads with statistics or data points"
        ]
        
        channel_config = self.CHANNEL_CONFIGS.get(channel, {})
        
        for i in range(min(num_variations, len(variation_prompts))):
            prompt = ChatPromptTemplate.from_template(f"""
Based on this marketing content:

{base_content}

{variation_prompts[i]}

Keep the same channel ({channel}) and content type ({content_type}) requirements.
Maximum length: {channel_config.get('max_length', 2000)} characters.

Generate the variation:
""")
            
            chain = prompt | self.llm
            response = chain.invoke({})
            log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.rag_variation", "gpt-4", self._key_source)
            variations.append(response.content)
        
        return variations
    
    def chat_response(self,
                     user_message: str,
                     documents: List[str],
                     knowledge_graph: Optional[Dict]) -> str:
        """
        Conversational response for iterative content refinement
        
        Args:
            user_message: User's chat message
            documents: Available source documents
            knowledge_graph: Knowledge graph structure
            
        Returns:
            Agent response
        """
        # Setup RAG
        self._setup_rag(documents)
        
        # Retrieve relevant context
        retrieved = self._retrieve_context(user_message, num_docs=3)
        context = '\n\n'.join(retrieved)
        
        # Get KG context
        kg_context = self._get_graph_context(knowledge_graph, user_message)
        
        # Build conversation prompt
        prompt = ChatPromptTemplate.from_template("""
You are a helpful marketing content assistant. Help the user with their marketing and content questions.

Use the following context from their documents:
{context}

Knowledge Graph Context:
{kg_context}

User Message: {user_message}

Provide a helpful, specific response that:
1. References information from their documents when relevant
2. Offers actionable marketing advice
3. Is conversational and friendly
4. Suggests content improvements or ideas when appropriate
""")
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "context": context[:1500],
            "kg_context": kg_context,
            "user_message": user_message
        })
        log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.rag_chat", "gpt-4", self._key_source)

        return response.content
    
    def bulk_generate(self,
                     documents: List[str],
                     knowledge_graph: Optional[Dict],
                     channels: List[str],
                     domain_context: Optional[str] = None) -> Dict[str, Dict]:
        """
        Generate content for multiple channels at once
        """
        results = {}
        
        for channel in channels:
            try:
                result = self.generate(
                    documents=documents,
                    knowledge_graph=knowledge_graph,
                    channel=channel,
                    content_type='post',
                    domain_context=domain_context
                )
                results[channel] = result
            except Exception as e:
                results[channel] = {'error': str(e)}
        
        return results
