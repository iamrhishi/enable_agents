"""
Knowledge Graph Builder Module
Constructs semantic knowledge graphs from documents
Enables traversal for intelligent content generation
"""

from typing import List, Dict, Set, Tuple, Optional
import json
import re
import networkx as nx
from langchain.prompts import ChatPromptTemplate

from core.ai_client import get_langchain_llm, log_langchain_usage
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class KnowledgeGraphBuilder:
    """
    Builds semantic knowledge graphs from documents
    Extracts entities, relationships, and concepts
    """
    
    def __init__(self, user_id: Optional[str] = None, project_id: Optional[str] = None):
        self.user_id = user_id
        self.project_id = project_id
        self.llm, self._key_source = get_langchain_llm(user_id, project_id, model="gpt-4", temperature=0)
        self.graph = nx.DiGraph()
        self.entity_types = {}
        self.relationships = []
        self.concepts = []
    
    def build_graph(self, 
                   documents: List[str], 
                   domain_context: Dict = None) -> Dict:
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of document texts
            domain_context: Domain specialization info
            
        Returns:
            Dictionary with graph structure, entities, and relationships
        """
        # Combine documents
        combined_text = '\n\n'.join(documents)
        
        # Extract entities
        entities = self._extract_entities(combined_text, domain_context)
        
        # Extract relationships
        relationships = self._extract_relationships(
            combined_text, 
            entities,
            domain_context
        )
        
        # Extract key concepts
        concepts = self._extract_concepts(combined_text, domain_context)
        
        # Build graph structure
        graph_structure = self._build_graph_structure(
            entities, 
            relationships, 
            concepts
        )
        
        return {
            'entities': entities,
            'relationships': relationships,
            'concepts': concepts,
            'graph': graph_structure,
            'graph_stats': {
                'entity_count': len(entities),
                'relationship_count': len(relationships),
                'concept_count': len(concepts),
                'density': nx.density(self.graph) if len(self.graph) > 0 else 0
            }
        }
    
    def _extract_entities(self, text: str, domain_context: Dict = None) -> List[Dict]:
        """
        Extract named entities and key concepts from text
        """
        prompt = ChatPromptTemplate.from_template("""
        Extract all important entities from the following business text.
        Focus on: products, services, companies, people, roles, features, benefits.
        
        Domain context: {domain_context}
        
        Text:
        {text}
        
        Return a JSON array of entities with structure:
        [
            {{
                "entity": "entity name",
                "type": "PRODUCT|SERVICE|COMPANY|PERSON|ROLE|FEATURE|BENEFIT|PROCESS",
                "description": "brief description",
                "frequency": number of mentions,
                "synonyms": ["synonym1", "synonym2"]
            }}
        ]
        
        Only return valid JSON array, no additional text.
        """)
        
        chain = prompt | self.llm
        
        # Use first 3000 chars to avoid token limits
        truncated_text = text[:3000]
        
        response = chain.invoke({
            "text": truncated_text,
            "domain_context": json.dumps(domain_context) if domain_context else "General business"
        })
        log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.kg_extract_entities", "gpt-4", self._key_source)

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                
                # Add to graph
                for entity in entities:
                    self.graph.add_node(
                        entity['entity'],
                        type=entity.get('type', 'ENTITY'),
                        description=entity.get('description', ''),
                        frequency=entity.get('frequency', 1)
                    )
                    self.entity_types[entity['entity']] = entity.get('type', 'ENTITY')
                
                return entities
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _extract_relationships(self, 
                              text: str, 
                              entities: List[Dict],
                              domain_context: Dict = None) -> List[Dict]:
        """
        Extract relationships between entities
        """
        entity_names = [e['entity'] for e in entities]
        
        if not entity_names:
            return []
        
        entity_list_str = ', '.join(entity_names[:20])  # Limit to first 20
        
        prompt = ChatPromptTemplate.from_template("""
        Identify relationships between entities in the following text.
        
        Entities to find relationships between:
        {entities}
        
        Text:
        {text}
        
        Return a JSON array with structure:
        [
            {{
                "source": "entity1",
                "target": "entity2",
                "relationship_type": "OFFERS|SOLVES|BENEFITS|REQUIRES|INTEGRATES|COMPETES|REFERENCES",
                "description": "description of relationship",
                "strength": 0.1 to 1.0
            }}
        ]
        
        Only return valid JSON array.
        """)
        
        chain = prompt | self.llm
        truncated_text = text[:3000]
        
        response = chain.invoke({
            "entities": entity_list_str,
            "text": truncated_text
        })
        log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.kg_extract_relationships", "gpt-4", self._key_source)

        try:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                relationships = json.loads(json_match.group())
                
                # Add to graph
                for rel in relationships:
                    if rel['source'] in entity_names and rel['target'] in entity_names:
                        self.graph.add_edge(
                            rel['source'],
                            rel['target'],
                            relationship=rel.get('relationship_type', 'RELATED'),
                            description=rel.get('description', ''),
                            strength=rel.get('strength', 0.5)
                        )
                        self.relationships.append(rel)
                
                return relationships
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _extract_concepts(self, text: str, domain_context: Dict = None) -> List[Dict]:
        """
        Extract abstract concepts and themes
        """
        prompt = ChatPromptTemplate.from_template("""
        Extract key concepts, themes, and value propositions from this business text.
        
        Domain: {domain_context}
        
        Text:
        {text}
        
        Return JSON array:
        [
            {{
                "concept": "concept name",
                "category": "VALUE_PROPOSITION|PROBLEM|SOLUTION|FEATURE|BENEFIT|PROCESS",
                "description": "description",
                "related_entities": ["entity1", "entity2"]
            }}
        ]
        
        Only return valid JSON array.
        """)
        
        chain = prompt | self.llm
        truncated_text = text[:3000]
        
        response = chain.invoke({
            "text": truncated_text,
            "domain_context": json.dumps(domain_context) if domain_context else "General"
        })
        log_langchain_usage(response, self.user_id, self.project_id, "content_marketing.kg_extract_concepts", "gpt-4", self._key_source)

        try:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                concepts = json.loads(json_match.group())
                self.concepts = concepts
                return concepts
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _build_graph_structure(self, 
                               entities: List[Dict],
                               relationships: List[Dict],
                               concepts: List[Dict]) -> Dict:
        """
        Build the final graph structure for visualization and traversal
        """
        nodes = []
        edges = []
        
        # Add entity nodes
        for entity in entities:
            nodes.append({
                'id': entity['entity'],
                'label': entity['entity'],
                'type': 'entity',
                'entityType': entity.get('type', 'ENTITY'),
                'description': entity.get('description', ''),
                'size': min(30 + entity.get('frequency', 1) * 5, 100)
            })
        
        # Add concept nodes
        for concept in concepts:
            nodes.append({
                'id': concept['concept'],
                'label': concept['concept'],
                'type': 'concept',
                'category': concept.get('category', 'CONCEPT'),
                'size': 20
            })
        
        # Add relationship edges
        for rel in relationships:
            edges.append({
                'source': rel['source'],
                'target': rel['target'],
                'label': rel.get('relationship_type', 'RELATED'),
                'description': rel.get('description', ''),
                'strength': rel.get('strength', 0.5)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'directed': True,
                'multigraph': False
            }
        }
    
    def get_entity_context(self, entity: str, depth: int = 2) -> Dict:
        """
        Get context around an entity by traversing the graph
        
        Args:
            entity: Entity name
            depth: Traversal depth
            
        Returns:
            Context dictionary with related entities and relationships
        """
        if entity not in self.graph:
            return {}
        
        context = {
            'entity': entity,
            'entity_type': self.entity_types.get(entity, 'UNKNOWN'),
            'neighbors': {},
            'paths': []
        }
        
        # Get immediate neighbors
        neighbors = list(self.graph.neighbors(entity))
        predecessors = list(self.graph.predecessors(entity))
        
        context['neighbors']['outgoing'] = neighbors
        context['neighbors']['incoming'] = predecessors
        
        # Get paths to related entities
        for target in set(neighbors + predecessors):
            if target != entity:
                try:
                    paths = list(nx.all_simple_paths(self.graph, entity, target, cutoff=depth))
                    if paths:
                        context['paths'].append({
                            'target': target,
                            'paths': paths[:3]  # Limit to 3 shortest paths
                        })
                except nx.NetworkXNoPath:
                    pass
        
        return context
    
    def get_relevant_subgraph(self, 
                             seed_entities: List[str],
                             depth: int = 2) -> Dict:
        """
        Extract relevant subgraph for content generation
        
        Args:
            seed_entities: Starting entities
            depth: Traversal depth
            
        Returns:
            Subgraph structure
        """
        subgraph_nodes = set()
        
        for entity in seed_entities:
            if entity in self.graph:
                # Add entity itself
                subgraph_nodes.add(entity)
                
                # Add neighbors up to depth
                for _ in range(depth):
                    new_nodes = set()
                    for node in subgraph_nodes:
                        new_nodes.update(self.graph.neighbors(node))
                        new_nodes.update(self.graph.predecessors(node))
                    subgraph_nodes.update(new_nodes)
        
        if not subgraph_nodes:
            return {'nodes': [], 'edges': []}
        
        # Create subgraph
        sub = self.graph.subgraph(subgraph_nodes).copy()
        
        return {
            'nodes': list(sub.nodes(data=True)),
            'edges': list(sub.edges(data=True)),
            'node_count': len(sub.nodes()),
            'edge_count': len(sub.edges())
        }
    
    def find_entity_paths(self, 
                         source: str, 
                         target: str,
                         max_length: int = 5) -> List[List[str]]:
        """
        Find paths between two entities
        Useful for understanding relationships in content
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths[:10]  # Return top 10 paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Rank entities by importance using PageRank
        """
        try:
            if len(self.graph.nodes()) == 0:
                return []
            
            pagerank = nx.pagerank(self.graph)
            ranked = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            return ranked[:20]  # Top 20 entities
        except:
            return []
