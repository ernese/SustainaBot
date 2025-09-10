"""
Stage 0: Context Retrieval Component for SustainaBot
Implements RAG (Retrieval-Augmented Generation) and vector search capabilities
for enhanced Philippine data analysis and semantic discovery.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
from datetime import datetime
import pickle

# Vector and embedding libraries
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from llm.llm_client import get_llm_client, get_model_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhilippineContextRetrieval:
    """
    Advanced context retrieval system that combines vector search, RAG, and 
    Philippine domain knowledge for intelligent data discovery and analysis enhancement.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 data_path: Optional[str] = None,
                 vector_db_path: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the context retrieval system."""
        
        # Accept common environment variable names for API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        # Resolve data path: runtime arg > DATA_PATH env > repo data folder
        default_repo_data = Path(__file__).resolve().parents[2] / "data"
        self.data_path = Path(data_path or os.getenv("DATA_PATH", str(default_repo_data)))
        self.vector_db_path = vector_db_path or str(self.data_path / "vector_store")
        self.embedding_model_name = embedding_model
        
        # Initialize OpenAI client
        if self.api_key:
            self.client = get_llm_client(api_key=self.api_key)
        else:
            logger.warning("OpenAI API key not provided. Some features may be limited.")
            self.client = None
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize vector database
        self._initialize_vector_db()
        
        # Load Philippine knowledge base
        self._load_philippine_knowledge()
        
        # Initialize context cache
        self.context_cache = {}
        
        logger.info("Philippine Context Retrieval system initialized")
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            self.embedding_model = None
            return
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database."""
        
        if not CHROMADB_AVAILABLE:
            logger.error("chromadb not available. Install with: pip install chromadb")
            self.vector_db = None
            return
        
        try:
            # Create vector store directory
            Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
            
            # Create collections for different types of context
            self.collections = {
                "tables": self._get_or_create_collection("philippine_tables", 
                                                       "Table metadata and descriptions"),
                "columns": self._get_or_create_collection("philippine_columns",
                                                        "Column descriptions and data types"),
                "domain_knowledge": self._get_or_create_collection("philippine_domain",
                                                                 "Philippine domain expertise"),
                "query_history": self._get_or_create_collection("query_patterns",
                                                              "Historical query patterns")
            }
            
            logger.info("Vector database initialized with collections")
            self.vector_db = True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            self.vector_db = None
    
    def _get_or_create_collection(self, name: str, description: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.chroma_client.get_collection(name=name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"description": description}
            )
    
    def _load_philippine_knowledge(self):
        """Load Philippine-specific domain knowledge."""
        
        # Philippine regional hierarchy and mappings
        self.philippine_knowledge = {
            "regions": {
                "ncr": ["national capital region", "metro manila", "manila"],
                "car": ["cordillera administrative region", "baguio", "mountain province"],
                "region_1": ["ilocos", "la union", "pangasinan"],
                "region_2": ["cagayan", "isabela", "nueva vizcaya"],
                "region_3": ["central luzon", "bulacan", "pampanga", "tarlac"],
                "calabarzon": ["cavite", "laguna", "batangas", "rizal", "quezon"],
                "mimaropa": ["mindoro", "marinduque", "romblon", "palawan"],
                "region_5": ["bicol", "albay", "camarines", "sorsogon"],
                "region_6": ["western visayas", "iloilo", "negros occidental", "aklan"],
                "region_7": ["central visayas", "cebu", "bohol", "negros oriental"],
                "region_8": ["eastern visayas", "leyte", "samar", "tacloban"],
                "region_9": ["zamboanga", "basilan", "sulu"],
                "region_10": ["northern mindanao", "cagayan de oro", "bukidnon"],
                "region_11": ["davao", "davao city", "davao del sur"],
                "region_12": ["soccsksargen", "south cotabato", "sarangani"],
                "region_13": ["caraga", "agusan", "surigao"],
                "barmm": ["bangsamoro", "maguindanao", "lanao del sur"]
            },
            
            "economic_indicators": {
                "gdp_terms": ["gross domestic product", "gdp", "economic output", "national income"],
                "employment_terms": ["jobs", "unemployment", "labor force", "employment rate"],
                "inflation_terms": ["price index", "cpi", "inflation rate", "cost of living"],
                "trade_terms": ["exports", "imports", "balance of trade", "foreign trade"],
                "poverty_terms": ["poverty incidence", "poor", "low income", "disadvantaged"]
            },
            
            "sustainability_indicators": {
                "climate_terms": ["temperature", "rainfall", "weather", "climate change", "monsoon"],
                "energy_terms": ["electricity", "power", "renewable energy", "consumption"],
                "environment_terms": ["emissions", "carbon", "forest", "deforestation", "pollution"],
                "sdg_terms": ["sustainable development goals", "sdg", "sustainability", "development"]
            },
            
            "data_sources": {
                "psa": ["philippine statistics authority", "statistics", "census"],
                "doe": ["department of energy", "energy", "electricity"],
                "nasa": ["climate data", "satellite", "temperature", "precipitation"],
                "worldbank": ["world bank", "development indicators", "international"]
            }
        }
        
        logger.info("Philippine domain knowledge loaded")
    
    def build_knowledge_base(self, table_registry: Dict[str, Any], 
                           schema_registry: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive knowledge base from table and schema registries."""
        # Keep local references so fallback context retrieval can use them
        try:
            self._table_registry = table_registry
            self._schema_registry = schema_registry
        except Exception:
            pass

        if not self.vector_db or not self.embedding_model:
            logger.error("Vector database or embedding model not available")
            return {"status": "error", "message": "Vector components not initialized"}
        
        try:
            knowledge_stats = {
                "tables_processed": 0,
                "columns_processed": 0,
                "domain_concepts_added": 0,
                "embeddings_created": 0
            }
            
            # Process table metadata
            self._index_table_metadata(table_registry, knowledge_stats)
            
            # Process column schemas
            self._index_column_schemas(schema_registry, knowledge_stats)
            
            # Index Philippine domain knowledge
            self._index_domain_knowledge(knowledge_stats)
            
            logger.info(f"Knowledge base built: {knowledge_stats}")
            return {"status": "success", "stats": knowledge_stats}
            
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _index_table_metadata(self, table_registry: Dict[str, Any], stats: Dict[str, int]):
        """Index table metadata into vector database."""
        
        documents = []
        metadatas = []
        ids = []
        
        for table_name, table_info in table_registry.items():
            # Create comprehensive document for each table
            doc_parts = [
                f"Table: {table_name}",
                f"Description: {table_info.get('description', '')}",
                f"Category: {table_info.get('category', '')}",
                f"Layer: {table_info.get('layer', '')}",
                f"Keywords: {', '.join(table_info.get('keywords', []))}"
            ]
            
            if 'data_source' in table_info:
                doc_parts.append(f"Source: {table_info['data_source']}")
            
            document = " | ".join(doc_parts)
            documents.append(document)
            
            metadatas.append({
                "table_name": table_name,
                "layer": table_info.get('layer', 'unknown'),
                "category": table_info.get('category', 'unknown'),
                "source": table_info.get('data_source', 'unknown'),
                "type": "table_metadata"
            })
            
            ids.append(f"table_{table_name}")
            stats["tables_processed"] += 1
        
        # Add to vector database
        if documents:
            self.collections["tables"].add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            stats["embeddings_created"] += len(documents)
    
    def _index_column_schemas(self, schema_registry: Dict[str, Any], stats: Dict[str, int]):
        """Index column schema information."""
        
        documents = []
        metadatas = []
        ids = []
        
        for table_name, schema_info in schema_registry.items():
            columns = schema_info.get('columns', [])
            
            for column in columns:
                if isinstance(column, dict):
                    col_name = column.get('name', '')
                    col_type = column.get('type', '')
                    col_desc = column.get('description', '')
                    
                    doc_parts = [
                        f"Column: {col_name}",
                        f"Table: {table_name}",
                        f"Type: {col_type}",
                        f"Description: {col_desc}"
                    ]
                    
                    document = " | ".join(doc_parts)
                    documents.append(document)
                    
                    metadatas.append({
                        "table_name": table_name,
                        "column_name": col_name,
                        "data_type": col_type,
                        "type": "column_schema"
                    })
                    
                    ids.append(f"col_{table_name}_{col_name}")
                    stats["columns_processed"] += 1
        
        # Add to vector database
        if documents:
            self.collections["columns"].add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            stats["embeddings_created"] += len(documents)
    
    def _index_domain_knowledge(self, stats: Dict[str, int]):
        """Index Philippine domain knowledge."""
        
        documents = []
        metadatas = []
        ids = []
        
        for domain, concepts in self.philippine_knowledge.items():
            if isinstance(concepts, dict):
                for concept_key, terms in concepts.items():
                    document = f"Domain: {domain} | Concept: {concept_key} | Terms: {', '.join(terms)}"
                    
                    documents.append(document)
                    metadatas.append({
                        "domain": domain,
                        "concept": concept_key,
                        "type": "domain_knowledge"
                    })
                    ids.append(f"domain_{domain}_{concept_key}")
                    stats["domain_concepts_added"] += 1
        
        # Add to vector database  
        if documents:
            self.collections["domain_knowledge"].add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            stats["embeddings_created"] += len(documents)
    
    def retrieve_context(self, user_query: str, context_types: List[str] = None, 
                        max_results: int = 10) -> Dict[str, Any]:
        """
        Retrieve relevant context for a user query using vector search and RAG.
        
        Args:
            user_query: Natural language query from user
            context_types: Types of context to retrieve ['tables', 'columns', 'domain', 'history']
            max_results: Maximum number of results per context type
            
        Returns:
            Dict containing retrieved context and metadata
        """
        
        if not self.vector_db or not self.embedding_model:
            logger.warning("Vector search not available, using fallback context")
            return self._fallback_context_retrieval(user_query)
        
        context_types = context_types or ["tables", "columns", "domain_knowledge"]
        
        try:
            retrieved_context = {
                "query": user_query,
                "timestamp": datetime.now().isoformat(),
                "context_types": context_types,
                "results": {}
            }
            
            # Retrieve from each context type
            for context_type in context_types:
                if context_type in self.collections:
                    results = self.collections[context_type].query(
                        query_texts=[user_query],
                        n_results=min(max_results, 10)
                    )
                    
                    retrieved_context["results"][context_type] = {
                        "documents": results["documents"][0] if results["documents"] else [],
                        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                        "distances": results["distances"][0] if results.get("distances") else []
                    }
            
            # Enhanced context processing
            enhanced_context = self._enhance_context_with_philippines_knowledge(
                retrieved_context, user_query
            )
            
            # Cache results
            cache_key = f"query_{hash(user_query)}"
            self.context_cache[cache_key] = enhanced_context
            
            logger.info(f"Context retrieved for query: {user_query[:50]}...")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return self._fallback_context_retrieval(user_query)
    
    def _enhance_context_with_philippines_knowledge(self, context: Dict[str, Any], 
                                                  user_query: str) -> Dict[str, Any]:
        """Enhance retrieved context with Philippine-specific knowledge."""
        
        enhanced = context.copy()
        enhanced["philippine_context"] = {}
        
        query_lower = user_query.lower()
        
        # Detect regional context
        detected_regions = []
        for region, terms in self.philippine_knowledge["regions"].items():
            if any(term in query_lower for term in terms):
                detected_regions.append(region.upper())
        
        if detected_regions:
            enhanced["philippine_context"]["regions"] = detected_regions
        
        # Detect domain context
        detected_domains = {}
        for domain, concepts in self.philippine_knowledge.items():
            if domain != "regions" and isinstance(concepts, dict):
                for concept, terms in concepts.items():
                    if any(term in query_lower for term in terms):
                        if domain not in detected_domains:
                            detected_domains[domain] = []
                        detected_domains[domain].append(concept)
        
        if detected_domains:
            enhanced["philippine_context"]["domains"] = detected_domains
        
        # Add recommendations for better context
        enhanced["recommendations"] = self._generate_context_recommendations(
            enhanced, user_query
        )
        
        return enhanced
    
    def _generate_context_recommendations(self, context: Dict[str, Any], 
                                        user_query: str) -> List[str]:
        """Generate recommendations for improved context retrieval."""
        
        recommendations = []
        
        # Check if specific regions mentioned
        ph_context = context.get("philippine_context", {})
        if "regions" not in ph_context:
            recommendations.append("Consider specifying a Philippine region for more targeted analysis")
        
        # Check table layer relevance
        table_results = context.get("results", {}).get("tables", {})
        if table_results.get("documents"):
            layers = [meta.get("layer", "") for meta in table_results.get("metadatas", [])]
            if "gold" not in layers and "sustainability" in user_query.lower():
                recommendations.append("Consider using Gold layer tables for comprehensive sustainability analysis")
        
        # Domain-specific recommendations
        if "domains" in ph_context:
            domains = ph_context["domains"]
            if "economic_indicators" in domains and "climate_terms" in domains:
                recommendations.append("Cross-domain analysis detected - consider integrated sustainability metrics")
        
        return recommendations
    
    def _fallback_context_retrieval(self, user_query: str) -> Dict[str, Any]:
        """Fallback context retrieval when vector search is unavailable."""
        
        fallback_context = {
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "method": "fallback_keyword_matching",
            "results": {},
            "philippine_context": {}
        }
        
        # Simple keyword matching for Philippine context
        query_lower = user_query.lower()
        
        # Check for regional mentions
        for region, terms in self.philippine_knowledge["regions"].items():
            if any(term in query_lower for term in terms):
                if "regions" not in fallback_context["philippine_context"]:
                    fallback_context["philippine_context"]["regions"] = []
                fallback_context["philippine_context"]["regions"].append(region.upper())
        
        # Check for domain terms
        for domain, concepts in self.philippine_knowledge.items():
            if isinstance(concepts, dict):
                for concept, terms in concepts.items():
                    if any(term in query_lower for term in terms):
                        if domain not in fallback_context["philippine_context"]:
                            fallback_context["philippine_context"][domain] = []
                        fallback_context["philippine_context"][domain].append(concept)
        
        # Lightweight retrieval over registries (tables and columns)
        try:
            # Table scoring
            if hasattr(self, '_table_registry') and isinstance(self._table_registry, dict):
                scored = []
                tokens = set(query_lower.split())
                for tname, tinfo in self._table_registry.items():
                    hay = " ".join([
                        tname.replace('_', ' '),
                        tinfo.get('description', ''),
                        " ".join(tinfo.get('keywords', [])),
                        tinfo.get('category', ''),
                        tinfo.get('layer', ''),
                    ]).lower()
                    score = sum(1 for tok in tokens if tok and tok in hay)
                    if score:
                        scored.append((score, tname, tinfo))
                scored.sort(reverse=True)
                if scored:
                    top = scored[:5]
                    fallback_context['results']['tables'] = {
                        'documents': [f"Table: {t} | Layer: {info.get('layer','')} | {info.get('description','')}" for _, t, info in top],
                        'metadatas': [{'table_name': t, 'layer': info.get('layer','') } for _, t, info in top]
                    }
            # Column scoring
            if hasattr(self, '_schema_registry') and isinstance(self._schema_registry, dict):
                col_hits = []
                tokens = set(query_lower.split())
                for tname, schema in self._schema_registry.items():
                    for col in schema.get('columns', []):
                        if isinstance(col, dict):
                            cname = col.get('name','')
                            cdesc = col.get('description','')
                            hay = f"{cname} {cdesc}".lower()
                            score = sum(1 for tok in tokens if tok and tok in hay)
                            if score:
                                col_hits.append((score, tname, cname))
                col_hits.sort(reverse=True)
                if col_hits:
                    topc = col_hits[:10]
                    fallback_context['results']['columns'] = {
                        'documents': [f"{c} (table: {t})" for _, t, c in topc],
                        'metadatas': [{'table_name': t, 'column_name': c} for _, t, c in topc]
                    }
        except Exception as e:
            logger.warning(f"Fallback registry retrieval failed: {e}")

        return fallback_context
    
    def store_query_pattern(self, user_query: str, selected_table: str, 
                          generated_sql: str, success: bool, metadata: Dict[str, Any] = None):
        """Store successful query patterns for future context retrieval."""
        
        if not self.vector_db:
            return
        
        try:
            document = (
                f"Query: {user_query} | "
                f"Table: {selected_table} | "
                f"SQL: {generated_sql[:200]}... | "
                f"Success: {success}"
            )
            
            pattern_metadata = {
                "user_query": user_query,
                "selected_table": selected_table,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "type": "query_pattern"
            }
            
            if metadata:
                pattern_metadata.update(metadata)
            
            self.collections["query_history"].add(
                documents=[document],
                metadatas=[pattern_metadata],
                ids=[f"query_{datetime.now().timestamp()}"]
            )
            
            logger.info("Query pattern stored for future context retrieval")
            
        except Exception as e:
            logger.error(f"Failed to store query pattern: {str(e)}")
    
    def get_similar_queries(self, user_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get similar historical queries for context and suggestions."""
        
        if not self.vector_db:
            return []
        
        try:
            results = self.collections["query_history"].query(
                query_texts=[user_query],
                n_results=max_results,
                where={"success": True}  # Only successful queries
            )
            
            similar_queries = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    
                    similar_queries.append({
                        "query": metadata.get("user_query", ""),
                        "table": metadata.get("selected_table", ""),
                        "similarity": 1 - distance,  # Convert distance to similarity
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            return similar_queries
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar queries: {str(e)}")
            return []
    
    def clear_context_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()
        logger.info("Context cache cleared")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of context retrieval components."""
        
        status = {
            "embedding_model": self.embedding_model is not None,
            "vector_database": self.vector_db is not None,
            "openai_client": self.client is not None,
            "collections_available": len(self.collections) if self.vector_db else 0,
            "cache_size": len(self.context_cache),
            "philippine_knowledge_loaded": bool(self.philippine_knowledge)
        }
        
        if self.vector_db:
            # Get collection stats
            collection_stats = {}
            for name, collection in self.collections.items():
                try:
                    count = collection.count()
                    collection_stats[name] = count
                except:
                    collection_stats[name] = 0
            status["collection_stats"] = collection_stats
        
        return status
