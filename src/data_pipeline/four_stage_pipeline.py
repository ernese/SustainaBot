"""
4-Stage SQL Query Pipeline for SustainaBot
Orchestrates the complete pipeline: Context Retrieval → Table Decider → SQL Generator → Data Transformer
Enhanced with RAG and vector search capabilities for intelligent Philippine data analysis.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import json

from core.context_retrieval import PhilippineContextRetrieval
from llm.table_decider import TableDecider
from core.sql_generator import SQLGenerator  
from data_pipeline.data_transformer import DataTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FourStageQueryPipeline:
    """
    Complete 4-stage pipeline implementation:
    Stage 0: Context Retrieval (RAG + Vector Search)
    Stage 1: Table Identification and Selection (Table Decider)
    Stage 2: SQL Query Generation (SQL Generator)  
    Stage 3: Data Execution and Transformation (Data Transformer)
    """
    
    def __init__(self, api_key: Optional[str] = None, data_path: Optional[str] = None, 
                 enable_vector_search: bool = True):
        """Initialize all four stages of the pipeline."""
        
        try:
            # Stage 0: Context Retrieval (RAG + Vector Search)
            if enable_vector_search:
                self.context_retrieval = PhilippineContextRetrieval(
                    api_key=api_key, 
                    data_path=data_path
                )
                logger.info("Stage 0: Context Retrieval initialized")
            else:
                self.context_retrieval = None
                logger.info("Stage 0: Context Retrieval disabled")
            
            # Stage 1: Table Decider
            self.table_decider = TableDecider(api_key=api_key)
            logger.info("Stage 1: Table Decider initialized")
            
            # Stage 2: SQL Generator  
            self.sql_generator = SQLGenerator(api_key=api_key, data_path=data_path)
            logger.info("Stage 2: SQL Generator initialized")
            
            # Stage 3: Data Transformer
            self.data_transformer = DataTransformer(api_key=api_key, data_path=data_path)
            logger.info("Stage 3: Data Transformer initialized")
            
            # Initialize knowledge base if context retrieval is enabled
            if self.context_retrieval:
                self._initialize_knowledge_base()
            
            logger.info("4-Stage Pipeline fully initialized and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize 4-stage pipeline: {str(e)}")
            raise e
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with table and schema registries."""
        try:
            # Build knowledge base from existing registries
            kb_result = self.context_retrieval.build_knowledge_base(
                self.table_decider.table_registry,
                self.sql_generator.schema_registry
            )
            
            if kb_result["status"] == "success":
                logger.info(f"Knowledge base initialized: {kb_result['stats']}")
            else:
                logger.warning(f"Knowledge base initialization failed: {kb_result['message']}")
                
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
    
    def execute_query_pipeline(self, user_query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete 4-stage pipeline for a user query.
        
        Args:
            user_query: Natural language query from user
            options: Optional configuration for pipeline stages
            
        Returns:
            Complete pipeline result with all stage outputs and final dataframe
        """
        
        pipeline_start = datetime.now()
        options = options or {}
        
        result = {
            "user_query": user_query,
            "pipeline_start_time": pipeline_start.isoformat(),
            "stage_0_context_retrieval": None,
            "stage_1_table_decision": None,
            "stage_2_sql_generation": None, 
            "stage_3_data_transformation": None,
            "final_dataframe": None,
            "pipeline_success": False,
            "total_execution_time": None,
            "error": None
        }
        
        try:
            logger.info(f"Starting 4-Stage Pipeline for query: '{user_query}'")
            
            # ==========================================
            # STAGE 0: CONTEXT RETRIEVAL (RAG + VECTOR SEARCH)
            # ==========================================
            stage0_start = datetime.now()
            logger.info("STAGE 0: Context Retrieval and Semantic Analysis")
            
            if self.context_retrieval:
                context_result = self.context_retrieval.retrieve_context(
                    user_query, 
                    context_types=["tables", "columns", "domain_knowledge"],
                    max_results=options.get("max_context_results", 10)
                )
                result["stage_0_context_retrieval"] = context_result
                
                logger.info(f"Context retrieved: {len(context_result.get('results', {}))} context types")
                
                # Log Philippine-specific context
                ph_context = context_result.get("philippine_context", {})
                if ph_context.get("regions"):
                    logger.info(f"Detected regions: {', '.join(ph_context['regions'])}")
                if ph_context.get("domains"):
                    logger.info(f"Detected domains: {list(ph_context['domains'].keys())}")
                    
                stage0_duration = (datetime.now() - stage0_start).total_seconds()
                logger.info(f"Stage 0 completed in {stage0_duration:.2f} seconds")
            else:
                context_result = None
                logger.info("Stage 0 skipped - context retrieval disabled")
                stage0_duration = 0
            
            # ==========================================
            # STAGE 1: TABLE IDENTIFICATION & SELECTION
            # ==========================================
            stage1_start = datetime.now()
            logger.info("STAGE 1: Table Identification and Selection")

            # Allow forced selection from options
            forced_table = options.get("force_table") if options else None
            forced_layer = options.get("force_layer") if options else None

            if forced_table:
                logger.info(f"Forced table selection: {forced_table} (layer={forced_layer})")
                table_decision = {
                    "table_name": forced_table,
                    "confidence": 100,
                    "reasoning": "User-selected table from UI",
                    "layer": forced_layer or ("bronze" if forced_table.startswith("bronze_") else "silver" if forced_table.startswith("silver_") else "gold"),
                    "valid": True,
                }
                result["stage_1_table_decision"] = table_decision
            else:
                # Enhanced table decision with context
                table_decision = self.table_decider.decide_table(user_query, context=context_result)
                result["stage_1_table_decision"] = table_decision
                
                if not table_decision.get("valid", False):
                    error_msg = f"Stage 1 failed: {table_decision.get('error', 'Invalid table selection')}"
                    logger.error(error_msg)
                    result["error"] = error_msg
                    return result
            
            selected_table = table_decision["table_name"]
            confidence = table_decision.get("confidence", 100 if forced_table else 0)
            layer = table_decision["layer"]

            # Resolve selected table to an actually discovered schema table if needed
            resolved_table = self._resolve_table_name(selected_table, layer)
            if resolved_table != selected_table:
                logger.info(f"Resolved table '{selected_table}' -> '{resolved_table}' based on available schemas")
                table_decision["resolved_table_name"] = resolved_table
                selected_table = resolved_table
            
            logger.info(f"Table Selected: {selected_table} ({layer} layer, {confidence}% confidence)")
            logger.info(f"Reasoning: {table_decision['reasoning']}")
            
            # Validation: prefer discovered schemas over static registry.
            # If the table exists in discovered schemas, proceed; otherwise, report a helpful error.
            if selected_table not in self.sql_generator.schema_registry:
                validation = self.table_decider.validate_table_name(selected_table)
                if not validation.get("valid", False):
                    error_msg = (
                        f"Table validation failed: {validation.get('error', 'Not found')} "
                        f"(note: dynamic datasets must exist in discovered schemas)."
                    )
                    logger.error(error_msg)
                    result["error"] = error_msg
                    return result
            
            stage1_duration = (datetime.now() - stage1_start).total_seconds()
            logger.info(f"Stage 1 completed in {stage1_duration:.2f} seconds")
            
            # ==========================================
            # STAGE 2: SQL QUERY GENERATION
            # ==========================================
            stage2_start = datetime.now()
            logger.info("STAGE 2: SQL Query Generation")
            
            # Schema Retrieval (as per mentor's diagram)
            schema = self.sql_generator.get_table_schema(selected_table)
            logger.info(f"Schema retrieved for {selected_table}: {len(schema.get('columns', []))} columns")
            
            # SQL Generator (as per mentor's diagram)
            # Enhanced SQL generation with context
            sql_generation = self.sql_generator.generate_sql(selected_table, user_query, context=context_result)
            result["stage_2_sql_generation"] = sql_generation
            
            if sql_generation.get("error"):
                error_msg = f"Stage 2 failed: {sql_generation['error']}"
                logger.error(error_msg)
                result["error"] = error_msg
                return result
            
            sql_statement = sql_generation["sql_statement"]
            logger.info(f"SQL Generated: {sql_statement[:100]}...")
            logger.info(f"Estimated rows: {sql_generation.get('estimated_rows', 'Unknown')}")
            
            stage2_duration = (datetime.now() - stage2_start).total_seconds()
            logger.info(f"Stage 2 completed in {stage2_duration:.2f} seconds")
            
            # ==========================================
            # STAGE 3: DATA EXECUTION & TRANSFORMATION
            # ==========================================
            stage3_start = datetime.now()
            logger.info("STAGE 3: Data Execution and Transformation")
            
            # Data Retrieval & Transformation (as per mentor's diagram)
            transformation_options = {
                "max_rows": options.get("max_rows", 1000),
                "max_columns": options.get("max_columns", 15),
                "include_formatting": options.get("include_formatting", True)
            }
            
            data_result = self.data_transformer.execute_and_transform(
                sql_statement, selected_table, user_query, transformation_options, schema=schema
            )
            result["stage_3_data_transformation"] = data_result
            
            if data_result.get("error"):
                error_msg = f"Stage 3 failed: {data_result['error']}"
                logger.error(error_msg)
                result["error"] = error_msg
                return result
            
            # Final outputs (as per mentor's diagram)
            raw_dataframe = data_result["raw_dataframe"]  # Initial dataframe from SQL
            transformed_dataframe = data_result["transformed_dataframe"]  # Altered dataframe
            
            logger.info(f"Data Retrieved: {len(raw_dataframe)} rows from SQL execution")
            logger.info(f"Data Transformed: {len(transformed_dataframe)} rows in final output")
            logger.info(f"Transformations Applied: {len(data_result.get('transformation_log', []))}")
            
            # Generate natural language summary
            nl_summary = self.data_transformer.create_natural_language_summary(data_result)
            
            stage3_duration = (datetime.now() - stage3_start).total_seconds()
            logger.info(f"Stage 3 completed in {stage3_duration:.2f} seconds")
            
            # ==========================================
            # PIPELINE COMPLETION
            # ==========================================
            result["final_dataframe"] = transformed_dataframe  # This is the final "Altered dataframe"
            result["raw_dataframe"] = raw_dataframe  # Original SQL result
            result["natural_language_summary"] = nl_summary
            result["pipeline_success"] = True
            
            pipeline_end = datetime.now()
            total_time = (pipeline_end - pipeline_start).total_seconds()
            result["total_execution_time"] = total_time
            
            # Store successful query pattern for future context retrieval
            if self.context_retrieval and result["pipeline_success"]:
                self.context_retrieval.store_query_pattern(
                    user_query=user_query,
                    selected_table=selected_table,
                    generated_sql=sql_statement,
                    success=True,
                    metadata={
                        "execution_time": total_time,
                        "result_rows": len(transformed_dataframe)
                    }
                )
            
            logger.info(f"4-STAGE PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"Final result: {len(transformed_dataframe)} rows × {len(transformed_dataframe.columns)} columns")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            result["total_execution_time"] = (datetime.now() - pipeline_start).total_seconds()
            return result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components."""
        
        status = {
            "pipeline_ready": True,
            "components": {
                "table_decider": {
                    "status": "ready" if hasattr(self, 'table_decider') else "not_initialized",
                    "table_count": len(self.table_decider.table_registry) if hasattr(self, 'table_decider') else 0
                },
                "sql_generator": {
                    "status": "ready" if hasattr(self, 'sql_generator') else "not_initialized", 
                    "schema_count": len(self.sql_generator.schema_registry) if hasattr(self, 'sql_generator') else 0
                },
                "data_transformer": {
                    "status": "ready" if hasattr(self, 'data_transformer') else "not_initialized",
                    "mapping_count": len(self.data_transformer.data_mappings) if hasattr(self, 'data_transformer') else 0
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if any component failed to initialize
        for component, info in status["components"].items():
            if info["status"] != "ready":
                status["pipeline_ready"] = False
        
        return status

    def _resolve_table_name(self, table_name: str, layer: Optional[str]) -> str:
        """Resolve registry table name to an existing discovered schema table.

        Uses token overlap and prefers matching layer prefixes when available.
        """
        try:
            available = list(self.sql_generator.schema_registry.keys())
            if table_name in available:
                return table_name
            # Normalize
            def strip_layer(name: str):
                for p in ("bronze_", "silver_", "gold_"):
                    if name.startswith(p):
                        return name[len(p):]
                return name
            tgt = strip_layer(table_name).lower()
            tgt_tokens = set(tgt.split('_'))
            best = table_name
            best_score = 0
            for cand in available:
                # Prefer same layer
                if layer and not cand.startswith(layer):
                    layer_bonus = 0
                else:
                    layer_bonus = 1
                ctgt = strip_layer(cand).lower()
                ctokens = set(ctgt.split('_'))
                score = len(tgt_tokens & ctokens) + layer_bonus
                if score > best_score:
                    best = cand
                    best_score = score
            return best
        except Exception:
            return table_name
    
    def get_table_registry_summary(self) -> Dict[str, Any]:
        """Get summary of available tables by layer."""
        
        if not hasattr(self, 'table_decider'):
            return {"error": "Table Decider not initialized"}
        
        return self.table_decider.get_registry_summary()
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of available schemas."""
        
        if not hasattr(self, 'sql_generator'):
            return {"error": "SQL Generator not initialized"}
        
        return self.sql_generator.get_schema_summary()

    def get_layer_eda_summary(self, max_tables_per_layer: int = 2) -> Dict[str, Any]:
        """Create a light EDA summary per layer based on discovered schemas.

        Returns counts per layer and a short preview of a few tables
        (column count, sample column names, and estimated rows if available).
        """
        try:
            schemas = getattr(self.sql_generator, 'schema_registry', {})
            by_layer: Dict[str, Dict[str, Any]] = {}

            # Organize schemas by layer (prefer explicit schema['layer'], else infer from name)
            denylist = {"bronze", "bronze_fixed", "bronze_improved", "new_bronze"}
            for name, schema in schemas.items():
                if name in denylist:
                    continue
                layer = schema.get('layer')
                if not layer:
                    if name.startswith('bronze'):
                        layer = 'bronze'
                    elif name.startswith('silver'):
                        layer = 'silver'
                    else:
                        layer = 'gold'
                if layer not in by_layer:
                    by_layer[layer] = {"tables": []}
                by_layer[layer]["tables"].append((name, schema))

            eda = {"layers": {}}
            for layer, data in by_layer.items():
                tables = data["tables"]
                tables_sorted = sorted(tables, key=lambda x: x[0])
                preview = []
                for name, schema in tables_sorted[:max_tables_per_layer]:
                    cols = [c.get('name') for c in schema.get('columns', []) if isinstance(c, dict) and c.get('name')]
                    preview.append({
                        "table": name,
                        "column_count": len(cols),
                        "columns_preview": cols[:6],
                        "row_count": schema.get('row_count', None)
                    })
                eda["layers"][layer] = {
                    "table_count": len(tables),
                    "preview": preview
                }

            return eda
        except Exception as e:
            logger.warning(f"Layer EDA summary failed: {e}")
            return {"error": str(e)}
    
    def test_pipeline_with_sample_queries(self) -> Dict[str, Any]:
        """Test pipeline with sample queries to verify functionality."""
        
        sample_queries = [
            "Show me recent income data for Philippine regions",
            "What are the employment statistics by region?", 
            "Compare GDP performance across island groups",
            "Show temperature trends for the Philippines",
            "Display sustainability scorecard for all regions"
        ]
        
        test_results = {
            "total_tests": len(sample_queries),
            "successful_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        for i, query in enumerate(sample_queries, 1):
            logger.info(f"Running test {i}/{len(sample_queries)}: {query}")
            
            try:
                result = self.execute_query_pipeline(query, {"max_rows": 10})
                
                if result["pipeline_success"]:
                    test_results["successful_tests"] += 1
                    status = "PASSED"
                else:
                    test_results["failed_tests"] += 1
                    status = "FAILED"
                
                test_results["test_details"].append({
                    "query": query,
                    "status": status,
                    "table_selected": result.get("stage_1_table_decision", {}).get("table_name"),
                    "sql_generated": result.get("stage_2_sql_generation", {}).get("sql_statement") is not None,
                    "data_returned": len(result.get("final_dataframe", pd.DataFrame())),
                    "execution_time": result.get("total_execution_time", 0),
                    "error": result.get("error")
                })
                
            except Exception as e:
                test_results["failed_tests"] += 1
                test_results["test_details"].append({
                    "query": query,
                    "status": "EXCEPTION",
                    "error": str(e)
                })
        
        success_rate = (test_results["successful_tests"] / test_results["total_tests"]) * 100
        test_results["success_rate"] = success_rate
        
        logger.info(f"Pipeline testing completed: {success_rate:.1f}% success rate")
        
        return test_results
    
    def optimize_pipeline_performance(self) -> Dict[str, Any]:
        """Analyze and suggest performance optimizations."""
        
        optimizations = {
            "current_performance": {},
            "recommendations": [],
            "estimated_improvements": {}
        }
        
        try:
            # Analyze table registry size
            table_count = len(self.table_decider.table_registry)
            if table_count > 50:
                optimizations["recommendations"].append(
                    "Consider table registry caching for improved Stage 1 performance"
                )
            
            # Analyze schema registry size  
            schema_count = len(self.sql_generator.schema_registry)
            optimizations["current_performance"]["schema_registry_size"] = schema_count
            
            if schema_count > 30:
                optimizations["recommendations"].append(
                    "Implement schema caching to speed up Stage 2 SQL generation"
                )
            
            # Data access optimization
            optimizations["recommendations"].extend([
                "Consider implementing data connection pooling for Stage 3",
                "Add result caching for frequently accessed tables",
                "Implement async processing for large datasets"
            ])
            
            optimizations["estimated_improvements"] = {
                "stage_1_speedup": "10-15%",
                "stage_2_speedup": "20-25%", 
                "stage_3_speedup": "30-40%",
                "overall_improvement": "25-35%"
            }
            
        except Exception as e:
            optimizations["error"] = str(e)
        
        return optimizations
    
    def export_pipeline_documentation(self) -> str:
        """Generate comprehensive 4-stage pipeline documentation."""
        
        doc_sections = []
        
        # Pipeline Overview
        doc_sections.append("# 4-Stage SQL Query Pipeline Documentation")
        doc_sections.append("\n## Architecture Overview")
        doc_sections.append("""
This system implements a 4-stage pipeline for intelligent data querying and analysis:

**Stage 0: Context Retrieval (RAG + Vector Search)**
- Input: Natural language query
- Component: Context Retrieval
- Function: Retrieves related tables/columns/domain hints to guide downstream stages
- Output: Context payload with semantic hints

**Stage 1: Table Identification and Selection**
- Input: User query + context
- Component: Table Decider 
- Function: Chooses the optimal Bronze/Silver/Gold table, with reasoning and confidence
- Output: Validated table name and layer

**Stage 2: SQL Query Generation**  
- Input: Table name + schema + user query + context
- Component: SQL Generator
- Function: Retrieves schema and generates optimized SQL statement
- Output: Executable SQL statement string

**Stage 3: Data Execution and Transformation**
- Input: SQL statement + table name
- Component: Data Transformer  
- Function: Executes query and transforms raw results into user-friendly format
- Output: Transformed dataframe ready for analysis
""")
        
        # Table Registry
        if hasattr(self, 'table_decider'):
            registry_summary = self.table_decider.get_registry_summary()
            doc_sections.append(f"\n## Table Registry")
            doc_sections.append(f"- **Total Tables**: {registry_summary['total_tables']}")
            
            for layer, count in registry_summary['by_layer'].items():
                doc_sections.append(f"- **{layer.title()} Layer**: {count} tables")
        
        # Schema Information
        if hasattr(self, 'sql_generator'):
            schema_summary = self.sql_generator.get_schema_summary() 
            doc_sections.append(f"\n## Schema Registry")
            doc_sections.append(f"- **Total Schemas**: {schema_summary['total_tables']}")
            doc_sections.append(f"- **Total Columns**: {schema_summary['total_columns']}")
        
        # Usage Example
        doc_sections.append("""
## Usage Example

```python
from data_pipeline.four_stage_pipeline import FourStageQueryPipeline

# Initialize pipeline
pipeline = FourStageQueryPipeline(api_key=None)  # offline mode supported

# Execute query
result = pipeline.execute_query_pipeline("Show me annual climate trends for the Philippines")

# Access results
final_data = result["final_dataframe"]  # Transformed, user-ready data
sql_used = result["stage_2_sql_generation"]["sql_statement"]  # Generated SQL
table_selected = result["stage_1_table_decision"]["table_name"]  # Selected table
```
""")
        
        return "\n".join(doc_sections)
