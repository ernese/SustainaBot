"""
Stage 1: Table Decider Component for SustainaBot
Intelligent table selection system that analyzes user queries and matches them to the most appropriate 
Bronze, Silver, or Gold table in the Philippine lakehouse architecture.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from llm.llm_client import get_llm_client, get_model_name
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableDecider:
    """
    Smart table router that analyzes user queries and selects the most appropriate table
    from the Philippine lakehouse (Bronze/Silver/Gold architecture).
    """
    
    def __init__(self, api_key: Optional[str] = None, table_registry_path: Optional[str] = None):
        """Initialize the Table Decider with OpenAI client and table registry.

        Falls back to offline mode (no LLM) when no API key is available
        or client creation fails.
        """
        
        self.api_key = api_key
        try:
            self.client = get_llm_client(api_key=api_key) if api_key else None
        except Exception as e:
            logger.warning(f"LLM client unavailable, using offline table selection: {e}")
            self.client = None
        
        # Initialize table registry
        self.table_registry = self._load_table_registry(table_registry_path)
        self.table_descriptions = self._build_table_descriptions()
        
        logger.info(f"Table Decider initialized with {len(self.table_registry)} tables")
    
    def _load_table_registry(self, registry_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Load table registry from file or create default registry."""
        
        if registry_path and Path(registry_path).exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        
        # Default Philippine lakehouse table registry
        return {
            # BRONZE TABLES (Raw Data) - 20 tables
            "bronze_psa_income_consumption": {
                "layer": "bronze",
                "category": "economic",
                "description": "Raw income and consumption data from Philippine Statistics Authority",
                "keywords": ["income", "consumption", "household", "spending", "earnings", "salary", "wages"],
                "data_source": "PSA",
                "update_frequency": "quarterly"
            },
            "bronze_psa_employment_labor": {
                "layer": "bronze", 
                "category": "economic",
                "description": "Raw employment and labor statistics from PSA",
                "keywords": ["employment", "unemployment", "jobs", "labor", "workforce", "occupation"],
                "data_source": "PSA",
                "update_frequency": "monthly"
            },
            "bronze_psa_population": {
                "layer": "bronze",
                "category": "demographic", 
                "description": "Raw population census and demographic data",
                "keywords": ["population", "demographics", "age", "gender", "residents", "census"],
                "data_source": "PSA",
                "update_frequency": "annual"
            },
            "bronze_psa_tourism": {
                "layer": "bronze",
                "category": "economic",
                "description": "Raw tourism arrivals and statistics data",
                "keywords": ["tourism", "tourists", "visitors", "arrivals", "travel", "hospitality"],
                "data_source": "PSA", 
                "update_frequency": "monthly"
            },
            "bronze_doe_energy_consumption": {
                "layer": "bronze",
                "category": "energy",
                "description": "Raw energy consumption data from Department of Energy",
                "keywords": ["energy", "consumption", "electricity", "power", "usage", "demand"],
                "data_source": "DOE",
                "update_frequency": "monthly"
            },
            "bronze_doe_electricity_pricing": {
                "layer": "bronze",
                "category": "energy",
                "description": "Raw electricity pricing and tariff data",
                "keywords": ["electricity", "price", "tariff", "rates", "cost", "billing"],
                "data_source": "DOE",
                "update_frequency": "quarterly"
            },
            "bronze_nasa_climate_temperature": {
                "layer": "bronze",
                "category": "climate",
                "description": "Raw temperature data from NASA climate monitoring",
                "keywords": ["temperature", "heat", "climate", "weather", "celsius", "fahrenheit"],
                "data_source": "NASA",
                "update_frequency": "daily"
            },
            "bronze_nasa_climate_precipitation": {
                "layer": "bronze",
                "category": "climate", 
                "description": "Raw precipitation and rainfall data from NASA",
                "keywords": ["precipitation", "rainfall", "rain", "weather", "monsoon", "wet"],
                "data_source": "NASA",
                "update_frequency": "daily"
            },
            "bronze_worldbank_gdp": {
                "layer": "bronze",
                "category": "economic",
                "description": "Raw GDP and economic indicators from World Bank",
                "keywords": ["gdp", "economic", "growth", "development", "national", "economy"],
                "data_source": "World Bank",
                "update_frequency": "annual"
            },
            "bronze_worldbank_poverty": {
                "layer": "bronze",
                "category": "social",
                "description": "Raw poverty and social indicators data",
                "keywords": ["poverty", "poor", "social", "welfare", "inequality", "disadvantaged"],
                "data_source": "World Bank",
                "update_frequency": "annual"
            },
            "bronze_environmental_emissions": {
                "layer": "bronze",
                "category": "environment",
                "description": "Raw carbon emissions and environmental data",
                "keywords": ["emissions", "carbon", "co2", "pollution", "environment", "greenhouse"],
                "data_source": "DENR",
                "update_frequency": "monthly"
            },
            "bronze_agricultural_land_use": {
                "layer": "bronze",
                "category": "agriculture",
                "description": "Raw agricultural land use and farming data",
                "keywords": ["agriculture", "farming", "land", "crops", "agricultural", "farm"],
                "data_source": "DA",
                "update_frequency": "seasonal"
            },
            "bronze_fisheries_production": {
                "layer": "bronze",
                "category": "agriculture", 
                "description": "Raw fisheries production and aquaculture data",
                "keywords": ["fisheries", "fish", "aquaculture", "marine", "fishing", "seafood"],
                "data_source": "BFAR",
                "update_frequency": "quarterly"
            },
            "bronze_forestry_data": {
                "layer": "bronze",
                "category": "environment",
                "description": "Raw forestry coverage and deforestation data", 
                "keywords": ["forest", "trees", "deforestation", "forestry", "woodland", "timber"],
                "data_source": "DENR",
                "update_frequency": "annual"
            },
            "bronze_mining_production": {
                "layer": "bronze",
                "category": "industry",
                "description": "Raw mining production and mineral data",
                "keywords": ["mining", "minerals", "extraction", "ore", "quarry", "industrial"],
                "data_source": "MGB",
                "update_frequency": "quarterly"
            },
            "bronze_transportation_data": {
                "layer": "bronze",
                "category": "infrastructure",
                "description": "Raw transportation and infrastructure usage data",
                "keywords": ["transportation", "transport", "traffic", "roads", "vehicles", "infrastructure"],
                "data_source": "DOTr",
                "update_frequency": "monthly"
            },
            "bronze_education_statistics": {
                "layer": "bronze",
                "category": "social",
                "description": "Raw education enrollment and performance data",
                "keywords": ["education", "school", "students", "enrollment", "academic", "learning"],
                "data_source": "DepEd",
                "update_frequency": "annually"
            },
            "bronze_health_statistics": {
                "layer": "bronze",
                "category": "social",
                "description": "Raw health indicators and medical statistics",
                "keywords": ["health", "medical", "hospital", "disease", "healthcare", "wellness"],
                "data_source": "DOH",
                "update_frequency": "monthly"
            },
            "bronze_trade_imports_exports": {
                "layer": "bronze",
                "category": "economic",
                "description": "Raw international trade data - imports and exports",
                "keywords": ["trade", "imports", "exports", "international", "commerce", "goods"],
                "data_source": "DTI",
                "update_frequency": "monthly"
            },
            "bronze_manufacturing_production": {
                "layer": "bronze",
                "category": "industry",
                "description": "Raw manufacturing output and industrial production data",
                "keywords": ["manufacturing", "production", "industry", "factory", "industrial", "output"],
                "data_source": "PSA",
                "update_frequency": "quarterly"
            },
            
            # SILVER TABLES (Cleaned & Structured) - 8 tables
            "silver_fact_climate_weather": {
                "layer": "silver",
                "category": "climate",
                "description": "Cleaned and structured climate and weather metrics with standardized measurements",
                "keywords": ["climate", "weather", "temperature", "rainfall", "humidity", "patterns"],
                "source_tables": ["bronze_nasa_climate_temperature", "bronze_nasa_climate_precipitation"],
                "transformations": "cleaned, standardized units, quality checks"
            },
            "gold_economic_performance_dashboard": {
                "layer": "gold", 
                "category": "economic",
                "description": "Economic performance dashboard with regional income data and GDP indicators",
                "keywords": ["economic", "indicators", "gdp", "income", "regional", "performance", "dashboard"],
                "source_tables": ["silver_fact_economic_indicators"],
                "transformations": "dashboard-ready, regional aggregations, performance metrics"
            },
            "silver_fact_agricultural_land": {
                "layer": "silver",
                "category": "agriculture",
                "description": "Cleaned agricultural land use with productivity metrics",
                "keywords": ["agricultural", "land", "farming", "crops", "productivity", "yield"],
                "source_tables": ["bronze_agricultural_land_use", "bronze_fisheries_production"],
                "transformations": "area calculations, productivity ratios"
            },
            "silver_fact_energy_consumption": {
                "layer": "silver",
                "category": "energy", 
                "description": "Structured energy consumption with efficiency metrics",
                "keywords": ["energy", "consumption", "electricity", "efficiency", "usage", "power"],
                "source_tables": ["bronze_doe_energy_consumption", "bronze_doe_electricity_pricing"],
                "transformations": "consumption ratios, cost analysis"
            },
            "silver_dim_location": {
                "layer": "silver",
                "category": "geography",
                "description": "Master location dimension with regional hierarchy and coordinates",
                "keywords": ["location", "region", "province", "city", "geography", "area"],
                "source_tables": ["multiple bronze tables"],
                "transformations": "standardized location names, hierarchy"
            },
            "silver_dim_time": {
                "layer": "silver", 
                "category": "time",
                "description": "Time dimension with fiscal years, quarters, and Philippine calendar",
                "keywords": ["time", "date", "year", "quarter", "month", "period"],
                "source_tables": ["derived"],
                "transformations": "fiscal calendar, holidays, seasons"
            },
            "silver_dim_indicator": {
                "layer": "silver",
                "category": "metadata",
                "description": "Master indicator definitions with units and calculation methods",
                "keywords": ["indicator", "metric", "definition", "units", "measurement", "kpi"],
                "source_tables": ["metadata"],
                "transformations": "standardized definitions, units"
            },
            "silver_fact_sustainability_metrics": {
                "layer": "silver",
                "category": "sustainability", 
                "description": "Structured sustainability and environmental metrics",
                "keywords": ["sustainability", "environment", "sdg", "green", "carbon", "renewable"],
                "source_tables": ["bronze_environmental_emissions", "bronze_forestry_data"],
                "transformations": "SDG alignment, sustainability scores"
            },
            
            # GOLD TABLES (Business Ready Analytics) - 2 tables
            "gold_sustainability_scorecard": {
                "layer": "gold",
                "category": "analytics",
                "description": "Executive sustainability scorecard with SDG progress and performance ratings across all domains",
                "keywords": ["sustainability", "scorecard", "sdg", "performance", "rating", "progress", "executive"],
                "source_tables": ["silver_fact_sustainability_metrics", "silver_fact_climate_weather", "silver_fact_economic_indicators"],
                "business_purpose": "Executive reporting, policy decisions, strategic planning",
                "aggregation_level": "national, regional, annual"
            },
            "gold_integrated_dashboard": {
                "layer": "gold", 
                "category": "analytics",
                "description": "Integrated cross-domain performance dashboard combining climate, economic, and sustainability KPIs",
                "keywords": ["dashboard", "integrated", "kpi", "performance", "cross-domain", "analytics", "insights"],
                "source_tables": ["all silver tables"],
                "business_purpose": "Strategic monitoring, trend analysis, comparative assessment",
                "aggregation_level": "multi-dimensional, time-series, regional"
            }
        }
    
    def _build_table_descriptions(self) -> str:
        """Build comprehensive table descriptions for the LLM."""
        
        descriptions = []
        descriptions.append("PHILIPPINE LAKEHOUSE TABLE REGISTRY")
        descriptions.append("=" * 50)
        
        # Group by layer
        for layer in ["bronze", "silver", "gold"]:
            layer_tables = {k: v for k, v in self.table_registry.items() if v["layer"] == layer}
            descriptions.append(f"\n{layer.upper()} LAYER ({len(layer_tables)} tables):")
            descriptions.append("-" * 30)
            
            for table_name, table_info in layer_tables.items():
                desc = f"\n{table_name}:"
                desc += f"\n  Description: {table_info['description']}"
                desc += f"\n  Category: {table_info['category']}"
                desc += f"\n  Keywords: {', '.join(table_info['keywords'])}"
                
                if 'data_source' in table_info:
                    desc += f"\n  Source: {table_info['data_source']}"
                if 'business_purpose' in table_info:
                    desc += f"\n  Purpose: {table_info['business_purpose']}"
                if 'aggregation_level' in table_info:
                    desc += f"\n  Level: {table_info['aggregation_level']}"
                    
                descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def decide_table(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user query and decide the best table to use with enhanced context.
        
        Args:
            user_query: Natural language query from user
            context: Retrieved context from Stage 0 (RAG + vector search)
        
        Returns:
            Dict containing selected table name, confidence score, reasoning, and layer
        """
        try:
            # Offline fallback if client is not available
            if self.client is None:
                suggestions = self.get_table_suggestions(user_query, top_n=5)
                if suggestions:
                    # Prefer GOLD/SILVER over BRONZE when available
                    preferred = next((s for s in suggestions if s.get("layer") in ("gold", "silver")), suggestions[0])
                    top = preferred
                    result = {
                        "table_name": top["table_name"],
                        "confidence": top.get("confidence", 70),
                        "reasoning": "Selected via offline keyword-based matching",
                        "layer": top.get("layer"),
                        "valid": True,
                    }
                    logger.info(f"[offline] Selected table: {result['table_name']} (confidence: {result['confidence']}%)")
                    return result
                else:
                    return {
                        "table_name": None,
                        "confidence": 0,
                        "reasoning": "No suitable table found via offline matching",
                        "layer": None,
                        "valid": False,
                        "error": "offline_no_match"
                    }

            # Build enhanced decision prompt with context
            decision_prompt = self._build_decision_prompt(user_query, context)
            
            # Call OpenAI for intelligent table selection
            response = self.client.chat.completions.create(
                model=get_model_name("gpt-4"),
                messages=[
                    {"role": "system", "content": decision_prompt},
                    {"role": "user", "content": f"Query: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            result = self._parse_decision_response(response.choices[0].message.content)
            
            # Validate table exists
            if result["table_name"] in self.table_registry:
                result["valid"] = True
                result["table_info"] = self.table_registry[result["table_name"]]
                logger.info(f"Selected table: {result['table_name']} (confidence: {result['confidence']}%)")
            else:
                result["valid"] = False
                result["error"] = f"Selected table '{result['table_name']}' not found in registry"
                logger.warning(f"Invalid table selected: {result['table_name']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM table decision failed, using offline suggestions: {str(e)}")
            suggestions = self.get_table_suggestions(user_query, top_n=5)
            if suggestions:
                preferred = next((s for s in suggestions if s.get("layer") in ("gold", "silver")), suggestions[0])
                return {
                    "table_name": preferred["table_name"],
                    "confidence": preferred.get("confidence", 60),
                    "reasoning": "Selected via offline keyword-based matching after LLM error",
                    "layer": preferred.get("layer"),
                    "valid": True
                }
            return {
                "table_name": None,
                "confidence": 0,
                "reasoning": f"Error in table selection: {str(e)}",
                "layer": None,
                "valid": False,
                "error": str(e)
            }
    
    def _build_decision_prompt(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the enhanced decision prompt for OpenAI with retrieved context."""
        
        # Build base prompt
        prompt_parts = [
            "You are an expert Table Decider for a Philippine lakehouse data system. Your job is to analyze user queries and select the SINGLE BEST table to answer their question.",
            "",
            "PHILIPPINE LAKEHOUSE ARCHITECTURE:",
            "- BRONZE (20 tables): Raw data from government agencies (PSA, DOE, NASA, World Bank, etc.)",
            "- SILVER (8 tables): Cleaned, structured data with fact and dimension tables", 
            "- GOLD (2 tables): Business-ready analytics and executive dashboards",
            "",
            "TABLE SELECTION STRATEGY:",
            "1. PREFER GOLD tables for executive/strategic questions and cross-domain analysis",
            "2. USE SILVER tables for specific domain analysis requiring clean, structured data",
            "3. USE BRONZE tables only when raw, unprocessed data is specifically needed"
        ]
        
        # Add retrieved context if available
        if context and context.get("results"):
            prompt_parts.extend([
                "",
                "RETRIEVED CONTEXT (from vector search):",
                "=" * 40
            ])
            
            # Add table context
            if "tables" in context["results"]:
                table_docs = context["results"]["tables"].get("documents", [])
                if table_docs:
                    prompt_parts.append("RELEVANT TABLES FOUND:")
                    for i, doc in enumerate(table_docs[:3]):  # Top 3 matches
                        prompt_parts.append(f"- {doc}")
            
            # Add Philippine context
            ph_context = context.get("philippine_context", {})
            if ph_context:
                prompt_parts.append("\nDETECTED PHILIPPINE CONTEXT:")
                if "regions" in ph_context:
                    prompt_parts.append(f"- Regions: {', '.join(ph_context['regions'])}")
                if "domains" in ph_context:
                    for domain, concepts in ph_context["domains"].items():
                        prompt_parts.append(f"- {domain.title()}: {', '.join(concepts)}")
            
            # Add recommendations
            if context.get("recommendations"):
                prompt_parts.append("\nCONTEXT RECOMMENDATIONS:")
                for rec in context["recommendations"]:
                    prompt_parts.append(f"- {rec}")
        
        # Add standard context
        prompt_parts.extend([
            "",
            "PHILIPPINE CONTEXT KEYWORDS:",
            "- Climate: typhoons, monsoons, temperature, rainfall, climate change",
            "- Economic: GDP, employment, income, poverty, trade, tourism", 
            "- Sustainability: SDG, emissions, renewable energy, environment",
            "- Geography: Luzon, Visayas, Mindanao, NCR, regions, provinces",
            "",
            self.table_descriptions
        ])
        
        # Add response format
        prompt_parts.extend([
            "",
            "RESPONSE FORMAT (return EXACTLY in this JSON format):"
        ])
        
        return "\n".join(prompt_parts) + f"""
{{
    "table_name": "exact_table_name_from_registry",
    "confidence": 85,
    "reasoning": "Clear explanation of why this table was selected, incorporating retrieved context if available",
    "layer": "bronze/silver/gold",
    "query_type": "descriptive_query_classification"
}}

IMPORTANT:
- Return ONLY the JSON response, no other text
- Use exact table names from the registry above
- Confidence should be 0-100
- Always provide clear reasoning that incorporates any retrieved context"""

    def _parse_decision_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the OpenAI response to extract decision details."""
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ["table_name", "confidence", "reasoning", "layer"]
                for field in required_fields:
                    if field not in result:
                        result[field] = "Not provided"
                
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse decision response: {str(e)}")
            
            # Fallback parsing
            return {
                "table_name": self._extract_table_name_fallback(response_text),
                "confidence": 50,
                "reasoning": "Fallback parsing used due to response format issue",
                "layer": "unknown",
                "query_type": "unknown"
            }
    
    def _extract_table_name_fallback(self, response_text: str) -> Optional[str]:
        """Fallback method to extract table name from response."""
        
        # Look for table names in the response text
        for table_name in self.table_registry.keys():
            if table_name in response_text:
                return table_name
        
        return None
    
    def validate_table_name(self, table_name: str) -> Dict[str, Any]:
        """
        Validate that the selected table exists in the registry.
        
        Returns:
            Dict with validation status and table information
        """
        if table_name in self.table_registry:
            return {
                "valid": True,
                "table_info": self.table_registry[table_name],
                "layer": self.table_registry[table_name]["layer"],
                "category": self.table_registry[table_name]["category"]
            }
        else:
            return {
                "valid": False,
                "error": f"Table '{table_name}' not found in registry",
                "available_tables": list(self.table_registry.keys())
            }
    
    def get_table_suggestions(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get multiple table suggestions for a query (backup option).
        
        Returns:
            List of top N table suggestions with confidence scores
        """
        try:
            # Simple keyword matching as fallback
            suggestions = []
            query_lower = query.lower()
            
            for table_name, table_info in self.table_registry.items():
                score = 0
                
                # Check keywords match
                for keyword in table_info["keywords"]:
                    if keyword.lower() in query_lower:
                        score += 10
                
                # Check description match
                description_words = table_info["description"].lower().split()
                query_words = query_lower.split()
                
                for query_word in query_words:
                    if query_word in description_words:
                        score += 5
                
                # Prefer gold/silver over bronze for general queries
                if table_info["layer"] == "gold":
                    score += 20
                elif table_info["layer"] == "silver":
                    score += 10
                
                if score > 0:
                    suggestions.append({
                        "table_name": table_name,
                        "confidence": min(score * 2, 100),  # Scale to 0-100
                        "layer": table_info["layer"],
                        "category": table_info["category"],
                        "description": table_info["description"]
                    })
            
            # Sort by confidence and return top N
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            return suggestions[:top_n]
            
        except Exception as e:
            logger.error(f"Error generating table suggestions: {str(e)}")
            return []
    
    def export_table_registry(self, output_path: str):
        """Export table registry to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.table_registry, f, indent=2)
            logger.info(f"Table registry exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting table registry: {str(e)}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the table registry."""
        
        summary = {
            "total_tables": len(self.table_registry),
            "by_layer": {},
            "by_category": {}
        }
        
        for table_info in self.table_registry.values():
            layer = table_info["layer"]
            category = table_info["category"]
            
            summary["by_layer"][layer] = summary["by_layer"].get(layer, 0) + 1
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        
        return summary
