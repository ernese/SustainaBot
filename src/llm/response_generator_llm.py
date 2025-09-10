"""
Enhanced Response Generator for SustainaBot
Generates EDA-focused responses for Philippine sustainability data analysis
"""

import os
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

import openai
from llm.llm_client import get_llm_client, get_model_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SustainaBotResponseGenerator:
    """Enhanced response generator for EDA and Philippine sustainability analysis."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        """Initialize the SustainaBot response generator."""
        
        # Set OpenAI configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = get_llm_client(api_key=self.api_key)
        
        # Initialize conversation memory
        self.conversation_history = []

        # Load external analyst prompt to declutter code
        self.analyst_prompt = self._load_external_prompt()
        
        logger.info("SustainaBot Response Generator initialized successfully")
    
    def _load_external_prompt(self) -> str:
        """Load the analyst system prompt from prompt.txt at project root."""
        try:
            base = Path(__file__).resolve().parent.parent
            prompt_path = base / "prompt.txt"
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load prompt.txt: {e}")
        # Minimal fallback without emojis
        return (
            "You are a senior data analyst. Explain datasets clearly, highlight key metrics, trends, and data quality. "
            "Keep it concise, actionable, and free of emojis."
        )
    
    def generate_response(self, user_input: str, context: Optional[Dict[str, Any]] = None, 
                         analysis_type: str = "general") -> Dict[str, Any]:
        """Generate contextual response based on user input and analysis context."""
        
        try:
            # Select appropriate prompt based on analysis type
            system_prompt = self._get_system_prompt(analysis_type)
            
            # Build context-aware prompt
            full_prompt = self._build_context_prompt(user_input, context, system_prompt)
            
            # Generate response
            response = self._call_openai(full_prompt, user_input)
            
            # Process and enhance response
            enhanced_response = self._enhance_response(response, context, analysis_type)
            
            return {
                "answer": enhanced_response,
                "analysis_type": analysis_type,
                "context_used": context is not None,
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while analyzing your request: {str(e)}",
                "analysis_type": analysis_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_system_prompt(self, analysis_type: str) -> str:
        """Get appropriate system prompt based on analysis type."""
        return self.analyst_prompt
    
    def _build_context_prompt(self, user_input: str, context: Optional[Dict[str, Any]], 
                             system_prompt: str) -> str:
        """Build comprehensive prompt with context and user input."""
        
        prompt_parts = [system_prompt]
        
        # Add context if available
        if context:
            prompt_parts.append("\n**Analysis Context:**")
            
            # Add dataset information
            if "dataset_info" in context:
                info = context["dataset_info"]
                prompt_parts.append(f"Dataset: {info.get('name', 'Unknown')} with {info.get('shape', ['?', '?'])[0]} rows and {info.get('shape', ['?', '?'])[1]} columns")
            
            # Add key insights
            if "insights" in context and context["insights"]:
                prompt_parts.append("Key Insights:")
                for insight in context["insights"][:5]:
                    prompt_parts.append(f"- {insight.get('category', '')}: {insight.get('insight', '')}")
            
            # Add Philippine-specific insights
            if "ph_insights" in context and context["ph_insights"]:
                prompt_parts.append("Philippine Context:")
                for insight in context["ph_insights"][:3]:
                    prompt_parts.append(f"- {insight.get('category', '')}: {insight.get('insight', '')}")
            
            # Add analysis results
            if "analysis_results" in context:
                results = context["analysis_results"]
                if "kpis" in results:
                    prompt_parts.append(f"KPIs analyzed: {list(results['kpis'].keys())}")
                if "trends" in results:
                    prompt_parts.append(f"Trends identified: {len(results['trends'])} patterns")
                if "recommendations" in results:
                    prompt_parts.append(f"Recommendations available: {len(results['recommendations'])} items")
        
        # Add conversation history
        if self.conversation_history:
            prompt_parts.append("\n**Recent Conversation:**")
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {entry['user']}")
                prompt_parts.append(f"Assistant: {entry['assistant'][:200]}...")
        
        # Add current user question
        prompt_parts.append(f"\n**Current Question:** {user_input}")
        prompt_parts.append("\nProvide a comprehensive, actionable response with specific insights and recommendations for the Philippines context.")
        
        return "\n".join(prompt_parts)
    
    def _call_openai(self, full_prompt: str, user_input: str) -> str:
        """Make API call to OpenAI."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e
    
    def _enhance_response(self, response: str, context: Optional[Dict[str, Any]], 
                         analysis_type: str) -> str:
        """Enhance response with additional context and formatting."""
        
        enhanced_parts = []
        
        # Add analysis type indicator
        type_emoji = {
            "climate": "CLIMATE",
            "economic": "ECONOMIC", 
            "sustainability": "SUSTAINABILITY",
            "data_quality": "DATA_QUALITY",
            "general": "GENERAL"
        }
        
        prefix = type_emoji.get(analysis_type, "GENERAL")
        enhanced_parts.append(f"[{prefix}] **SustainaBot Analysis**\n")
        
        # Add main response
        enhanced_parts.append(response)
        
        # Add data source acknowledgment if context provided
        if context and "dataset_info" in context:
            dataset_name = context["dataset_info"].get("name", "uploaded dataset")
            enhanced_parts.append(f"\n*Analysis based on: {dataset_name}*")
        
        # Add helpful next steps
        next_steps = self._get_next_steps_suggestions(analysis_type, response)
        if next_steps:
            enhanced_parts.append("\n**Suggested Next Steps:**")
            for step in next_steps:
                enhanced_parts.append(f"• {step}")
        
        return "\n".join(enhanced_parts)
    
    def _get_next_steps_suggestions(self, analysis_type: str, response: str) -> List[str]:
        """Generate context-appropriate next steps suggestions."""
        
        suggestions = {
            "climate": [
                "Analyze climate trends over different time periods",
                "Compare climate patterns across Philippine regions",
                "Assess climate vulnerability and adaptation needs"
            ],
            "economic": [
                "Perform regional economic comparison analysis",
                "Analyze sector-specific performance trends", 
                "Evaluate economic diversification opportunities"
            ],
            "sustainability": [
                "Deep dive into specific SDG indicators",
                "Analyze cross-cutting sustainability themes",
                "Evaluate policy intervention effectiveness"
            ],
            "data_quality": [
                "Implement recommended data quality improvements",
                "Validate findings with additional data sources",
                "Develop data monitoring and evaluation framework"
            ],
            "general": [
                "Focus analysis on specific domain (climate/economic/sustainability)",
                "Upload additional datasets for comparative analysis",
                "Generate detailed data profile report"
            ]
        }
        
        return suggestions.get(analysis_type, suggestions["general"])[:3]
    
    def update_conversation_history(self, user_input: str, assistant_response: str):
        """Update conversation history for context awareness."""
        
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def generate_summary_response(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary response from comprehensive analysis results."""
        
        try:
            # Build summary prompt
            summary_prompt = f"""
            {self.eda_prompts['system_prompt']}
            
            **Comprehensive Analysis Results:**
            
            Climate Analysis: {json.dumps(analysis_results.get('climate_analysis', {}), indent=2)}
            Economic Analysis: {json.dumps(analysis_results.get('economic_analysis', {}), indent=2)}  
            Sustainability Analysis: {json.dumps(analysis_results.get('sustainability_analysis', {}), indent=2)}
            
            Generate an executive summary that:
            1. Highlights the top 3-5 key findings across all domains
            2. Identifies critical interconnections between climate, economic, and sustainability factors
            3. Provides strategic recommendations for Philippine development priorities
            4. Suggests actionable next steps for policy makers and stakeholders
            
            Format as a clear, executive-level briefing with specific insights and recommendations.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "answer": f"**Executive Summary - Philippine Sustainability Analysis**\n\n{response.choices[0].message.content}",
                "analysis_type": "executive_summary",
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating summary response: {str(e)}")
            return {
                "answer": f"Error generating executive summary: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_recommendation_response(self, priority_areas: List[str], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific recommendations based on priority areas."""
        
        try:
            recommendation_prompt = f"""
            {self.eda_prompts['system_prompt']}
            
            **Priority Areas Identified:** {', '.join(priority_areas)}
            
            **Context:** {json.dumps(context, indent=2)}
            
            Generate specific, actionable recommendations for each priority area focusing on:
            1. Immediate actions (0-6 months)
            2. Medium-term initiatives (6-24 months)  
            3. Long-term strategic goals (2-5 years)
            4. Key performance indicators to track progress
            5. Required resources and partnerships
            
            Tailor recommendations to Philippine context, institutional capacity, and development priorities.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": recommendation_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "answer": f"**Strategic Recommendations**\n\n{response.choices[0].message.content}",
                "analysis_type": "recommendations",
                "priority_areas": priority_areas,
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "answer": f"Error generating recommendations: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Backward compatibility with existing code
class StackOverflowResponseGenerator(SustainaBotResponseGenerator):
    """Backward compatibility wrapper."""
    
    def __init__(self, azure_endpoint: str = None, azure_api_key: str = None, 
                 deployment_name: str = "gpt-4", api_key: str = None, **kwargs):
        # Use api_key parameter directly for new SustainaBot
        final_api_key = api_key or azure_api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(api_key=final_api_key, model=deployment_name, **kwargs)
        
        # Store for backward compatibility
        self.available_collections = []
        
    def switch_collection(self, collection_name: str):
        """Backward compatibility method."""
        logger.info(f"Collection switching not needed in EDA mode: {collection_name}")
        pass
