"""
Centralized LLM Client Configuration
Handles both Azure OpenAI and standard OpenAI configurations with unified precedence:
1. Sidebar/runtime parameters (highest priority)
2. Environment variables
3. .env file values (lowest priority)
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from openai import OpenAI
try:
    # Ensure .env variables are loaded even when app.py hasn't run yet
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Safe to ignore if dotenv isn't available; env may already be set
    pass

logger = logging.getLogger(__name__)

class LLMClientManager:
    """Centralized manager for OpenAI/Azure OpenAI client configuration."""
    
    def __init__(self):
        self._client = None
        self._config = {}
        
    def load_config(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with proper precedence:
        1. Runtime parameters (api_key)
        2. Environment variables
        3. .env file (loaded separately)
        """
        
        # Start with environment variables
        # Gather environment-based config with sensible fallbacks to support existing .envs
        env_api_key = (
            os.getenv('OPENAI_API_KEY')
            or os.getenv('OPENAI_KEY')
            or os.getenv('AZURE_OPENAI_API_KEY')
        )
        env_endpoint = (
            os.getenv('OPENAI_ENDPOINT')
            or os.getenv('AZURE_OPENAI_ENDPOINT')
        )
        env_deployment = (
            os.getenv('OPENAI_DEPLOYMENT_NAME')
            or os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        )
        env_api_version = (
            os.getenv('OPENAI_DEPLOYMENT_VERSION')
            or os.getenv('OPENAI_API_VERSION')
            or os.getenv('AZURE_OPENAI_API_VERSION')
        )
        env_embedding = (
            os.getenv('OPENAI_EMBEDDING_NAME')
            or os.getenv('AZURE_OPENAI_EMBEDDING_NAME')
        )

        config = {
            'api_key': env_api_key,
            'endpoint': env_endpoint,
            'deployment_name': env_deployment,
            'api_version': env_api_version,
            'embedding_deployment': env_embedding,
        }
        
        # Override with runtime parameter (highest priority)
        if api_key:
            config['api_key'] = api_key
            
        # Determine if we should use Azure configuration
        config['is_azure'] = bool(config['endpoint'] and config['deployment_name'])
        
        self._config = config
        return config
    
    def get_client(self, api_key: Optional[str] = None, model: Optional[str] = None) -> OpenAI:
        """
        Get configured OpenAI client (Azure or standard).
        
        Args:
            api_key: Optional API key override
            model: Optional model override
            
        Returns:
            Configured OpenAI client
        """
        
        config = self.load_config(api_key)
        
        if not config['api_key']:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable, "
                "configure .env file, or pass api_key parameter."
            )
        
        # Configure for Azure OpenAI
        if config['is_azure']:
            logger.info("Configuring Azure OpenAI client")
            
            base_url = f"{config['endpoint'].rstrip('/')}/openai/deployments/{config['deployment_name']}"
            
            client = OpenAI(
                api_key=config['api_key'],
                base_url=base_url,
                default_query={"api-version": config['api_version']} if config['api_version'] else None,
                timeout=30.0,  # Add timeout for resilience
                max_retries=3   # Add retries for resilience
            )
            
            logger.info(f"Azure OpenAI client configured: {config['endpoint']} -> {config['deployment_name']}")
            
        else:
            # Standard OpenAI configuration
            logger.info("Configuring standard OpenAI client")
            
            client = OpenAI(
                api_key=config['api_key'],
                timeout=30.0,
                max_retries=3
            )
            
        self._client = client
        return client
    
    def get_model_name(self, default_model: str = "gpt-4") -> str:
        """
        Get the appropriate model name based on configuration.
        For Azure, returns deployment name. For OpenAI, returns model name.
        """
        if self._config.get('is_azure') and self._config.get('deployment_name'):
            return self._config['deployment_name']
        return default_model
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration summary for debugging."""
        return {
            'provider': 'Azure OpenAI' if self._config.get('is_azure') else 'OpenAI',
            'endpoint': self._config.get('endpoint', 'api.openai.com'),
            'deployment': self._config.get('deployment_name', 'N/A'),
            'api_version': self._config.get('api_version', 'N/A'),
            'has_api_key': bool(self._config.get('api_key')),
            'model_name': self.get_model_name()
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the LLM connection and return status.
        
        Returns:
            Dict with connection test results
        """
        try:
            if not self._client:
                return {
                    'status': 'error',
                    'message': 'No client configured'
                }
            
            # Simple test with minimal token usage
            response = self._client.chat.completions.create(
                model=self.get_model_name(),
                messages=[{"role": "user", "content": "Test connection - respond with 'OK'"}],
                max_tokens=5,
                timeout=10
            )
            
            return {
                'status': 'success',
                'message': 'Connection successful',
                'response': response.choices[0].message.content.strip(),
                'model_used': response.model if hasattr(response, 'model') else self.get_model_name()
            }
            
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }


# Global instance for app-wide use
llm_manager = LLMClientManager()


def get_llm_client(api_key: Optional[str] = None, model: Optional[str] = None) -> OpenAI:
    """
    Convenience function to get configured LLM client.
    
    Args:
        api_key: Optional API key override
        model: Optional model override (not used for client config, but available)
        
    Returns:
        Configured OpenAI client
    """
    return llm_manager.get_client(api_key=api_key, model=model)


def get_model_name(default: str = "gpt-4") -> str:
    """Get the appropriate model name for the current configuration."""
    return llm_manager.get_model_name(default)


def test_llm_connection(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Test LLM connection with optional API key override."""
    if api_key:
        llm_manager.get_client(api_key=api_key)
    return llm_manager.test_connection()


def get_llm_config_summary() -> Dict[str, Any]:
    """Get current LLM configuration summary."""
    return llm_manager.get_config_summary()
