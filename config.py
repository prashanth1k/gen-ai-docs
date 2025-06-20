"""
Configuration module for gen-ai-docs CLI tool.
Handles loading environment variables and initializing API clients.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. "
        "Please create a .env file with your Gemini API key. "
        "You can copy .env.example and add your key."
    )

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)


# Default LLM Configuration - can be overridden via CLI options
class LLMConfig:
    """Configuration for LLM processing parameters."""
    
    # Model Selection
    DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    AVAILABLE_MODELS = [
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",  # Legacy fallback
    ]
    
    # Text Chunking Parameters
    DEFAULT_CHUNK_SIZE = 1500
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MAX_CHUNKS = 50
    
    # Vector Store Parameters
    EMBEDDING_MODEL = "models/embedding-001"
    DEFAULT_SIMILARITY_SEARCH_RESULTS = 3
    
    # Memory Management
    HIGH_MEMORY_THRESHOLD_MB = 1000
    EXTREME_MEMORY_THRESHOLD_MB = 4000
    MAX_TOTAL_CHARS = 500000  # Safety limit for text processing
    
    # API and Processing Limits
    DEFAULT_CONTEXT_LIMIT = 4000  # Characters for AI context
    DEFAULT_MAX_WORKERS = 2  # For parallel processing
    
    @classmethod
    def validate_model(cls, model_name: str) -> str:
        """Validate and normalize model name."""
        if model_name in cls.AVAILABLE_MODELS:
            return model_name
        
        # Try to match partial names
        for available_model in cls.AVAILABLE_MODELS:
            if model_name.lower() in available_model.lower():
                return available_model
        
        # Fallback to default with warning
        print(f"⚠️  Unknown model '{model_name}', using default: {cls.DEFAULT_MODEL}")
        return cls.DEFAULT_MODEL
    
    @classmethod
    def get_config_summary(cls, **overrides) -> dict:
        """Get current configuration with any overrides applied."""
        config = {
            'model': overrides.get('model', cls.DEFAULT_MODEL),
            'chunk_size': overrides.get('chunk_size', cls.DEFAULT_CHUNK_SIZE),
            'chunk_overlap': overrides.get('chunk_overlap', cls.DEFAULT_CHUNK_OVERLAP),
            'max_chunks': overrides.get('max_chunks', cls.DEFAULT_MAX_CHUNKS),
            'similarity_results': overrides.get('similarity_results', cls.DEFAULT_SIMILARITY_SEARCH_RESULTS),
            'context_limit': overrides.get('context_limit', cls.DEFAULT_CONTEXT_LIMIT),
            'max_workers': overrides.get('max_workers', cls.DEFAULT_MAX_WORKERS),
        }
        
        # Validate model
        config['model'] = cls.validate_model(config['model'])
        
        return config 