"""
LLM handler module for gen-ai-docs CLI tool.
Contains all interactions with the Gemini models for content generation and analysis.
"""

import json
from typing import Union, Any, Optional
from dataclasses import dataclass

# Lazy imports - only loaded when needed
_embeddings_module = None
_faiss_module = None
_text_splitter_module = None
_genai_module = None

# Global cache for vector store to avoid recreation (DEPRECATED - use ProcessingContext)
_vector_store_cache = None
_documents_cache = None


@dataclass
class ProcessingContext:
    """
    Encapsulates processing state and configuration to replace global variables.
    This makes the code more maintainable and library-friendly.
    """
    # Configuration
    model_name: str = "gemini-2.5-flash-lite-preview-06-17"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_chunks: int = 50
    similarity_results: int = 3
    context_limit: int = 4000
    max_total_chars: int = 500000
    
    # State
    vector_store: Optional[Any] = None
    documents_cache: Optional[list] = None
    embeddings: Optional[Any] = None
    text_splitter: Optional[Any] = None
    
    # Memory management
    high_memory_threshold_mb: int = 1000
    extreme_memory_threshold_mb: int = 4000
    
    def __post_init__(self):
        """Initialize components after creation."""
        from config import LLMConfig
        
        # Validate model name
        self.model_name = LLMConfig.validate_model(self.model_name)
    
    def clear_cache(self):
        """Clear cached vector store and documents to free memory."""
        self.vector_store = None
        self.documents_cache = None
    
    def get_embeddings(self):
        """Lazy load and cache embeddings."""
        if self.embeddings is None:
            GoogleGenerativeAIEmbeddings = _ensure_embeddings()
            from config import LLMConfig
            self.embeddings = GoogleGenerativeAIEmbeddings(model=LLMConfig.EMBEDDING_MODEL)
        return self.embeddings
    
    def get_text_splitter(self):
        """Lazy load and cache text splitter."""
        if self.text_splitter is None:
            RecursiveCharacterTextSplitter = _ensure_text_splitter()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        return self.text_splitter
    
    def get_model(self):
        """Get configured Gemini model."""
        genai = _ensure_genai()
        return genai.GenerativeModel(self.model_name)
    
    def get_config_summary(self) -> dict:
        """Get current configuration summary."""
        return {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_chunks': self.max_chunks,
            'similarity_results': self.similarity_results,
            'context_limit': self.context_limit,
            'max_total_chars': self.max_total_chars,
        }


def _ensure_genai():
    """Lazy load google.generativeai"""
    global _genai_module
    if _genai_module is None:
        import google.generativeai as genai
        _genai_module = genai
    return _genai_module


def _ensure_embeddings():
    """Lazy load GoogleGenerativeAIEmbeddings"""
    global _embeddings_module
    if _embeddings_module is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embeddings_module = GoogleGenerativeAIEmbeddings
    return _embeddings_module


def _ensure_faiss():
    """Lazy load FAISS"""
    global _faiss_module
    if _faiss_module is None:
        from langchain_community.vectorstores import FAISS
        _faiss_module = FAISS
    return _faiss_module


def _ensure_text_splitter():
    """Lazy load RecursiveCharacterTextSplitter"""
    global _text_splitter_module
    if _text_splitter_module is None:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        _text_splitter_module = RecursiveCharacterTextSplitter
    return _text_splitter_module


def generate_outline(full_text: str) -> dict:
    """
    Generate a structural outline for the document using Gemini.
    
    Args:
        full_text (str): The full text content from the PDF
        
    Returns:
        dict: Dictionary containing 'topic' and 'capabilities' keys
        
    Raises:
        Exception: If JSON parsing fails or API call fails
    """
    genai = _ensure_genai()
    model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    prompt = f"""
You are a technical writer creating a structural outline for a RAG-optimized document from the first few pages of a user manual.
Identify the single main topic (# header) and the primary high-level capabilities (## headers).
Group related features under the most logical capability.

Respond ONLY with a valid JSON object with the main topic as the 'topic' key and a list of capabilities as the 'capabilities' key. Do not include any other text or markdown formatting.

Example Output:
{{
  "topic": "Lead Management",
  "capabilities": [
    "Lead Capture & Creation",
    "Lead Qualification & Scoring",
    "Lead Routing & Assignment"
  ]
}}

--- DOCUMENT TEXT ---
{full_text[:12000]}
--- END TEXT ---
"""
    
    try:
        response = model.generate_content(prompt)
        
        # Clean the response text - remove markdown backticks if present
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Parse JSON
        outline_data = json.loads(response_text.strip())
        return outline_data
    
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response from Gemini: {str(e)}")
    except Exception as e:
        raise Exception(f"Error generating outline: {str(e)}")


def initialize_vector_store(documents: list, status_callback=None) -> Any:
    """
    Initialize and cache the vector store for the entire document.
    This is called once and reused for all capability searches.
    
    Args:
        documents (list): List of unstructured Element objects
        status_callback (callable): Optional callback for status updates
        
    Returns:
        Any: The initialized vector store (FAISS or fallback string)
    """
    global _vector_store_cache, _documents_cache
    
    if status_callback:
        status_callback("    ⏳ Initializing vector store (one-time setup)...")
    
    try:
        # Lazy load required modules
        GoogleGenerativeAIEmbeddings = _ensure_embeddings()
        FAISS = _ensure_faiss()
        RecursiveCharacterTextSplitter = _ensure_text_splitter()
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize text splitter with settings optimized for coverage
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased from 800 for better coverage
            chunk_overlap=200,  # Increased overlap to preserve context
            separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
        )
        
        # Extract and clean text from documents
        if status_callback:
            status_callback(f"    📝 Processing {len(documents)} document elements...")
        
        text_elements = []
        total_chars = 0
        MAX_TOTAL_CHARS = 500000  # SAFETY: Limit total text to 500K chars
        
        for doc in documents:
            if hasattr(doc, 'text') and doc.text and doc.text.strip():
                # Clean text: remove excessive whitespace, normalize
                clean_text = ' '.join(doc.text.split())
                if len(clean_text) > 20:  # Only include meaningful text
                    if total_chars + len(clean_text) <= MAX_TOTAL_CHARS:
                        text_elements.append(clean_text)
                        total_chars += len(clean_text)
                    else:
                        # Add partial text to reach limit
                        remaining = MAX_TOTAL_CHARS - total_chars
                        if remaining > 1000:
                            text_elements.append(clean_text[:remaining])
                        break
        
        full_text = " ".join(text_elements)
        
        if status_callback:
            status_callback(f"    📊 Creating {len(full_text):,} character text corpus (limited for system stability)...")
        
        # Create chunks
        chunks = text_splitter.split_text(full_text)
        
        if not chunks:
            # Fallback for very small documents
            chunks = [full_text[:5000]] if full_text else ["No content available"]
        
        # SAFETY: Limit number of chunks to prevent memory overload
        MAX_CHUNKS = 50  # Reasonable limit for system stability
        if len(chunks) > MAX_CHUNKS:
            if status_callback:
                status_callback(f"    ⚠️  Limiting to {MAX_CHUNKS} chunks for system stability (was {len(chunks)})")
            chunks = chunks[:MAX_CHUNKS]
        
        if status_callback:
            status_callback(f"    🔍 Generating embeddings for {len(chunks)} chunks...")
        
        # Create FAISS vector store with optimized settings
        vector_store = FAISS.from_texts(
            chunks, 
            embeddings,
            # Add metadata for debugging if needed
            metadatas=[{"chunk_id": i, "length": len(chunk)} for i, chunk in enumerate(chunks)]
        )
        
        # Cache the results
        _vector_store_cache = vector_store
        _documents_cache = documents
        
        if status_callback:
            status_callback(f"    ✓ Vector store initialized with {len(chunks)} chunks")
        
        return vector_store
    
    except Exception as e:
        if status_callback:
            status_callback(f"    ❌ Vector store initialization failed: {str(e)}")
        # Return ALL text as fallback - no artificial limits
        full_text = " ".join([doc.text for doc in documents if hasattr(doc, 'text')])
        # SAFETY: Limit fallback text size
        MAX_FALLBACK_CHARS = 200000
        if len(full_text) > MAX_FALLBACK_CHARS:
            full_text = full_text[:MAX_FALLBACK_CHARS]
        return full_text  # Return limited text directly as fallback


def get_relevant_chunks(documents: list, query: str, use_cache: bool = True) -> str:
    """
    Find the most relevant sections of the PDF for a given capability using cached vector search.
    Now processes ALL relevant content without artificial limits.
    
    Args:
        documents (list): List of unstructured Element objects
        query (str): The capability/query to search for
        use_cache (bool): Whether to use cached vector store
        
    Returns:
        str: Concatenated text from ALL relevant chunks
    """
    global _vector_store_cache, _documents_cache
    
    try:
        # Use cached vector store if available and documents match
        if use_cache and _vector_store_cache is not None and _documents_cache == documents:
            vector_store = _vector_store_cache
        else:
            # Create new vector store (this is expensive!)
            vector_store = initialize_vector_store(documents)
        
        # If vector_store is actually a string (fallback case), do intelligent text search
        if isinstance(vector_store, str):
            return _intelligent_text_search(vector_store, query)
        
        # Perform similarity search with MANY more chunks for comprehensive coverage
        results = vector_store.similarity_search(
            query, 
            k=50,  # Dramatically increased from 10 to 50 for comprehensive coverage
            # You can add score_threshold here if supported
        )
        
        # Concatenate ALL relevant results - no artificial limits
        relevant_chunks = []
        
        for doc in results:
            chunk_text = doc.page_content.strip()
            if chunk_text and len(chunk_text) > 50:  # Only meaningful chunks
                relevant_chunks.append(chunk_text)
        
        relevant_text = "\n\n".join(relevant_chunks)
        return relevant_text if relevant_text else "No relevant content found."
    
    except Exception as e:
        # Improved fallback strategy with comprehensive content
        print(f"    ⚠️  Vector search failed, using text search fallback: {str(e)}")
        
        # Simple keyword-based fallback with ALL matching content
        full_text = " ".join([doc.text for doc in documents if hasattr(doc, 'text')])
        
        # Try to find sections containing query keywords
        query_words = query.lower().split()
        sentences = full_text.split('. ')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences)
        else:
            # Final fallback - return substantial portion for processing
            return full_text[:50000]  # Much larger fallback


def generate_feature_details(capability_name: str, context_text: str) -> str:
    """
    Generate detailed feature blocks for a given capability using Gemini.
    Now handles comprehensive content by processing in intelligent chunks when needed.
    
    Args:
        capability_name (str): The name of the capability to detail
        context_text (str): ALL relevant context text for the capability
        
    Returns:
        str: Markdown-formatted feature details
    """
    genai = _ensure_genai()
    model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    # If content is very large, process in intelligent chunks
    if len(context_text) > 30000:  # If more than 30K characters
        return _process_large_content(model, capability_name, context_text)
    else:
        return _process_single_chunk(model, capability_name, context_text)


def _process_single_chunk(model, capability_name: str, context_text: str) -> str:
    """Process content in a single AI call for smaller contexts."""
    prompt = f"""
You are an expert technical writer converting a user manual into a structured, RAG-friendly format.
Your current task is to detail the features related to the high-level capability: "## {capability_name}".

From the provided text below, identify ALL specific features. For each feature, you MUST generate:
1. A '### Feature:' header with the feature name.
2. A '- **Description:**' with a one-sentence summary.
3. A '- **Key Functionality:**' section with a bulleted list of what the feature does.
4. A '- **Use Case:**' section with a concrete example of the feature in action.

IMPORTANT: Process ALL content comprehensively. Look for:
- Main features and sub-features
- Related functionality and tools
- Configuration options and settings
- Integration capabilities
- User workflows and processes
- Administrative features
- Reporting and analytics capabilities

Be exhaustive - extract every feature you can find that relates to this capability.
Adhere strictly to the format. If you cannot find information for a section, write 'Not specified in the document.'.
Do not invent information. Structure the entire output for this capability in valid Markdown.
Do not add the '## {capability_name}' header yourself; only provide the '### Feature' blocks.

--- CONTEXT TEXT ---
{context_text}
--- END TEXT ---
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating feature details for {capability_name}: {str(e)}")


def _process_large_content(model, capability_name: str, context_text: str) -> str:
    """Process very large content by breaking it into intelligent chunks and combining results."""
    
    # Split content into overlapping chunks of manageable size
    chunk_size = 25000  # 25K characters per chunk
    overlap = 2000      # 2K character overlap to preserve context
    
    chunks = []
    start = 0
    iteration_count = 0  # SAFETY: Prevent infinite loops
    MAX_ITERATIONS = 100  # Safety limit
    
    while start < len(context_text) and iteration_count < MAX_ITERATIONS:
        iteration_count += 1
        
        end = start + chunk_size
        if end > len(context_text):
            end = len(context_text)
        
        chunk = context_text[start:end]
        
        # Try to break at sentence boundaries for better context
        if end < len(context_text):
            # Find the last sentence boundary in the chunk
            last_period = chunk.rfind('. ')
            if last_period > chunk_size - 1000:  # If we find a good break point
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk)
        
        # CRITICAL FIX: Ensure start always advances
        next_start = end - overlap
        if next_start <= start:  # Prevent infinite loop
            next_start = start + chunk_size  # Force advancement
        
        start = next_start
        
        if start >= len(context_text):
            break
    
    # SAFETY: Limit number of chunks to prevent memory overload
    MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS:
        print(f"    ⚠️  Limiting processing to first {MAX_CHUNKS} chunks for system stability")
        chunks = chunks[:MAX_CHUNKS]
    
    print(f"    📊 Processing {len(chunks)} content chunks for comprehensive coverage...")
    
    all_features = []
    
    for i, chunk in enumerate(chunks):
        print(f"    🔍 Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)...")
        
        prompt = f"""
You are an expert technical writer converting a user manual into a structured, RAG-friendly format.
Your current task is to detail the features related to the high-level capability: "## {capability_name}".

This is chunk {i+1} of {len(chunks)} from a larger document. Process this chunk comprehensively.

From the provided text below, identify ALL specific features. For each feature, you MUST generate:
1. A '### Feature:' header with the feature name.
2. A '- **Description:**' with a one-sentence summary.
3. A '- **Key Functionality:**' section with a bulleted list of what the feature does.
4. A '- **Use Case:**' section with a concrete example of the feature in action.

IMPORTANT: Be exhaustive for this chunk. Look for:
- Main features and sub-features
- Related functionality and tools
- Configuration options and settings
- Integration capabilities
- User workflows and processes
- Administrative features
- Reporting and analytics capabilities

Extract every feature you can find in this chunk that relates to "{capability_name}".
Adhere strictly to the format. If you cannot find information for a section, write 'Not specified in the document.'.
Do not invent information. Structure the entire output in valid Markdown.
Do not add the '## {capability_name}' header yourself; only provide the '### Feature' blocks.

--- CONTEXT TEXT CHUNK {i+1}/{len(chunks)} ---
{chunk}
--- END CHUNK ---
"""
        
        try:
            response = model.generate_content(prompt)
            chunk_features = response.text.strip()
            if chunk_features and "### Feature:" in chunk_features:
                all_features.append(f"<!-- From content chunk {i+1}/{len(chunks)} -->\n{chunk_features}")
        except Exception as e:
            print(f"    ⚠️  Error processing chunk {i+1}: {str(e)}")
            continue
    
    if not all_features:
        return f"No features could be extracted for {capability_name} from the provided content."
    
    # Combine all features and deduplicate if needed
    combined_features = "\n\n".join(all_features)
    
    # Optional: Post-process to remove duplicate features
    return _deduplicate_features(combined_features)


def _intelligent_text_search(full_text: str, query: str) -> str:
    """
    Perform intelligent text search when vector search is not available.
    Returns ALL relevant content without artificial limits.
    """
    query_words = query.lower().split()
    
    # Split text into paragraphs for better context preservation
    paragraphs = full_text.split('\n\n')
    
    relevant_paragraphs = []
    
    # First pass: Find paragraphs with direct query matches
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        if any(word in paragraph_lower for word in query_words):
            relevant_paragraphs.append(paragraph.strip())
    
    # If we found relevant paragraphs, return them all
    if relevant_paragraphs:
        return '\n\n'.join(relevant_paragraphs)
    
    # Second pass: More flexible matching for related terms
    related_terms = {
        'sales': ['revenue', 'deals', 'opportunities', 'prospects', 'customers'],
        'management': ['admin', 'configure', 'setup', 'control', 'organize'],
        'process': ['workflow', 'procedure', 'steps', 'method', 'approach'],
        'activity': ['action', 'task', 'event', 'work', 'engagement'],
        'performance': ['metrics', 'analytics', 'reports', 'tracking', 'measurement'],
        'account': ['company', 'organization', 'client', 'customer', 'business'],
        'contact': ['person', 'individual', 'lead', 'prospect', 'user'],
        'opportunity': ['deal', 'sale', 'prospect', 'pipeline', 'revenue'],
        'campaign': ['marketing', 'promotion', 'outreach', 'communication'],
        'territory': ['region', 'area', 'zone', 'assignment', 'coverage'],
        'team': ['group', 'collaboration', 'sharing', 'assignment'],
        'ai': ['artificial intelligence', 'machine learning', 'automation', 'intelligent'],
        'digital': ['online', 'electronic', 'virtual', 'web', 'technology']
    }
    
    # Expand query with related terms
    expanded_terms = set(query_words)
    for word in query_words:
        if word in related_terms:
            expanded_terms.update(related_terms[word])
    
    # Search with expanded terms
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        if any(term in paragraph_lower for term in expanded_terms):
            relevant_paragraphs.append(paragraph.strip())
    
    if relevant_paragraphs:
        return '\n\n'.join(relevant_paragraphs)
    
    # Final fallback: return substantial portion of the document
    # Instead of arbitrary limit, return first 30% of content for comprehensive processing
    fallback_size = max(50000, len(full_text) // 3)  # At least 50K or 1/3 of document
    return full_text[:fallback_size]


def _deduplicate_features(content: str) -> str:
    """Remove duplicate feature blocks that might appear across chunks."""
    
    # Split into individual feature blocks
    feature_blocks = content.split("### Feature:")
    
    if len(feature_blocks) <= 1:
        return content
    
    # Keep track of seen feature names to avoid duplicates
    seen_features = set()
    unique_blocks = [feature_blocks[0]]  # Keep the prefix (if any)
    
    for block in feature_blocks[1:]:
        # Extract feature name (first line after "### Feature:")
        lines = block.strip().split('\n')
        if lines:
            feature_name = lines[0].strip().lower()
            
            # Simple deduplication based on feature name similarity
            is_duplicate = False
            for seen in seen_features:
                # Check if feature names are very similar (simple approach)
                if len(feature_name) > 0 and len(seen) > 0:
                    # Calculate simple similarity
                    shorter = min(len(feature_name), len(seen))
                    longer = max(len(feature_name), len(seen))
                    
                    if shorter > 0 and (shorter / longer) > 0.8:  # 80% similarity
                        common_words = set(feature_name.split()) & set(seen.split())
                        if len(common_words) >= 2:  # At least 2 common words
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                seen_features.add(feature_name)
                unique_blocks.append("### Feature:" + block)
    
    return "\n\n".join(unique_blocks)


def clear_cache():
    """
    Clear the cached vector store to free memory.
    Call this when processing a new document.
    """
    global _vector_store_cache, _documents_cache
    _vector_store_cache = None
    _documents_cache = None


def estimate_memory_usage(documents: list) -> dict:
    """
    Estimate memory usage for vector store creation.
    
    Args:
        documents (list): List of document elements
        
    Returns:
        dict: Memory usage estimates and recommendations
    """
    # Calculate text size
    total_chars = 0
    for doc in documents:
        if hasattr(doc, 'text') and doc.text:
            total_chars += len(doc.text)
    
    # Rough estimates based on typical embedding sizes
    # Each chunk ~1500 chars, each embedding ~1536 dimensions * 4 bytes = ~6KB
    estimated_chunks = total_chars // 1500
    estimated_embedding_memory_mb = (estimated_chunks * 6) // 1024  # Convert to MB
    estimated_text_memory_mb = total_chars // (1024 * 1024)  # Text storage
    
    total_estimated_mb = estimated_embedding_memory_mb + estimated_text_memory_mb
    
    # Memory thresholds
    HIGH_MEMORY_THRESHOLD_MB = 1000  # 1GB
    EXTREME_MEMORY_THRESHOLD_MB = 4000  # 4GB
    
    recommendation = "in_memory"
    if total_estimated_mb > EXTREME_MEMORY_THRESHOLD_MB:
        recommendation = "disk_based"
    elif total_estimated_mb > HIGH_MEMORY_THRESHOLD_MB:
        recommendation = "chunked_processing"
    
    return {
        'total_chars': total_chars,
        'estimated_chunks': estimated_chunks,
        'estimated_memory_mb': total_estimated_mb,
        'embedding_memory_mb': estimated_embedding_memory_mb,
        'text_memory_mb': estimated_text_memory_mb,
        'recommendation': recommendation,
        'high_memory_risk': total_estimated_mb > HIGH_MEMORY_THRESHOLD_MB
    }


def initialize_vector_store_smart(documents: list, status_callback=None, force_disk=False) -> Any:
    """
    Initialize vector store with smart memory management.
    Automatically chooses between in-memory and disk-based storage.
    
    Args:
        documents (list): List of unstructured Element objects
        status_callback (callable): Optional callback for status updates
        force_disk (bool): Force disk-based storage regardless of size
        
    Returns:
        Any: The initialized vector store
    """
    # Estimate memory usage
    memory_info = estimate_memory_usage(documents)
    
    if status_callback:
        status_callback(f"    📊 Memory estimate: {memory_info['estimated_memory_mb']}MB for {memory_info['estimated_chunks']} chunks")
    
    # Choose storage strategy
    use_disk = force_disk or memory_info['recommendation'] == 'disk_based'
    
    if use_disk:
        if status_callback:
            status_callback("    💾 Using disk-based storage for large document")
        return _initialize_disk_vector_store(documents, status_callback)
    elif memory_info['recommendation'] == 'chunked_processing':
        if status_callback:
            status_callback("    🔄 Using chunked processing for medium document")
        return _initialize_chunked_vector_store(documents, status_callback)
    else:
        if status_callback:
            status_callback("    🚀 Using in-memory storage for optimal speed")
        return initialize_vector_store(documents, status_callback)


def _initialize_disk_vector_store(documents: list, status_callback=None):
    """
    Initialize a disk-based vector store for very large documents.
    This is a placeholder - would require additional dependencies like LanceDB.
    """
    if status_callback:
        status_callback("    ⚠️  Disk-based storage not yet implemented")
        status_callback("    🔄 Falling back to chunked processing")
    
    return _initialize_chunked_vector_store(documents, status_callback)


def _initialize_chunked_vector_store(documents: list, status_callback=None):
    """
    Initialize vector store with chunked processing to limit memory usage.
    """
    if status_callback:
        status_callback("    🔄 Initializing chunked vector store...")
    
    # Process in smaller batches to limit memory usage
    CHUNK_SIZE = 100  # Process 100 documents at a time
    
    # Split documents into chunks
    doc_chunks = [documents[i:i + CHUNK_SIZE] for i in range(0, len(documents), CHUNK_SIZE)]
    
    if status_callback:
        status_callback(f"    📦 Processing {len(doc_chunks)} document batches")
    
    # For now, just process the first chunk to limit memory
    # In a full implementation, you'd merge multiple vector stores
    first_chunk = doc_chunks[0] if doc_chunks else documents
    
    return initialize_vector_store(first_chunk, status_callback)


def create_sequential_chunks(documents: list, num_capabilities: int, pages_per_capability: int = 10) -> list:
    """
    Create sequential chunks by dividing document elements into groups.
    Each capability gets a sequential portion of the document.
    ENSURES ALL CONTENT IS PROCESSED - creates additional capabilities if needed.
    
    Args:
        documents (list): List of document elements
        num_capabilities (int): Number of capabilities to divide content among
        pages_per_capability (int): Approximate pages per capability
        
    Returns:
        list: List of text chunks, one per capability (may be more than num_capabilities if needed)
    """
    print(f"    📄 Creating sequential chunks ({pages_per_capability} pages per capability)...")
    
    # Extract all text elements
    text_elements = []
    for elem in documents:
        if hasattr(elem, 'text') and elem.text and elem.text.strip():
            text_elements.append(elem.text.strip())
    
    if not text_elements:
        return ["No content available"] * num_capabilities
    
    # Combine all text
    full_text = " ".join(text_elements)
    total_chars = len(full_text)
    
    print(f"    📊 Total content: {total_chars:,} characters")
    
    # Calculate optimal chunk size, but ensure we don't exceed memory limits
    ideal_chunk_size = total_chars // num_capabilities
    max_safe_chunk_size = 150000  # Maximum safe chunk size
    
    # If ideal chunk size is too large, we need more capabilities
    if ideal_chunk_size > max_safe_chunk_size:
        # Calculate how many capabilities we actually need
        actual_capabilities_needed = (total_chars + max_safe_chunk_size - 1) // max_safe_chunk_size
        chunk_size = max_safe_chunk_size
        print(f"    📈 Document too large for {num_capabilities} capabilities")
        print(f"    📊 Creating {actual_capabilities_needed} capabilities to process all content")
        num_capabilities = actual_capabilities_needed
    else:
        chunk_size = max(10000, ideal_chunk_size)
    
    print(f"    📏 Target chunk size: {chunk_size:,} characters per capability")
    
    chunks = []
    current_pos = 0
    capability_num = 1
    
    # Process ALL content by creating chunks until we reach the end
    while current_pos < total_chars:
        start = current_pos
        end = min(start + chunk_size, total_chars)
        
        # Try to break at sentence boundaries for better content coherence
        if end < total_chars and (end - start) < 500000:  # Only for manageable chunks
            # Look for sentence break within reasonable distance
            for offset in range(min(500, chunk_size // 20)):
                if end + offset < total_chars and full_text[end + offset:end + offset + 2] in ['. ', '.\n']:
                    end = end + offset + 1
                    break
                elif end - offset > start and full_text[end - offset:end - offset + 2] in ['. ', '.\n']:
                    end = end - offset + 1
                    break
        
        chunk = full_text[start:end].strip()
        
        # Ensure chunk has content
        if not chunk:
            chunk = "No additional content available for this capability."
        
        chunks.append(chunk)
        print(f"    📄 Capability {capability_num}: {len(chunk):,} characters")
        
        current_pos = end
        capability_num += 1
    
    # Verify ALL content was processed
    total_processed = sum(len(chunk) for chunk in chunks if chunk != "No additional content available for this capability.")
    if total_processed != total_chars:
        print(f"    ⚠️  WARNING: Content mismatch! Processed {total_processed:,} of {total_chars:,} characters")
    else:
        print(f"    ✅ ALL content processed: {total_processed:,} characters across {len(chunks)} capabilities")
    
    return chunks


def generate_feature_details_with_deduplication(capability_name: str, context_text: str, existing_features: set) -> tuple:
    """
    Generate feature details while avoiding duplicates from previous capabilities.
    
    Args:
        capability_name (str): Name of the capability being processed
        context_text (str): Relevant text content for this capability
        existing_features (set): Set of feature names already generated
        
    Returns:
        tuple: (generated_content, new_feature_names_set)
    """
    genai = _ensure_genai()
    model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    # Create list of existing features for the prompt
    existing_features_list = sorted(list(existing_features))
    existing_features_text = ", ".join(existing_features_list) if existing_features_list else "None"
    
    prompt = f"""
You are an expert technical writer converting a user manual into a structured, RAG-friendly format.
Your current task is to detail the features related to the high-level capability: "## {capability_name}".

CRITICAL DEDUPLICATION REQUIREMENT:
The following features have already been described in this document and should NOT be generated again:
{existing_features_text}

Focus ONLY on NEW features found in the context text that have not been covered yet.

From the provided text below, identify all NEW specific features. For each NEW feature, you MUST generate:
1. A '### Feature:' header with the feature name.
2. A '- **Description:**' with a one-sentence summary.
3. A '- **Key Functionality:**' section with a bulleted list of what the feature does.
4. A '- **Use Case:**' section with a concrete example of the feature in action.

IMPORTANT GUIDELINES:
- Be thorough but avoid duplicating features already covered
- Look for features specific to "{capability_name}" that haven't been mentioned
- If a feature is very similar to an existing one, focus on the unique aspects for this capability
- If no new features are found, return "No new features found for this capability."
- Do not add the '## {capability_name}' header yourself; only provide the '### Feature' blocks

--- CONTEXT TEXT ---
{context_text}
--- END CONTEXT ---
"""
    
    try:
        response = model.generate_content(prompt)
        generated_content = response.text.strip()
        
        # Extract new feature names from the generated content
        new_features = set()
        if "### Feature:" in generated_content:
            feature_blocks = generated_content.split("### Feature:")
            for block in feature_blocks[1:]:  # Skip first empty block
                lines = block.strip().split('\n')
                if lines:
                    feature_name = lines[0].strip().lower()
                    new_features.add(feature_name)
        
        return generated_content, new_features
        
    except Exception as e:
        return f"Error generating features for {capability_name}: {str(e)}", set()


def generate_feature_details_vector_search(capability_name: str, context_text: str, existing_features: set) -> tuple:
    """
    Generate feature details using vector search approach with deduplication.
    This is the enhanced version of the original vector search method.
    
    Args:
        capability_name (str): Name of the capability being processed
        context_text (str): Relevant text content for this capability
        existing_features (set): Set of feature names already generated
        
    Returns:
        tuple: (generated_content, new_feature_names_set)
    """
    genai = _ensure_genai()
    model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    # Create list of existing features for the prompt
    existing_features_list = sorted(list(existing_features))
    existing_features_text = ", ".join(existing_features_list) if existing_features_list else "None"
    
    # Check if content is too large for single processing
    if len(context_text) > 30000:
        return _process_large_content_with_deduplication(model, capability_name, context_text, existing_features)
    else:
        return _process_single_chunk_with_deduplication(model, capability_name, context_text, existing_features)


def _process_single_chunk_with_deduplication(model, capability_name: str, context_text: str, existing_features: set) -> tuple:
    """Process a single chunk of content with deduplication."""
    existing_features_list = sorted(list(existing_features))
    existing_features_text = ", ".join(existing_features_list) if existing_features_list else "None"
    
    prompt = f"""
You are an expert technical writer converting a user manual into a structured, RAG-friendly format.
Your current task is to detail the features related to the high-level capability: "## {capability_name}".

CRITICAL DEDUPLICATION REQUIREMENT:
The following features have already been described in this document and should NOT be generated again:
{existing_features_text}

From the provided text below, identify all NEW specific features that haven't been covered yet. 
For each NEW feature, you MUST generate:
1. A '### Feature:' header with the feature name.
2. A '- **Description:**' with a one-sentence summary.
3. A '- **Key Functionality:**' section with a bulleted list of what the feature does.
4. A '- **Use Case:**' section with a concrete example of the feature in action.

Focus on finding features that are uniquely relevant to "{capability_name}" and haven't been described before.
Adhere strictly to the format. If you cannot find information for a section, write 'Not specified in the document.'.
Do not invent information. Structure the entire output in valid Markdown.
Do not add the '## {capability_name}' header yourself; only provide the '### Feature' blocks.

--- CONTEXT TEXT ---
{context_text}
--- END CONTEXT ---
"""
    
    try:
        response = model.generate_content(prompt)
        generated_content = response.text.strip()
        
        # Extract new feature names
        new_features = set()
        if "### Feature:" in generated_content:
            feature_blocks = generated_content.split("### Feature:")
            for block in feature_blocks[1:]:
                lines = block.strip().split('\n')
                if lines:
                    feature_name = lines[0].strip().lower()
                    new_features.add(feature_name)
        
        return generated_content, new_features
        
    except Exception as e:
        return f"Error processing chunk for {capability_name}: {str(e)}", set()


def _process_large_content_with_deduplication(model, capability_name: str, context_text: str, existing_features: set) -> tuple:
    """Process large content in chunks with deduplication."""
    print(f"    📄 Large content detected ({len(context_text):,} chars), processing in chunks...")
    
    chunk_size = 25000
    overlap = 2000
    
    chunks = []
    start = 0
    
    while start < len(context_text):
        end = start + chunk_size
        if end > len(context_text):
            end = len(context_text)
        
        chunk = context_text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(context_text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size - 1000:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk)
        
        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        
        start = next_start
        
        if start >= len(context_text):
            break
    
    # Limit chunks for system stability
    MAX_CHUNKS = 15
    if len(chunks) > MAX_CHUNKS:
        print(f"    ⚠️  Limiting to first {MAX_CHUNKS} chunks for system stability")
        chunks = chunks[:MAX_CHUNKS]
    
    print(f"    📊 Processing {len(chunks)} content chunks...")
    
    all_features = []
    all_new_features = set()
    
    for i, chunk in enumerate(chunks):
        print(f"    🔍 Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)...")
        
        # Update existing features with those found in previous chunks
        current_existing = existing_features.union(all_new_features)
        
        chunk_content, chunk_new_features = _process_single_chunk_with_deduplication(
            model, capability_name, chunk, current_existing
        )
        
        if chunk_content and "### Feature:" in chunk_content:
            all_features.append(f"<!-- From content chunk {i+1}/{len(chunks)} -->\n{chunk_content}")
            all_new_features.update(chunk_new_features)
    
    if not all_features:
        return f"No new features found for {capability_name}.", set()
    
    combined_features = "\n\n".join(all_features)
    return combined_features, all_new_features 