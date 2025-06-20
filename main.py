"""
Main CLI application for gen-ai-docs.
Orchestrates the entire process of converting PDF documents to RAG-optimized markdown.
"""

import click
from tqdm import tqdm
import time
import os
import logging
import psutil  # For monitoring system resources
import gc  # For garbage collection
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lazy import flag - expensive modules loaded on demand
_modules_loaded = False


def monitor_system_resources():
    """
    Monitor system resources and return current usage.
    Returns dict with memory and CPU usage.
    """
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=0.1)
        system_memory = psutil.virtual_memory()
        
        return {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'system_memory_percent': system_memory.percent,
            'system_memory_available_gb': system_memory.available / (1024**3)
        }
    except:
        return {'memory_mb': 0, 'cpu_percent': 0, 'system_memory_percent': 0, 'system_memory_available_gb': 0}


def check_resource_safety():
    """
    Check if system resources are safe to continue processing.
    Raises exception if resources are critically low.
    """
    resources = monitor_system_resources()
    
    # Check available system memory
    if resources['system_memory_available_gb'] < 1.0:  # Less than 1GB available
        raise Exception("❌ CRITICAL: Less than 1GB system memory available. Operation aborted to prevent system crash.")
    
    # Check process memory usage
    if resources['memory_mb'] > 4000:  # More than 4GB used by this process
        print(f"⚠️  HIGH MEMORY USAGE: {resources['memory_mb']:.0f}MB used by this process")
        # Force garbage collection
        gc.collect()
    
    # Check system memory usage
    if resources['system_memory_percent'] > 90:
        raise Exception(f"❌ CRITICAL: System memory usage at {resources['system_memory_percent']:.1f}%. Operation aborted to prevent system crash.")
    
    return resources


def load_heavy_modules():
    """
    Lazy load heavy modules only when needed for processing.
    This significantly improves startup time.
    """
    global _modules_loaded
    if _modules_loaded:
        return
    
    print("⏳ Loading AI and processing modules (one-time setup)...")
    start_time = time.time()
    
    # Check system resources before loading heavy modules
    try:
        check_resource_safety()
    except Exception as e:
        print(f"  {str(e)}")
        exit(1)
    
    # Import heavy modules with progress indicators
    try:
        print("  📦 Loading configuration...")
        global config
        import config  # This initializes the Gemini API
        
        print("  📄 Loading PDF parser...")
        global pdf_parser
        import pdf_parser
        
        print("  🤖 Loading AI handler...")
        global llm_handler
        import llm_handler
        
        print("  📝 Loading output writer...")
        global output_writer
        import output_writer
        
        load_time = time.time() - start_time
        print(f"  ✅ All modules loaded ({load_time:.1f}s)")
        
        # Check resources after loading
        resources = monitor_system_resources()
        print(f"  📊 Memory usage: {resources['memory_mb']:.0f}MB | System: {resources['system_memory_percent']:.1f}%")
        print()
        
        _modules_loaded = True
        
    except Exception as e:
        print(f"  ❌ Failed to load modules: {str(e)}")
        raise


def setup_logging(input_pdf_path):
    """
    Set up logging to file with timestamp-based filename.
    
    Args:
        input_pdf_path (str): Path to input PDF for log naming
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamp-based log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
    log_filename = f"logs/gen-ai-docs_{pdf_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger, log_filename


def log_and_print(message, logger=None):
    """
    Print to console and log to file.
    
    Args:
        message (str): Message to print and log
        logger: Logger instance (optional)
    """
    print(message)
    if logger:
        # Remove emoji and clean for log file
        clean_message = message.encode('ascii', 'ignore').decode('ascii')
        logger.info(clean_message)


def process_capability_parallel(capability_data):
    """
    Process a single capability in parallel execution.
    
    Args:
        capability_data (dict): Contains capability info and processing function
        
    Returns:
        dict: Processing results including timing and content
    """
    capability_name = capability_data['capability']
    content_chunk = capability_data['content']
    existing_features = capability_data['existing_features']
    process_func = capability_data['process_func']
    thread_id = threading.current_thread().ident
    
    start_time = time.time()
    
    try:
        # Call the appropriate processing function
        if 'deduplication' in process_func.__name__:
            feature_details, new_features = process_func(capability_name, content_chunk, existing_features)
        else:
            # Fallback for non-deduplication functions
            feature_details = process_func(capability_name, content_chunk)
            new_features = set()
        
        processing_time = time.time() - start_time
        
        return {
            'capability': capability_name,
            'feature_details': feature_details,
            'new_features': new_features,
            'processing_time': processing_time,
            'content_chars': len(content_chunk),
            'generated_chars': len(feature_details),
            'thread_id': thread_id,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'capability': capability_name,
            'feature_details': f"Error processing {capability_name}: {str(e)}",
            'new_features': set(),
            'processing_time': processing_time,
            'content_chars': len(content_chunk),
            'generated_chars': 0,
            'thread_id': thread_id,
            'success': False,
            'error': str(e)
        }


@click.group()
def cli():
    """gen-ai-docs: Convert PDF documents to RAG-optimized Markdown files using AI.
    
    DEFAULT: Sequential chunking with PyPDF2 (fast, comprehensive, no redundancy).
    Use --vector-search for targeted content retrieval with deduplication.
    Use --enhanced or --hi-res for better PDF parsing quality.
    """
    pass


@cli.command()
@click.option('--input-pdf', required=True, help='Path to the input PDF file.')
@click.option('--output-dir', default='./output', help='Directory to save the output Markdown files.')
@click.option('--enhanced', is_flag=True, help='Use unstructured library with fast strategy (slower but better formatting).')
@click.option('--hi-res', is_flag=True, help='Use unstructured library with hi-res strategy (slowest, best quality).')
@click.option('--vector-search', is_flag=True, help='Use vector search with deduplication (slower but more targeted). Default uses sequential chunking.')
@click.option('--sequential-pages', type=int, default=10, help='Pages per capability for sequential chunking (default: 10).')
@click.option('--parallel', type=int, default=2, help='Number of parallel workers for capability processing (default: 2, use 0 to disable parallel processing).')
@click.option('--smart-memory', is_flag=True, help='Enable smart memory management for large documents (auto-detects optimal strategy).')
@click.option('--model', default=None, help='Gemini model to use (default: gemini-2.5-flash-lite-preview-06-17).')
@click.option('--chunk-size', type=int, default=None, help='Text chunk size for vector processing (default: 1500).')
@click.option('--chunk-overlap', type=int, default=None, help='Overlap between text chunks (default: 200).')
@click.option('--similarity-results', type=int, default=None, help='Number of similar chunks to retrieve (default: 3).')
@click.option('--context-limit', type=int, default=None, help='Maximum context characters for AI (default: 4000).')
def process(input_pdf, output_dir, enhanced, hi_res, vector_search, sequential_pages, parallel, smart_memory, model, chunk_size, chunk_overlap, similarity_results, context_limit):
    """
    Process a PDF file and generate RAG-optimized markdown documentation.
    
    DEFAULT: Uses sequential page chunking with 2-worker parallel processing (fast, comprehensive coverage, minimal redundancy).
    
    PROCESSING MODES:
    - Default: Sequential chunking with parallel processing (2 workers) - divides PDF into page groups, processes capabilities concurrently
    - --vector-search: Vector search with deduplication - finds most relevant content per capability
    - --parallel 0: Disable parallel processing (sequential, one capability at a time)
    
    PDF PARSING MODES:
    - Default: Ultra-fast PyPDF2 text extraction (maximum speed, basic formatting)
    - --enhanced: Unstructured library with fast strategy (slower but better formatting)
    - --hi-res: Unstructured library with hi-res strategy (slowest, best quality)
    
    Args:
        input_pdf (str): Path to the input PDF file
        output_dir (str): Directory to save the output markdown files
        enhanced (bool): Use unstructured library with fast strategy
        hi_res (bool): Use unstructured library with hi-res strategy
        vector_search (bool): Use vector search with deduplication instead of sequential chunking
        sequential_pages (int): Pages per capability for sequential chunking
        parallel (int): Number of parallel workers (0 disables parallel processing)
        smart_memory (bool): Enable smart memory management for large documents
        model (str): Gemini model to use for generation
        chunk_size (int): Text chunk size for vector processing
        chunk_overlap (int): Overlap between text chunks
        similarity_results (int): Number of similar chunks to retrieve
        context_limit (int): Maximum context characters for AI
    """
    app_start_time = time.time()
    
    # Quick validation before heavy imports
    if not os.path.exists(input_pdf):
        click.echo(f"❌ Error: PDF file not found: {input_pdf}", err=True)
        exit(1)
    
    # Show file info before loading heavy modules
    try:
        file_size_mb = os.path.getsize(input_pdf) / (1024 * 1024)
        print(f"🚀 gen-ai-docs starting...")
        print(f"📄 Input: {input_pdf} ({file_size_mb:.1f}MB)")
        print(f"📁 Output: {output_dir}")
        
        # Show processing modes
        print("🔧 PROCESSING MODE:")
        if vector_search:
            print(f"  📊 Vector search with deduplication (targeted content)")
        else:
            print(f"  📄 Sequential chunking ({sequential_pages} pages per capability)")
        
        if parallel > 0:
            print(f"  🚀 Parallel processing ({parallel} workers)")
        else:
            print(f"  🔄 Sequential processing (one capability at a time)")
        
        if smart_memory:
            print(f"  🧠 Smart memory management (auto-detects optimal strategy)")
        
        print("🔧 PDF PARSING MODE:")
        if hi_res:
            print(f"  🔍 High-resolution (slowest, best quality)")
        elif enhanced:
            print(f"  ⚡ Enhanced (balanced speed/quality)")
        else:
            print(f"  ⚡ Ultra-fast (default - maximum speed)")
        
        print("🔧 AI CONFIGURATION:")
        config_summary = processing_context.get_config_summary()
        print(f"  🤖 Model: {config_summary['model_name']}")
        print(f"  📝 Chunk Size: {config_summary['chunk_size']} chars")
        print(f"  🔄 Chunk Overlap: {config_summary['chunk_overlap']} chars")
        print(f"  🎯 Similarity Results: {config_summary['similarity_results']}")
        print(f"  📊 Context Limit: {config_summary['context_limit']} chars")
        
        print()
    except Exception:
        pass
    
    # Load heavy modules only when we actually need them
    load_heavy_modules()
    
    # Create ProcessingContext with user configuration (after heavy modules are loaded)
    from config import LLMConfig
    
    # Build configuration overrides from CLI options
    config_overrides = {}
    if model is not None:
        config_overrides['model'] = model
    if chunk_size is not None:
        config_overrides['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        config_overrides['chunk_overlap'] = chunk_overlap
    if similarity_results is not None:
        config_overrides['similarity_results'] = similarity_results
    if context_limit is not None:
        config_overrides['context_limit'] = context_limit
    if parallel != 2:  # Only override if different from default
        config_overrides['max_workers'] = parallel
    
    # Get final configuration
    final_config = LLMConfig.get_config_summary(**config_overrides)
    
    # Create processing context
    processing_context = llm_handler.ProcessingContext(
        model_name=final_config['model'],
        chunk_size=final_config['chunk_size'],
        chunk_overlap=final_config['chunk_overlap'],
        similarity_results=final_config['similarity_results'],
        context_limit=final_config['context_limit'],
        max_chunks=final_config['max_chunks'],
        max_total_chars=LLMConfig.MAX_TOTAL_CHARS,
        high_memory_threshold_mb=LLMConfig.HIGH_MEMORY_THRESHOLD_MB,
        extreme_memory_threshold_mb=LLMConfig.EXTREME_MEMORY_THRESHOLD_MB
    )
    
    # Setup logging
    logger, log_filename = setup_logging(input_pdf)
    
    # Statistics tracking
    stats = {
        'app_start_time': app_start_time,
        'processing_start_time': time.time(),
        'input_pdf': input_pdf,
        'output_dir': output_dir,
        'log_file': log_filename,
        'pdf_parsing_time': 0,
        'outline_generation_time': 0,
        'feature_processing_time': 0,
        'file_writing_time': 0,
        'total_elements': 0,
        'element_types': {},
        'outline_elements_used': 0,
        'outline_characters': 0,
        'topic': '',
        'capabilities_count': 0,
        'capabilities': [],
        'capability_processing_times': [],
        'total_relevant_content_chars': 0,
        'total_generated_content_chars': 0,
        'final_markdown_chars': 0,
        'output_files_generated': 0,
        'ai_api_calls': 0,
        'vector_searches': 0,
        'enhanced': enhanced,
        'hi_res': hi_res,
        'vector_search': vector_search,
        'sequential_pages': sequential_pages,
        'parallel': parallel,
        'smart_memory': smart_memory,
        'llm_config': processing_context.get_config_summary()
    }
    
    startup_time = time.time() - app_start_time
    
    try:
        log_and_print(f"🚀 Processing {input_pdf}...", logger)
        log_and_print(f"📁 Output directory: {output_dir}", logger)
        log_and_print(f"📝 Log file: {log_filename}", logger)
        log_and_print(f"⚡ Startup time: {startup_time:.1f}s", logger)
        log_and_print("", logger)
        
        # Step 1: Parse PDF
        log_and_print("📄 Step 1/4: Parsing PDF document...", logger)
        log_and_print("  ⏳ This may take a while for large or complex PDFs...", logger)
        pdf_start = time.time()
        
        # Use enhanced mode if specified or auto-detected
        document_elements = pdf_parser.parse_pdf(input_pdf, enhanced_mode=enhanced, hi_res=hi_res)
        
        stats['pdf_parsing_time'] = time.time() - pdf_start
        stats['total_elements'] = len(document_elements)
        
        log_and_print(f"  ✓ Extracted {len(document_elements)} elements from PDF ({stats['pdf_parsing_time']:.1f}s)", logger)
        
        # Show element types breakdown
        element_types = {}
        for elem in document_elements:
            elem_type = type(elem).__name__
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        stats['element_types'] = element_types
        
        log_and_print("  📊 Element breakdown:", logger)
        for elem_type, count in sorted(element_types.items()):
            log_and_print(f"    - {elem_type}: {count}", logger)
        log_and_print("", logger)
        
        # Step 2: Generate Outline
        log_and_print("🧠 Step 2/4: Generating document outline...", logger)
        log_and_print("  ⏳ Analyzing document structure with AI...", logger)
        outline_start = time.time()
        
        # Use more elements for better outline generation - aim for ~100K characters
        target_chars = 100000
        outline_elements = []
        current_chars = 0
        
        for elem in document_elements:
            if hasattr(elem, 'text') and elem.text and elem.text.strip():
                elem_text = elem.text.strip()
                if current_chars + len(elem_text) <= target_chars:
                    outline_elements.append(elem)
                    current_chars += len(elem_text)
                else:
                    # Add partial element if it helps reach target
                    remaining_chars = target_chars - current_chars
                    if remaining_chars > 1000:  # Only if meaningful amount
                        partial_elem = type(elem)(elem_text[:remaining_chars] + "...")
                        outline_elements.append(partial_elem)
                    break
        
        outline_text = " ".join([elem.text for elem in outline_elements])
        
        stats['outline_elements_used'] = len(outline_elements)
        stats['outline_characters'] = len(outline_text)
        
        log_and_print(f"  📝 Using {len(outline_elements)} elements ({len(outline_text):,} characters) for analysis...", logger)
        
        # Calculate coverage based on total PDF characters
        total_pdf_chars = sum(len(elem.text) for elem in document_elements if hasattr(elem, 'text') and elem.text)
        coverage_percent = (len(outline_text) / total_pdf_chars * 100) if total_pdf_chars > 0 else 0
        log_and_print(f"  📊 Coverage: {coverage_percent:.1f}% of document content", logger)
        
        outline_data = llm_handler.generate_outline(outline_text)
        stats['ai_api_calls'] += 1
        
        stats['outline_generation_time'] = time.time() - outline_start
        topic = outline_data['topic']
        capabilities = outline_data['capabilities']
        
        stats['topic'] = topic
        stats['capabilities_count'] = len(capabilities)
        stats['capabilities'] = capabilities
        original_capabilities_count = len(capabilities)  # Store before potential expansion
        
        log_and_print(f"  ✓ Generated outline ({stats['outline_generation_time']:.1f}s)", logger)
        log_and_print(f"  📖 Identified topic: '{topic}'", logger)
        log_and_print(f"  🎯 Found {len(capabilities)} capabilities:", logger)
        for i, cap in enumerate(capabilities, 1):
            log_and_print(f"    {i}. {cap}", logger)
        log_and_print("", logger)
        
        # Step 3: Process Each Capability
        log_and_print("🔍 Step 3/4: Extracting features for each capability...", logger)
        processing_start = time.time()
        
        final_docs = {}
        topic_content = []
        
        # Add topic header
        topic_content.append(f"# {topic}")
        topic_content.append("")  # Empty line
        
        # Track features across capabilities to prevent duplication
        existing_features = set()
        
        # Prepare content chunks for all capabilities
        capability_chunks = []
        
        if vector_search:
            # Vector search approach with deduplication
            log_and_print("  🚀 Using vector search with deduplication...", logger)
            vector_init_start = time.time()
            
            def status_callback(msg):
                log_and_print(msg, logger)
            
            # Initialize vector store once for all capabilities
            if smart_memory:
                llm_handler.initialize_vector_store_smart(document_elements, status_callback)
            else:
                llm_handler.initialize_vector_store(document_elements, status_callback)
            vector_init_time = time.time() - vector_init_start
            log_and_print(f"  ✓ Vector optimization complete ({vector_init_time:.1f}s)", logger)
            log_and_print("", logger)
            
            # Prepare vector search chunks
            log_and_print("  📊 Preparing vector search chunks for all capabilities...", logger)
            for capability in capabilities:
                relevant_chunks = llm_handler.get_relevant_chunks(document_elements, capability, use_cache=True)
                capability_chunks.append(relevant_chunks)
                stats['vector_searches'] += 1
                stats['total_relevant_content_chars'] += len(relevant_chunks)
        
        else:
            # Sequential chunking approach (default)
            log_and_print("  📄 Using sequential chunking approach...", logger)
            chunk_start = time.time()
            
            # Create sequential chunks for all capabilities
            capability_chunks = llm_handler.create_sequential_chunks(
                document_elements, len(capabilities), sequential_pages
            )
            chunk_time = time.time() - chunk_start
            log_and_print(f"  ✓ Created {len(capability_chunks)} sequential chunks ({chunk_time:.1f}s)", logger)
            
            # If we have more chunks than capabilities, create additional capability names
            if len(capability_chunks) > len(capabilities):
                additional_capabilities_needed = len(capability_chunks) - len(capabilities)
                log_and_print(f"  📈 Document required {additional_capabilities_needed} additional capabilities to process all content", logger)
                
                # Generate additional capability names
                for i in range(additional_capabilities_needed):
                    additional_cap_name = f"{capabilities[i % len(capabilities)]} (Part {i // len(capabilities) + 2})"
                    capabilities.append(additional_cap_name)
                
                log_and_print(f"  ✅ Total capabilities: {len(capabilities)} (includes {additional_capabilities_needed} additional)", logger)
                
                # Update stats to reflect actual capabilities processed
                stats['capabilities_count'] = len(capabilities)
                stats['capabilities'] = capabilities
                stats['original_capabilities_count'] = original_capabilities_count
                stats['additional_capabilities_created'] = additional_capabilities_needed
            
            for chunk in capability_chunks:
                stats['total_relevant_content_chars'] += len(chunk)
        
        log_and_print("", logger)
        
        # Process capabilities (parallel or sequential)
        if parallel > 0:
            log_and_print(f"  🚀 Processing {len(capabilities)} capabilities in parallel ({parallel} workers)...", logger)
            
            # Prepare tasks for parallel execution
            tasks = []
            for i, (capability, content_chunk) in enumerate(zip(capabilities, capability_chunks)):
                # For parallel processing, we can't maintain stateful deduplication
                # So we pass an empty set for now - this is a trade-off
                task_data = {
                    'capability': capability,
                    'content': content_chunk,
                    'existing_features': set(),  # Empty for parallel processing
                    'process_func': (llm_handler.generate_feature_details_vector_search 
                                   if vector_search 
                                   else llm_handler.generate_feature_details_with_deduplication)
                }
                tasks.append(task_data)
            
            # Execute tasks in parallel
            results = []
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(process_capability_parallel, task): task for task in tasks}
                
                # Process completed tasks with progress bar
                with tqdm(total=len(capabilities), desc=f"Processing capabilities (parallel)") as pbar:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result['success']:
                                log_and_print(f"  ✓ {result['capability']}: {result['generated_chars']:,} chars, {len(result['new_features'])} features ({result['processing_time']:.1f}s)", logger)
                            else:
                                log_and_print(f"  ❌ {result['capability']}: {result['error']}", logger)
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            log_and_print(f"  ❌ Task failed: {str(e)}", logger)
                            pbar.update(1)
            
            # Sort results by original capability order
            capability_order = {cap: i for i, cap in enumerate(capabilities)}
            results.sort(key=lambda x: capability_order[x['capability']])
            
            # Process results and build content
            all_new_features = set()
            for result in results:
                stats['ai_api_calls'] += 1
                stats['total_generated_content_chars'] += result['generated_chars']
                all_new_features.update(result['new_features'])
                
                stats['capability_processing_times'].append({
                    'capability': result['capability'],
                    'chunk_retrieval_time': 0,  # Included in processing time for parallel
                    'ai_generation_time': result['processing_time'],
                    'total_time': result['processing_time'],
                    'relevant_content_chars': result['content_chars'],
                    'generated_content_chars': result['generated_chars'],
                    'new_features_count': len(result['new_features'])
                })
                
                # Add to content
                topic_content.append("---")
                topic_content.append("")
                topic_content.append(f"## {result['capability']}")
                topic_content.append("")
                topic_content.append(result['feature_details'])
                topic_content.append("")
            
            existing_features = all_new_features
            
        else:
            # Sequential processing (original logic with deduplication)
            log_and_print(f"  🔄 Processing {len(capabilities)} capabilities sequentially...", logger)
            
            with tqdm(total=len(capabilities), desc="Processing capabilities (sequential)") as pbar:
                for i, (capability, content_chunk) in enumerate(zip(capabilities, capability_chunks)):
                    pbar.set_description(f"Processing: {capability[:30]}...")
                    
                    # SAFETY: Check resources before each capability
                    try:
                        check_resource_safety()
                    except Exception as e:
                        log_and_print(f"  {str(e)}", logger)
                        exit(1)
                    
                    capability_start_time = time.time()
                    log_and_print(f"  🔄 {i+1}/{len(capabilities)}: {capability}", logger)
                    
                    if vector_search:
                        log_and_print(f"    📊 Using pre-retrieved content: {len(content_chunk):,} characters", logger)
                    else:
                        log_and_print(f"    📄 Processing chunk: {len(content_chunk):,} characters", logger)
                    
                    ai_start = time.time()
                    log_and_print(f"    ⏳ Generating features with AI (avoiding duplicates)...", logger)
                    
                    # Generate feature details with deduplication
                    if vector_search:
                        feature_details, new_features = llm_handler.generate_feature_details_vector_search(
                            capability, content_chunk, existing_features
                        )
                    else:
                        feature_details, new_features = llm_handler.generate_feature_details_with_deduplication(
                            capability, content_chunk, existing_features
                        )
                    
                    ai_time = time.time() - ai_start
                    stats['ai_api_calls'] += 1
                    stats['total_generated_content_chars'] += len(feature_details)
                    
                    # Update existing features set
                    existing_features.update(new_features)
                    
                    log_and_print(f"    ✓ Generated {len(feature_details)} characters, {len(new_features)} new features ({ai_time:.1f}s)", logger)
                    
                    capability_total_time = time.time() - capability_start_time
                    stats['capability_processing_times'].append({
                        'capability': capability,
                        'chunk_retrieval_time': 0,  # Pre-retrieved or included in processing
                        'ai_generation_time': ai_time,
                        'total_time': capability_total_time,
                        'relevant_content_chars': len(content_chunk),
                        'generated_content_chars': len(feature_details),
                        'new_features_count': len(new_features)
                    })
                    
                    # Add to content
                    topic_content.append("---")
                    topic_content.append("")
                    topic_content.append(f"## {capability}")
                    topic_content.append("")
                    topic_content.append(feature_details)
                    topic_content.append("")
                    
                    pbar.update(1)
        
        # Clear the cache to free memory
        llm_handler.clear_cache()
        
        stats['feature_processing_time'] = time.time() - processing_start
        log_and_print(f"  ✓ Completed all capabilities ({stats['feature_processing_time']:.1f}s)", logger)
        log_and_print("", logger)
        
        # Step 4: Assemble and Write Output
        log_and_print("📝 Step 4/4: Writing output files...", logger)
        write_start = time.time()
        
        final_markdown_string = "\n".join(topic_content)
        final_docs[topic] = final_markdown_string
        stats['final_markdown_chars'] = len(final_markdown_string)
        stats['output_files_generated'] = len(final_docs)
        
        log_and_print(f"  📄 Generated {len(final_markdown_string)} characters of markdown content", logger)
        log_and_print(f"  💾 Saving to directory: {output_dir}", logger)
        
        output_writer.write_markdown_files(final_docs, output_dir)
        
        stats['file_writing_time'] = time.time() - write_start
        stats['total_time'] = time.time() - stats['processing_start_time']
        stats['total_app_time'] = time.time() - stats['app_start_time']
        
        log_and_print(f"  ✓ Files written ({stats['file_writing_time']:.1f}s)", logger)
        log_and_print("", logger)
        
        # Print comprehensive statistics
        print_and_log_statistics(stats, logger)
        
    except FileNotFoundError as e:
        error_msg = f"❌ Error: {str(e)}"
        log_and_print(error_msg, logger)
        click.echo(error_msg, err=True)
        exit(1)
    except ValueError as e:
        error_msg = f"❌ Configuration Error: {str(e)}"
        log_and_print(error_msg, logger)
        click.echo(error_msg, err=True)
        exit(1)
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {str(e)}"
        log_and_print(error_msg, logger)
        click.echo(error_msg, err=True)
        exit(1)


def print_and_log_statistics(stats, logger):
    """
    Print and log comprehensive processing statistics.
    
    Args:
        stats (dict): Statistics dictionary
        logger: Logger instance
    """
    log_and_print("=" * 80, logger)
    log_and_print("📊 PROCESSING STATISTICS", logger)
    log_and_print("=" * 80, logger)
    
    # Basic info
    log_and_print(f"📄 Input PDF: {stats['input_pdf']}", logger)
    log_and_print(f"📁 Output Directory: {stats['output_dir']}", logger)
    log_and_print(f"📝 Log File: {stats['log_file']}", logger)
    log_and_print(f"⏰ Start Time: {datetime.fromtimestamp(stats['processing_start_time']).strftime('%Y-%m-%d %H:%M:%S')}", logger)
    log_and_print("", logger)
    
    # Timing breakdown
    startup_time = stats['processing_start_time'] - stats['app_start_time']
    
    log_and_print("⏱️  TIMING BREAKDOWN:", logger)
    log_and_print(f"  🚀 App Startup: {startup_time:.1f}s ({startup_time/stats['total_app_time']*100:.1f}%)", logger)
    log_and_print(f"  📄 PDF Parsing: {stats['pdf_parsing_time']:.1f}s ({stats['pdf_parsing_time']/stats['total_app_time']*100:.1f}%)", logger)
    log_and_print(f"  🧠 Outline Generation: {stats['outline_generation_time']:.1f}s ({stats['outline_generation_time']/stats['total_app_time']*100:.1f}%)", logger)
    log_and_print(f"  🔍 Feature Processing: {stats['feature_processing_time']:.1f}s ({stats['feature_processing_time']/stats['total_app_time']*100:.1f}%)", logger)
    log_and_print(f"  💾 File Writing: {stats['file_writing_time']:.1f}s ({stats['file_writing_time']/stats['total_app_time']*100:.1f}%)", logger)
    log_and_print(f"  ⚡ Processing Time: {stats['total_time']:.1f}s", logger)
    log_and_print(f"  🏁 TOTAL APP TIME: {stats['total_app_time']:.1f}s", logger)
    log_and_print("", logger)
    
    # Content statistics
    log_and_print("📊 CONTENT STATISTICS:", logger)
    log_and_print(f"  📄 Total PDF Elements: {stats['total_elements']}", logger)
    log_and_print(f"  📝 Elements Used for Outline: {stats['outline_elements_used']}", logger)
    log_and_print(f"  📖 Topic Identified: '{stats['topic']}'", logger)
    original_count = stats.get('original_capabilities_count', stats['capabilities_count'])
    additional_count = stats.get('additional_capabilities_created', 0)
    if additional_count > 0:
        log_and_print(f"  🎯 Capabilities Found: {original_count} (expanded to {stats['capabilities_count']} due to document size)", logger)
    else:
        log_and_print(f"  🎯 Capabilities Found: {stats['capabilities_count']}", logger)
    log_and_print(f"  📚 Total Relevant Content Retrieved: {stats['total_relevant_content_chars']:,} characters", logger)
    log_and_print(f"  ✨ Total AI-Generated Content: {stats['total_generated_content_chars']:,} characters", logger)
    log_and_print(f"  📑 Final Markdown Output: {stats['final_markdown_chars']:,} characters", logger)
    log_and_print(f"  📁 Output Files Generated: {stats['output_files_generated']}", logger)
    log_and_print("", logger)
    
    # Element types breakdown
    log_and_print("🏗️  ELEMENT TYPES BREAKDOWN:", logger)
    for elem_type, count in sorted(stats['element_types'].items()):
        percentage = count / stats['total_elements'] * 100
        log_and_print(f"  - {elem_type}: {count} ({percentage:.1f}%)", logger)
    log_and_print("", logger)
    
    # API usage
    log_and_print("🤖 AI API USAGE:", logger)
    log_and_print(f"  📞 Total API Calls: {stats['ai_api_calls']}", logger)
    log_and_print(f"  🔍 Vector Searches: {stats['vector_searches']}", logger)
    avg_chars_per_call = stats['total_generated_content_chars'] / max(stats['ai_api_calls'] - 1, 1)  # -1 for outline call
    log_and_print(f"  📈 Avg Characters Generated per Call: {avg_chars_per_call:.0f}", logger)
    log_and_print("", logger)
    
    # Processing approach
    log_and_print("🔧 PROCESSING APPROACH:", logger)
    if stats.get('vector_search', False):
        log_and_print(f"  📊 Vector Search with Deduplication", logger)
    else:
        log_and_print(f"  📄 Sequential Chunking ({stats.get('sequential_pages', 10)} pages per capability)", logger)
    
    if stats.get('parallel', 0) > 0:
        log_and_print(f"  🚀 Parallel Processing ({stats.get('parallel', 2)} workers)", logger)
        log_and_print(f"  ⚠️  Note: Parallel processing trades some deduplication for speed", logger)
    else:
        log_and_print(f"  🔄 Sequential Processing (full deduplication)", logger)
    
    if stats.get('smart_memory', False):
        log_and_print(f"  🧠 Smart Memory Management (auto-detects optimal strategy)", logger)
    
    # LLM Configuration
    if 'llm_config' in stats:
        log_and_print("🔧 LLM CONFIGURATION:", logger)
        llm_config = stats['llm_config']
        log_and_print(f"  🤖 Model: {llm_config['model_name']}", logger)
        log_and_print(f"  📝 Chunk Size: {llm_config['chunk_size']} chars", logger)
        log_and_print(f"  🔄 Chunk Overlap: {llm_config['chunk_overlap']} chars", logger)
        log_and_print(f"  🎯 Similarity Results: {llm_config['similarity_results']}", logger)
        log_and_print(f"  📊 Context Limit: {llm_config['context_limit']} chars", logger)
    
    log_and_print("", logger)
    
    # Capability processing details
    log_and_print("🎯 CAPABILITY PROCESSING DETAILS:", logger)
    total_new_features = 0
    for i, cap_stats in enumerate(stats['capability_processing_times'], 1):
        new_features = cap_stats.get('new_features_count', 0)
        total_new_features += new_features
        log_and_print(f"  {i}. {cap_stats['capability']}", logger)
        if cap_stats['chunk_retrieval_time'] > 0:
            log_and_print(f"     🔍 Content Retrieval: {cap_stats['chunk_retrieval_time']:.1f}s", logger)
        log_and_print(f"     🤖 AI Generation: {cap_stats['ai_generation_time']:.1f}s", logger)
        log_and_print(f"     ⏱️  Total Time: {cap_stats['total_time']:.1f}s", logger)
        log_and_print(f"     📊 Content: {cap_stats['relevant_content_chars']:,} → {cap_stats['generated_content_chars']:,} chars", logger)
        log_and_print(f"     ✨ New Features: {new_features}", logger)
        log_and_print("", logger)
    
    # Deduplication effectiveness
    log_and_print("🔄 DEDUPLICATION METRICS:", logger)
    log_and_print(f"  ✨ Total Unique Features Generated: {total_new_features}", logger)
    if total_new_features > 0:
        avg_features_per_capability = total_new_features / len(stats['capability_processing_times'])
        log_and_print(f"  📊 Average Features per Capability: {avg_features_per_capability:.1f}", logger)
    log_and_print("", logger)
    
    # Performance metrics
    log_and_print("🚀 PERFORMANCE METRICS:", logger)
    elements_per_sec = stats['total_elements'] / stats['pdf_parsing_time']
    chars_per_sec = stats['final_markdown_chars'] / stats['total_time']
    log_and_print(f"  📄 PDF Elements Processed: {elements_per_sec:.1f} elements/second", logger)
    log_and_print(f"  ✨ Content Generation Rate: {chars_per_sec:.0f} characters/second", logger)
    log_and_print(f"  🎯 Average Time per Capability: {stats['feature_processing_time']/stats['capabilities_count']:.1f}s", logger)
    log_and_print("", logger)
    
    log_and_print("✅ Processing completed successfully!", logger)
    log_and_print("=" * 80, logger)


@cli.command()
@click.option('--pdf', required=True, help='Path to the original PDF file.')
@click.option('--markdown', required=True, help='Path to the generated markdown file.')
@click.option('--report', help='Optional path to save detailed JSON report.')
def validate(pdf, markdown, report):
    """
    Validate generated markdown against the original PDF.
    
    Analyzes coverage, efficiency, and quality metrics to assess
    how well the markdown represents the original PDF content.
    
    Args:
        pdf (str): Path to the original PDF file
        markdown (str): Path to the generated markdown file  
        report (str): Optional path to save detailed JSON report
    """
    try:
        # Import validator (lazy loading)
        print("📊 Loading validation module...")
        import validator
        
        # Generate validation report
        validation_report = validator.generate_validation_report(pdf, markdown, report)
        
        # Print summary
        validator.print_validation_summary(validation_report)
        
    except FileNotFoundError as e:
        click.echo(f"❌ File not found: {str(e)}", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        exit(1)


if __name__ == "__main__":
    cli() 