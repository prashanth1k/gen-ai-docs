# gen-ai-docs

A Python CLI tool that converts PDF documents into RAG-optimized Markdown files using Google's Gemini AI.

## Features

- **Ultra-fast PDF text extraction** (default mode - no external dependencies)
- **Parallel processing** with configurable workers for faster generation
- **Comprehensive content processing** - ensures ALL document content is processed
- **Smart memory management** - automatically handles large documents
- Generate structured outlines automatically using AI
- Extract and organize content by capabilities and features
- Create RAG-optimized markdown with consistent formatting
- Vector-based content retrieval for relevant section extraction
- Optional enhanced parsing modes for better formatting
- **Configurable AI parameters** - model, chunk size, context limits
- **Built-in validation** with detailed quality metrics

## Installation

### Quick Start (Recommended)

1. **Clone or download this repository**

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google Gemini API key:**

   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your Gemini API key
   # Get your API key from: https://aistudio.google.com/
   ```

4. **Edit the `.env` file:**
   ```
   GEMINI_API_KEY="your_actual_api_key_here"
   ```

**That's it!** The default mode requires no external dependencies.

### Enhanced Mode Setup (Optional)

For better formatting and layout detection, you can optionally install additional dependencies:

**Install Poppler (for enhanced parsing only):**

**Windows:**

```bash
# Option 1: Using Chocolatey (recommended)
choco install poppler

# Option 2: Using Conda
conda install -c conda-forge poppler

# Option 3: Manual installation
# Download from: https://github.com/oschwartz10612/poppler-windows/releases/
# Extract to C:\poppler and add C:\poppler\Library\bin to your PATH
```

**macOS:**

```bash
# Using Homebrew
brew install poppler

# Using Conda
conda install -c conda-forge poppler
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install poppler-utils

# Or using Conda
conda install -c conda-forge poppler
```

## Usage

### Default Mode (Ultra-Fast with Parallel Processing)

```bash
# Uses PyPDF2 with 2-worker parallel processing - no external dependencies needed
python main.py process --input-pdf path/to/your/document.pdf
```

### Enhanced Mode (Better Formatting)

```bash
# Uses unstructured library with fast strategy - requires Poppler
python main.py process --input-pdf path/to/your/document.pdf --enhanced
```

### High-Resolution Mode (Best Quality)

```bash
# Uses unstructured library with hi-res strategy - requires Poppler, slowest
python main.py process --input-pdf path/to/your/document.pdf --hi-res
```

### Parallel Processing Options

```bash
# Use 4 parallel workers for faster processing
python main.py process --input-pdf path/to/your/document.pdf --parallel 4

# Disable parallel processing (sequential mode)
python main.py process --input-pdf path/to/your/document.pdf --parallel 0

# Single worker (minimal parallelization)
python main.py process --input-pdf path/to/your/document.pdf --parallel 1
```

### Advanced Configuration

```bash
# Configure AI model and parameters
python main.py process --input-pdf path/to/your/document.pdf \
  --model gemini-1.5-pro \
  --chunk-size 2000 \
  --context-limit 5000

# Enable smart memory management for large documents
python main.py process --input-pdf path/to/your/document.pdf --smart-memory

# Use vector search with deduplication (slower but more targeted)
python main.py process --input-pdf path/to/your/document.pdf --vector-search
```

### Specify Output Directory

```bash
python main.py process --input-pdf path/to/your/document.pdf --output-dir ./my-output
```

### Validate Generated Output

```bash
# Analyze coverage and quality metrics
python main.py validate --pdf path/to/original.pdf --markdown path/to/generated.md

# Generate detailed JSON report
python main.py validate --pdf path/to/original.pdf --markdown path/to/generated.md --report validation_report.json
```

### Help

```bash
python main.py --help
python main.py process --help
python main.py validate --help
```

## Processing Modes

| Mode         | Speed          | Quality   | Dependencies | Parallel      | Use Case                         |
| ------------ | -------------- | --------- | ------------ | ------------- | -------------------------------- |
| **Default**  | ⚡⚡⚡ Fastest | ✓ Good    | None         | ✓ (2 workers) | Quick processing, text-only PDFs |
| **Enhanced** | ⚡⚡ Fast      | ✓✓ Better | Poppler      | ✓ (2 workers) | Better formatting, tables        |
| **Hi-Res**   | ⚡ Slow        | ✓✓✓ Best  | Poppler      | ✓ (2 workers) | Complex layouts, best quality    |

### Processing Strategies

| Strategy                              | Description                                                     | Best For                                |
| ------------------------------------- | --------------------------------------------------------------- | --------------------------------------- |
| **Sequential Chunking** (default)     | Divides document into equal-sized chunks, processes all content | Comprehensive coverage, large documents |
| **Vector Search** (`--vector-search`) | Finds most relevant content per capability using AI embeddings  | Targeted extraction, specific topics    |

### Parallel Processing Benefits

- **60-80% faster** capability processing with 2+ workers
- **Automatic scaling** - creates additional capabilities for large documents
- **Memory safe** - maintains 150K character chunks regardless of document size
- **All content processed** - no content is ever discarded

## How It Works

1. **PDF Parsing**:

   - Default: Uses PyPDF2 for ultra-fast text extraction
   - Enhanced: Uses unstructured library for better formatting
   - Hi-Res: Uses unstructured with highest quality settings

2. **Outline Generation**: Analyzes the first few pages to identify the main topic and capabilities

3. **Content Processing Strategy**:

   - **Sequential Chunking** (default): Divides all content into balanced chunks, ensures comprehensive coverage
   - **Vector Search**: Uses AI embeddings to find most relevant content per capability

4. **Parallel Processing**: Processes multiple capabilities simultaneously using configurable workers

5. **Smart Content Handling**:

   - Automatically creates additional capabilities for large documents
   - Maintains memory-safe chunk sizes (≤150K characters)
   - Guarantees ALL content is processed - nothing is discarded

6. **Feature Extraction**: Uses AI to generate structured features with deduplication

7. **Markdown Generation**: Creates structured markdown with consistent formatting

## Output Format

The tool generates markdown files with this structure:

```markdown
# Main Topic

---

## Capability 1

### Feature: Feature Name

- **Description:** One-sentence summary
- **Key Functionality:**
  - Bullet point 1
  - Bullet point 2
- **Use Case:** Concrete example

---

## Capability 2

...
```

## Requirements

### Minimum Requirements (Default Mode)

- Python 3.8+
- Google Gemini API key
- Internet connection for API calls

### Additional Requirements (Enhanced/Hi-Res Modes)

- **Poppler** (for enhanced PDF parsing)

**Note:** Tesseract OCR is NOT required - all modes work with native PDF text only.

## Troubleshooting

### Common Issues

1. **Default Mode Issues**:

   - `PyPDF2 not installed`: Run `pip install -r requirements.txt`
   - `PDF file not found`: Verify the path to your PDF file
   - Text extraction incomplete: Try `--enhanced` mode for better results

2. **Enhanced Mode Issues**:

   - `Poppler Error`: Install poppler using one of the methods above
   - Restart your terminal/command prompt after installation
   - Verify installation: `pdftoppm -h` should show help text

3. **API Key Error**: Make sure your `.env` file exists and contains a valid Gemini API key

4. **Memory Issues**: For large PDFs, the tool may need more memory due to vector operations

### Error Messages

- `GEMINI_API_KEY not found`: Create/check your `.env` file
- `PDF file not found`: Verify the path to your PDF file
- `Failed to parse JSON response`: The AI response was malformed, try again
- `PyPDF2 not installed`: Run `pip install -r requirements.txt`
- `poppler installed and in PATH`: Only needed for `--enhanced` or `--hi-res` modes

## Performance Tips

### Processing Speed

- **Use default mode** for maximum speed with text-only PDFs
- **Use `--enhanced`** only when you need better table/formatting detection
- **Use `--hi-res`** only for complex layouts or when quality is critical
- Large files (>25MB) process much faster in default mode

### Parallel Processing

- **Default 2 workers** provides good balance of speed and resource usage
- **Increase workers** (`--parallel 4`) for faster processing on powerful machines
- **Reduce workers** (`--parallel 1`) for memory-constrained environments
- **Disable parallel** (`--parallel 0`) for full deduplication accuracy

### Large Documents

- **Smart memory management** (`--smart-memory`) auto-detects optimal strategy
- **Sequential chunking** (default) ensures all content is processed
- **Vector search** (`--vector-search`) for targeted content extraction
- Tool automatically creates additional capabilities for very large documents

### Memory Optimization

- Default chunk size (150K chars) balances quality and memory usage
- Adjust `--chunk-size` and `--context-limit` for fine-tuning
- Use `--smart-memory` for automatic memory management

## Validation & Quality Metrics

The validation feature analyzes how well the generated markdown represents the original PDF:

### Key Metrics

- **Content Coverage**: Percentage of original content preserved in markdown
- **Word Coverage**: Percentage of original words found in the output
- **Key Terms Coverage**: Coverage of important domain-specific terms
- **Heading Coverage**: How well document structure is preserved
- **Structure Completeness**: Quality of markdown formatting (topics, capabilities, features)
- **Compression Ratio**: How much the content was condensed

### Quality Scores

- **Overall Score**: Weighted combination of all metrics (0-100)
- **Content Preservation**: Excellent (80%+), Good (60%+), Fair (40%+), Poor (<40%)
- **Structure Quality**: Based on proper markdown hierarchy and completeness

### Usage

```bash
# Quick validation
python main.py validate --pdf original.pdf --markdown generated.md

# Detailed analysis with JSON report
python main.py validate --pdf original.pdf --markdown generated.md --report metrics.json
```

The validation helps you:

- **Assess conversion quality** - Is important content missing?
- **Choose optimal settings** - Should you use enhanced mode?
- **Compare approaches** - Which processing mode works best for your PDFs?
- **Track improvements** - Measure quality over time

## Recent Improvements

### v2.0 - Comprehensive Processing & Performance

**🚀 Major Performance Improvements:**

- **Parallel processing by default** - 60-80% faster with 2-worker processing
- **Consolidated `--parallel` option** - specify worker count directly (e.g., `--parallel 4`)
- **Smart memory management** - automatic optimization for large documents

**📄 Complete Content Processing:**

- **Fixed content discarding issue** - ALL document content is now processed
- **Automatic capability scaling** - creates additional capabilities for large documents
- **Memory-safe chunking** - maintains optimal chunk sizes regardless of document size
- **No content loss guarantee** - eliminates "characters not processed" messages

**⚙️ Enhanced Configuration:**

- **Configurable AI models** - choose from multiple Gemini models
- **Adjustable parameters** - chunk size, overlap, context limits, similarity results
- **Processing strategy options** - sequential chunking vs. vector search
- **Advanced CLI options** - fine-tune processing for your specific needs

**📊 Improved Validation:**

- **Comprehensive quality metrics** - coverage, structure, key terms analysis
- **Performance tracking** - detailed processing statistics and timing
- **Quality scoring** - overall assessment with actionable recommendations

### Migration from v1.x

**Breaking Changes:**

- `--max-workers` removed - use `--parallel N` instead
- Parallel processing now enabled by default (was opt-in)

**Recommended Updates:**

```bash
# Old v1.x command
python main.py process --input-pdf doc.pdf --max-workers 4

# New v2.0 command
python main.py process --input-pdf doc.pdf --parallel 4
```

## License

This project is open source. Feel free to modify and distribute as needed.
