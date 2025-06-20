### **Project Brief for AI Coding Agent**

**Objective:** Create a Python Command-Line Interface (CLI) tool named `gen-ai-docs`. This tool will ingest one or more PDF documents, use the Google Gemini API to understand and restructure the content, and output one or more RAG-optimized Markdown files according to a specific, structured format.

**Core Philosophy:**

- **Modularity:** Code must be highly modular and organized into separate files based on function (e.g., `pdf_parser.py`, `llm_handler.py`). This is non-negotiable.
- **Efficiency:** Use modern, fast libraries. Leverage libraries to minimize custom code, especially for complex tasks like PDF parsing, text chunking, and vector operations.
- **User Experience:** The CLI must be user-friendly, providing clear progress indicators using `tqdm` so the user knows the application is working and not frozen. It must handle potential API errors gracefully.
- **Configuration:** All sensitive information (API keys) must be managed via a `.env` file, not hardcoded.

---

### **1. Technology Stack & Dependencies**

You will use the following Python libraries. Create a `requirements.txt` file with these contents:

```
# requirements.txt
click
google-generativeai
python-dotenv
unstructured[pdf]
langchain-google-genai
langchain-community
faiss-cpu
tqdm
```

- **`click`**: For creating a clean, modern CLI.
- **`google-generativeai`**: The official Google SDK for the Gemini API.
- **`python-dotenv`**: To load environment variables from a `.env` file.
- **`unstructured[pdf]`**: State-of-the-art library for parsing complex PDFs while preserving layout information.
- **`langchain-google-genai`**: LangChain integration for Gemini embeddings.
- **`langchain-community`**: For vector stores (FAISS) and other utilities.
- **`faiss-cpu`**: For fast, in-memory vector similarity searches.
- **`tqdm`**: For creating user-friendly progress bars in the CLI.

---

### **2. Project Structure**

Organize the project into the following file structure:

```
gen-ai-docs/
├── .env                  # For API key and other configurations
├── requirements.txt      # List of project dependencies
├── main.py               # Main CLI application entry point and workflow orchestration
├── config.py             # Handles loading configuration and initializing API clients
├── pdf_parser.py         # Contains all logic for parsing PDF files
├── llm_handler.py        # Contains all functions that interact with the Gemini API
└── output_writer.py      # Contains logic for writing the final Markdown files
```

---

### **3. Detailed Task Breakdown (File by File)**

Implement the logic for each file as described below.

#### **A. `config.py`**

- **Purpose:** Centralize configuration and API client initialization.
- **Tasks:**
  1.  Import `os` and `dotenv`.
  2.  Use `dotenv.load_dotenv()` to load the `.env` file.
  3.  Define a variable `GEMINI_API_KEY` by fetching it from the environment variables.
  4.  If the API key is not found, raise a `ValueError` with an informative message telling the user to create a `.env` file.
  5.  Import `google.generativeai as genai`.
  6.  Configure the `genai` client using `genai.configure(api_key=GEMINI_API_KEY)`.

#### **B. `pdf_parser.py`**

- **Purpose:** Abstract the PDF parsing logic.
- **Tasks:**
  1.  Import `unstructured.partition.pdf`.
  2.  Define a function `parse_pdf(pdf_path: str) -> list`.
  3.  Inside this function, use `partition_pdf(filename=pdf_path, strategy="hi_res")`. The "hi_res" strategy is better for complex documents.
  4.  The function should return the list of `Element` objects produced by `unstructured`.
  5.  Include basic error handling for `FileNotFoundError`.

#### **C. `llm_handler.py`**

- **Purpose:** All interactions with the Gemini models. This is the core intelligence of the application.
- **Tasks:**

  1.  Import `genai` from `google.generativeai`.
  2.  Import `GeminiEmbeddings` from `langchain_google_genai`.
  3.  Import `FAISS` from `langchain_community.vectorstores`.
  4.  Import `RecursiveCharacterTextSplitter` from `langchain.text_splitter`.
  5.  Import `json`.

  6.  **Function 1: `generate_outline(full_text: str) -> dict`**

      - **Purpose:** To create the high-level structure (`# Topic` and `## Capabilities`).
      - **Logic:**

        - Instantiate the Gemini Pro model: `model = genai.GenerativeModel('gemini-pro')`.
        - Define the prompt. **Use this exact prompt template**:

          ```python
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
          ```

          _(Note: Use a slice of the text, like the first 12,000 characters, to represent the "first few pages" and stay within context limits.)_

        - Make the API call: `response = model.generate_content(prompt)`.
        - Extract the JSON string from `response.text`. It may be wrapped in markdown backticks, so clean it.
        - Parse the cleaned string into a Python dictionary using `json.loads()`.
        - Return the dictionary. Implement `try...except` for the JSON parsing step.

  7.  **Function 2: `get_relevant_chunks(documents: list, query: str) -> str`**

      - **Purpose:** To find the most relevant sections of the PDF for a given capability.
      - **Logic:**
        - Initialize embeddings: `embeddings = GeminiEmbeddings(model="models/embedding-001")`.
        - Initialize a text splitter: `text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`.
        - Create text chunks from the `unstructured` elements: `chunks = text_splitter.split_text(" ".join([doc.text for doc in documents]))`.
        - Create a FAISS vector store from the chunks: `vector_store = FAISS.from_texts(chunks, embeddings)`.
        - Perform a similarity search: `results = vector_store.similarity_search(query, k=5)`.
        - Concatenate the text from the result documents and return it as a single string.

  8.  **Function 3: `generate_feature_details(capability_name: str, context_text: str) -> str`**

      - **Purpose:** To generate the detailed `### Feature` blocks.
      - **Logic:**

        - Instantiate the model: `model = genai.GenerativeModel('gemini-pro')`.
        - Define the prompt. **Use this exact prompt template**:

          ```python
          prompt = f"""
          You are an expert technical writer converting a user manual into a structured, RAG-friendly format.
          Your current task is to detail the features related to the high-level capability: "## {capability_name}".

          From the provided text below, identify specific features. For each feature, you MUST generate:
          1. A '### Feature:' header with the feature name.
          2. A '- **Description:**' with a one-sentence summary.
          3. A '- **Key Functionality:**' section with a bulleted list of what the feature does.
          4. A '- **Use Case:**' section with a concrete example of the feature in action.

          Adhere strictly to this format. If you cannot find information for a section (e.g., Use Case), write 'Not specified in the document.'.
          Do not invent information. Structure the entire output for this capability in valid Markdown.
          Do not add the '## {capability_name}' header yourself; only provide the '### Feature' blocks.

          --- CONTEXT TEXT ---
          {context_text}
          --- END TEXT ---
          """
          ```

        - Make the API call and return `response.text`.

#### **D. `output_writer.py`**

- **Purpose:** To write the generated content to files.
- **Tasks:**
  1.  Import `os`.
  2.  Define a function `write_markdown_files(structured_data: dict, output_dir: str)`.
  3.  The `structured_data` will be a dictionary where keys are topic names and values are the full markdown content for that topic.
  4.  Create the output directory if it doesn't exist using `os.makedirs(output_dir, exist_ok=True)`.
  5.  Loop through the `structured_data` items.
  6.  For each topic, generate a filename like `topic.lower().replace(' ', '_') + '.md'`.
  7.  Construct the full output path and write the markdown content to the file.

#### **E. `main.py`**

- **Purpose:** The main entry point that orchestrates the entire process.
- **Tasks:**
  1.  Import `click` and `tqdm`.
  2.  Import all necessary functions from your other modules (`config`, `pdf_parser`, `llm_handler`, `output_writer`).
  3.  Set up the CLI using `click` decorators.
      - Create a main command group: `@click.group()`.
      - Create a `process` command: `@cli.command()`
      - Add options for input and output: `@click.option('--input-pdf', required=True, help='Path to the input PDF file.')` and `@click.option('--output-dir', default='./output', help='Directory to save the output Markdown files.')`.
  4.  Implement the `process(input_pdf, output_dir)` function. This is the main workflow:
      - Print a status message: `print(f"Processing {input_pdf}...")`
      - **Step 1: Parse PDF.** Call `pdf_parser.parse_pdf(input_pdf)` to get the document elements.
      - **Step 2: Generate Outline.**
        - Concatenate text from the first ~15 elements for the outline generation.
        - Call `llm_handler.generate_outline()`.
        - Store the result (e.g., `outline_data`).
      - **Step 3: Process Each Capability.**
        - Initialize an empty dictionary to hold the final markdown content: `final_docs = {}`.
        - Initialize an empty list to build the content for the current topic: `topic_content = []`.
        - Add the topic header and summary to `topic_content`.
        - Get the list of capabilities from `outline_data['capabilities']`.
        - **Create a `tqdm` progress bar for iterating over the capabilities.** `for capability in tqdm(capabilities, desc="Extracting Features"):`
        - **Inside the loop:**
          a. Call `llm_handler.get_relevant_chunks()` with the full document elements and the current `capability` name.
          b. Call `llm_handler.generate_feature_details()` with the `capability` name and the relevant chunks.
          c. Append the result to `topic_content` in the correct format (`\n---\n\n## {capability}\n{feature_details}`).
      - **Step 4: Assemble and Write Output.**
        - Join all parts in `topic_content` into a single string.
        - Store it in your final dictionary: `final_docs[outline_data['topic']] = final_markdown_string`.
        - Call `output_writer.write_markdown_files(final_docs, output_dir)`.
      - Print a final success message: `print(f"✅ Successfully processed and saved output to '{output_dir}'.")`.
  5.  Create the standard Python entry point check: `if __name__ == "__main__": cli()`.

---

### **Final Instructions**

1.  Begin by creating the `requirements.txt` file.
2.  Next, create the `.env.example` file with `GEMINI_API_KEY="YOUR_API_KEY_HERE"`. The user will rename this to `.env`.
3.  Implement each Python module (`config.py`, `pdf_parser.py`, etc.) one by one, following the specifications precisely.
4.  Finally, implement `main.py` to tie all the modules together into a functioning CLI application.
5.  Ensure all public functions have docstrings explaining their purpose, arguments, and return values.
