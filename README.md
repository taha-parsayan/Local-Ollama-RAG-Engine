# Ollama and HuggingFace RAG Engine
![Static Badge](https://img.shields.io/badge/LLM-FF0000)
![Static Badge](https://img.shields.io/badge/Python-8A2BE2)
![Static Badge](https://img.shields.io/badge/RAG-8A2BE2)
![Static Badge](https://img.shields.io/badge/Ollama-4CAF50)
![Static Badge](https://img.shields.io/badge/Hugging%20Face%20Transformer-4CAF50)

This repository contains a chatbot implementation leveraging large language models (LLMs) for retrieval-augmented question answering (QA). The system integrates document chunking, embedding generation, and FAISS-based vector search to create a high-performance, context-aware chatbot.

## Features

- **Document Ingestion**: Load and process `.txt` files for retrieval.
- **Chunking**: Split long documents into manageable pieces using `RecursiveCharacterTextSplitter`.
- **Embeddings**: Generate text embeddings using Ollama's embedding model.
- **Vector Search**: Perform similarity-based search using FAISS.
- **LLM QA**: Retrieve relevant context and answer user queries using the `ChatOllama` LLM.
- **Retrieval Augmented Generation (RAG)**: Combine retrieved context with a question-answering template to enhance LLM outputs.

## Installation

### Prerequisites
- Python 3.8+
- Pip

### Setup
1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv chatbot
    # On Windows
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\chatbot\Scripts\Activate.ps1
    # On macOS/Linux
    source chatbot/bin/activate
    ```

3. Install the required packages:
    ```bash
    python -m pip install --upgrade pip
    pip install llama-index-llms-openai
    pip install llama-index-llms-huggingface
    pip install llama-index-llms-huggingface-api
    pip install "transformers[torch]" "huggingface_hub[inference]"
    ```


## Usage

1. **Environment Setup**:
   - Update environment variables as needed in the `.env` file.
   - Suppress warnings with built-in functionality.

2. **Prepare Data**:
   - Place `.txt` files in the `data/` directory.

3. **Run the Chatbot**:
    ```bash
    python chatbot.py
    ```

   - Type your questions when prompted.
   - Enter `exit` to terminate the session.

### Directory Structure
```
<repository-name>/
├── data/                 # Directory for .txt files
├── chatbot.py            # Main script
├── requirements.txt      # Additional dependencies
├── README.md             # Project documentation
└── .env                  # Environment variables
```

## Configuration

Modify the following components as needed:

- **Document Chunking**:
  Adjust chunk size and overlap in the `chunk_documents` function.

- **Embedding Model**:
  Update the `OllamaEmbeddings` initialization to use a different model or base URL.

- **Prompt Template**:
  Customize the QA prompt in the `ChatPromptTemplate` object.

- **Retriever Parameters**:
  Tune retriever parameters such as `k`, `fetch_k`, and `lambda_mult` for performance.

## Key Dependencies

- [LangChain](https://docs.langchain.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Ollama Embeddings](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the [LangChain](https://www.langchain.com/) community and Ollama's LLM embeddings.
- Thanks to Hugging Face for their open-source tools.
