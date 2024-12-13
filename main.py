'''
https://www.youtube.com/watch?v=WL7V9JUy2sE&list=PLVEEucA9MYhNrD8TBI5UqM6WHPUlVv89w

https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/

python.exe -m pip install --upgrade pip
python -m venv chatbot
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\packages\Scripts\Activate.ps1

pip install llama-index-llms-openai
pip install llama-index-llms-huggingface
pip install llama-index-llms-huggingface-api
pip install "transformers[torch]" "huggingface_hub[inference]"
'''

#%%
import os
import warnings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document  # Import Document class
import faiss
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


#%%
def initialize_environment():
    """Set up environment variables and suppress warnings."""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")
    load_dotenv()

def load_txts_from_directory(directory):
    """Load all .txt files from the specified directory."""
    txts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    txts.append(f.read())
    return txts

def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(" ".join(docs))

def create_vector_store(embedding_function, dimensions):
    """Create and return a FAISS vector store."""
    index = faiss.IndexFlatL2(dimensions)
    return FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#%%

def main():
    # Initialize environment
    initialize_environment()

    # Load .txt files and extract their text
    txt_directory = 'data'
    texts = load_txts_from_directory(txt_directory)

    # Chunk the documents
    chunks = chunk_documents(texts)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    embedding_dimensions = len(embeddings.embed_query("test query"))  # Determine embedding dimensions
    vector_store = create_vector_store(embedding_function=embeddings, dimensions=embedding_dimensions)

    # Convert chunks into Document objects and add to vector store
    documents = [Document(page_content=chunk) for chunk in chunks]
    document_ids = vector_store.add_documents(documents=documents)


    model = ChatOllama(model="mistral", base_url="http://localhost:11434")

    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Question: {question} 
        Context: {context} 
        Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break;

        #results = vector_store.search(query=question, search_type='similarity')

        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 5, 'fetch_k': 50, 'lambda_mult': 0.5})

        docs = retriever.invoke(question)


        rag_chain = (
            {"context": retriever|format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        output = rag_chain.invoke(question)
        print(output)

if __name__ == "__main__":
    main()
        # %%
