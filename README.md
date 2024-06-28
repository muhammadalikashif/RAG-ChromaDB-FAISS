# Document Assistant Project

Welcome to the Document Assistant Project! This project leverages LangChain to create an intelligent assistant capable of reading, understanding, and retrieving relevant information from large documents. This README will guide you through the setup and usage of the project.

## Table of Contents

 Introduction
 Features
 Installation
 Usage
 Example
 Contributing
 License

## Introduction

The Document Assistant Project is designed to help users quickly find relevant information within large documents. By using advanced natural language processing (NLP) techniques and machine learning models, the assistant can understand and retrieve contextually relevant text segments in response to user queries.

## Features

 Document Loading: Supports loading of text files and PDFs.
 Text Splitting: Splits large documents into manageable chunks for better processing.
 Embeddings: Uses stateoftheart machine learning models to understand the context and meaning of text.
 Vector Store: Efficiently stores text chunks for quick retrieval.
 Similarity Search: Finds and retrieves the most relevant text chunks based on user queries.

## Installation

To get started with the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/muhammadalikashif/RAG-ChromaDB-FAISS
   cd RAG ChromaDB FAISS
   ```

2. Create a virtual environment:
   ```sh
   python m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install r requirements.txt
   ```

## Usage

To use the Document Assistant, follow these steps:

1. Load Documents:
   Load your text or PDF documents using the provided loaders.

   ```python
   from langchain_community.document_loaders import TextLoader, PyPDFLoader
   
   text_loader = TextLoader("file.txt")
   text_doc = text_loader.load()

   pdf_loader = PyPDFLoader('python.pdf')
   pdf_content = pdf_loader.load()

   combined_documents = text_doc + pdf_content
   ```

2. Split Documents:
   Split the loaded documents into smaller chunks.

   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
   documents = text_splitter.split_documents(combined_documents)
   ```

3. Create Embeddings and Vector Store:
   Create embeddings for the text chunks and store them in a vector store.

   ```python
   from langchain_community.embeddings import OllamaEmbeddings
   from langchain_community.vectorstores import Chroma
   
   embeddings = OllamaEmbeddings(model="llama2uncensored")
   db = Chroma.from_documents(documents, embedding=embeddings)
   ```

4. Query the Assistant:
   Perform a similarity search to retrieve relevant information.

   ```python
   query = 'Python supports multiple programming paradigms, including'
   result = db.similarity_search(query)
   print(result[0].page_content)
   ```

## Example

Here's an example script that demonstrates the complete process:

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load text and PDF documents
text_loader = TextLoader("file.txt")
text_doc = text_loader.load()

pdf_loader = PyPDFLoader('python.pdf')
pdf_content = pdf_loader.load()

# Combine documents
combined_documents = text_doc + pdf_content

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(combined_documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama2uncensored")
db = Chroma.from_documents(documents, embedding=embeddings)

# Query the assistant
query = 'Python supports multiple programming paradigms, including'
result = db.similarity_search(query)
print(result[0].page_content)
```

## Contributing

I welcome contributions to the Document Assistant Project! Please fork the repository and submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License. 

