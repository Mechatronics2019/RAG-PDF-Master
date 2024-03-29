# Retrieval-Augmented-Generation (RAG) PDF Master

PDF Master is a Python application designed to provide intelligent insights from PDF documents using state-of-the-art AI models. Leveraging Retrieval-Augmented Generation (RAG) framework, this application integrates LangChain, Chroma DB, and OpenAI's advanced models for text embedding and generation. The user interface is built using Streamlit, offering a seamless experience for interacting with PDF documents and asking questions about their content.

# How It Works
PDF Master follows a simple workflow to assist users in extracting information from PDF documents:

### 1)Upload PDF Documents:
Users can upload one or multiple PDF files containing the content they want to analyze.

### 2)Ask Questions: 
After uploading the documents, users can ask questions related to the content of the PDF files.

### 3)Generate Answers: 
The application processes the PDF documents, splits the text into manageable chunks and identifies relevant information based on the user's question. It then employs LangChain and OpenAI models to generate precise answers to the questions asked.

## Note 
Users need to have their OpenAi Key in order for this to work.

# Installation
Install dependencies.
pip install -r requirements.txt