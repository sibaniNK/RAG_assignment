# RAG_assignment
This project is a PDF Question-Answering Chatbot built using Langchain and Streamlit. The app allows users to upload a PDF document and ask questions about its content. By leveraging document chunking, vector-based retrieval, and a large language model (LLM), the chatbot provides accurate and context-aware answers based on the content of the uploaded PDF.

The system splits the document into manageable chunks, generates embeddings using a Hugging Face model, and stores them in a Chroma vector store for efficient retrieval. A ChatGroq model is then used to generate human-like responses. The app is user-friendly, featuring an interactive web interface that lets users easily upload files and ask questions.

The application demonstrates how Natural Language Processing (NLP) and retrieval-augmented generation (RAG) can be combined to create smart document assistants capable of answering questions based on large unstructured documents.

## Installation
pip install -r requirements.txt
