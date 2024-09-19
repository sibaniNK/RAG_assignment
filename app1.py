


from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

api_key= os.getenv('HF_TOKEN')
groq_token= os.getenv('GROQ_API_KEY')
def pdf_file(file ,question):

    bytes_data = file.read()
    f = open(f"{file.name}.pdf", "wb")
    f.write(bytes_data)
    f.close()
    loader = PyPDFLoader(f"{file.name}.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag-chroma",
                                        embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    template = """Answer the question based only on the following context:
                {context}
                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(temperature=0, groq_api_key=groq_token, model_name="llama-3.1-8b-instant")
    output_parser = StrOutputParser()
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    return chain.invoke(question)


def main():
    st.title("Your Chatbot Title")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["pdf"])
    if uploaded_file is not None:
        # Ask user a question
        question = st.text_input("Ask your question")
    conversation_history = []
    if st.button("Get Answer"):
        # Process CSV file and get answer
        answer = pdf_file(uploaded_file, question)
        conversation_history.append({"user": question, "bot": answer})

        # Display conversation history
        for message in conversation_history:
            st.write(f"User: {message['user']}")
            st.write(f"Bot: {message['bot']}")

    # Run the Streamlit app


if __name__ == "__main__":
    main()






