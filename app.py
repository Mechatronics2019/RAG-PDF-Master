import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate, prompt
from langchain_community.chat_models import ChatOpenAI


# Setting up the Streamlit page configuration 
st.set_page_config(page_title='PDF Master')
st.markdown("""
### PDF Master: Intelligent PDF Query Assistant:book::closed_book::page_facing_up:	:memo:

PDF Master harnesses cutting-edge AI technology to provide instant insights from your PDF documents. Utilizing the advanced Retrieval-Augmented Generation (RAG) framework, PDF Master integrates LangChain, Chroma DB and OpenAI's state-of-the-art Generative AI models. 

#### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your OpenAI API Key**: You'll need an OpenAI API key for the chatbot to access OpenAI's Generative AI models.

2. **Upload Your Documents**: The system accepts multiple PDF files at a time, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the document, ask any question related to its content for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
with st.sidebar:
    st.header("Step 1")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="openai_api_key_input")
CHROMA_PATH = "chromaDB"

# Function to extract text from a PDF file
def pdf_to_text(pdf): 
    pdf_reader=PdfReader(pdf)
    text=''
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text 

# Function to split text into chunks
def text_split_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True)
    chunks=text_splitter.split_text(text)
    return chunks

# Function to save text chunks to Chroma DB
def save_to_chromadb(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)
    db = Chroma.from_texts(chunks, embeddings,persist_directory=CHROMA_PATH)
    db.persist()

# Function to process multiple PDF files into text and save them into Chroma DB
def multiple_pdfFiles_to_text(pdf_files):
    for pdf in pdf_files:
        text=pdf_to_text(pdf)
        chunks=text_split_into_chunks(text)
        save_to_chromadb(chunks)

# Function to retrieve the question-answering chain
def getting_chain():
    prompt_template = """
        Answer  the question using only the details provided in the context.If the necessary information is not available, 
        simply state "answer is not available in the context" instead of providing incorrect information.
        
        Context:\n {context}?\n
        Question: \n{question}\n
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  
    llm = ChatOpenAI(model='gpt-3.5-turbo',api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt)
    return chain

# Function to provide response to user's question
def user_query_response(question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)
    db=Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    docs = db.similarity_search(question)
    # st.write(docs)
    chain=getting_chain()
    response = chain.run(input_documents=docs, question=question)
    st.header('Answer')
    st.write(response)

# Main function to control the flow of the Streamlit app
def main():
    with st.sidebar:
        st.header('Step 2')
        pdf_files =st.file_uploader('Upload your pdf',type='pdf',accept_multiple_files=True)
        st.header('Step 3')
        user_question = st.text_input("Ask a question about your PDF:")
        
    if pdf_files is not None:  # Process PDF files if uploaded
        multiple_pdfFiles_to_text(pdf_files)

    if user_question:  # Respond to user's question if provided
        user_query_response(user_question) 

if __name__ == '__main__':
    main()
