import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from prompts import prompt, dfa_rag_prompt
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough

# Load OpenAI API Key: This should be kept in the ~/.streamlit/secrets.toml file
openai_api_key = st.secrets['openai']['api_key']
hf_api_key = st.secrets['huggingface']['api_key']


def main():
    title = "<h1 style='font-size:50px; color:white'>Welcome to the DFA Chatbot!</h1>"
    ask = "<p style='font-size:25px'>What is your question?</p>"
    st.markdown(title, unsafe_allow_html=True)
    st.markdown(ask, unsafe_allow_html=True)
    user_question = st.text_input(label="", placeholder="I lost my passport. What do I do?")
    if user_question:
        response = generate_response(user_question)
        st.info(response)
    else:
        st.warning("Please enter a question.", icon='⚠')


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    documents = loader.load()

    pdf_loader = PyPDFDirectoryLoader(DATA_PATH, glob="**/*.pdf", recursive=True)
    pdf_documents = pdf_loader.load()
    return [*documents, *pdf_documents]


def split_text(documents: list[Document]):
    text_splitter = CharacterTextSplitter(
        separator='}',
        chunk_size=50,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()


def generate_response(user_question):
    try:
	# The LLM model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, token=hf_api_key)
	
	# Where the relevant docs will be stored, and how many
        CHROMA_PATH = 'chroma'
        n_retrieved_docs = 10

	# Embedding function
        embedding_function = OpenAIEmbeddings(api_key=openai_api_key)

	# Initialize Chroma and set up the retriever
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        retriever = db.as_retriever(search_kwargs={'k': n_retrieved_docs})

	# Formatting the retrieved docs
        def format_docs(docs):
            return f"\n\n".join(f"[FAQ]" + doc.page_content.replace("\n", " ") for n, doc in enumerate(docs, start=1))
        chain = prompt | llm | {"context": retriever | format_docs, "question": RunnablePassthrough()} | dfa_rag_prompt | llm
	
	# Input dictionary
        input_dict = {"question": user_question}
	
	# Response
        return chain.invoke(input_dict)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None


if __name__ == "__main__":
    main()






