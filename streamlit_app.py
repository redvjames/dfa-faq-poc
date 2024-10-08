__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# Show title and description.
st.title("📄 DFA Q&A")
st.write(
    "This app answers questions based on FAQs found [here](https://consular.dfa.gov.ph/faqs-menu?). "
)

# streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

question = st.text_area(
    "Enter your email text here!",
    placeholder="""Dear DFA,

My name is Juan Dela Cruz. How can I apply for a passport?

Thank you!
    """,
    height=200
)

from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_core.output_parsers import StrOutputParser


if question:
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)
    # llm2 = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

    # st.write(question)

    from dotenv import load_dotenv
    load_dotenv()

    from prompts import prompt, dfa_rag_prompt

    # RETRIEVER 
    CHROMA_PATH = "chroma"
    n_retrieved_docs = 5

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever =  db.as_retriever(search_kwargs={'k': n_retrieved_docs})

    def format_docs(docs):
        return f"\n\n".join(f"[FAQ]" + doc.page_content.replace("\n", " ") for n, doc in enumerate(docs, start=1))

    chain = prompt | llm | {"context": retriever | format_docs, "question": RunnablePassthrough()} | dfa_rag_prompt | llm

    input_dict = {"question": question}

    response = chain.invoke(input_dict)

    st.write(response)