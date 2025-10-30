from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st

@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Initializes the FAISS retriever and LLM pipeline."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.load_local("esg_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

def answer_query(question: str) -> str:
    """Runs the query through the QA pipeline."""
    qa = init_pipeline()
    response = qa.run(question)
    return response
