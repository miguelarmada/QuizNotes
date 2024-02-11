import streamlit as st #For the GUI
from langchain_community.vectorstores import FAISS #Store the embeddings locally with FAISS
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
import os

def load_llm():
    # Load the Hugging Face model from the Hub
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={'max_new_tokens': 1048, 'temperature': 0.0}
    )
    return llm

def file_processing(file_path):
    question_gen = ''

    # Extract text from each page of the PDF
    for pdf in file_path:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            question_gen += page.extract_text()

    # Split text into chunks using RecursiveCharacterTextSplitter
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    # Create a list of documents from the text chunks
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # Split documents into smaller chunks for better processing
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    # Load the language model for question generation
    llm_ques_gen_pipeline = load_llm()

    # Define prompt templates for question generation
    prompt_template = """
    You are an expert at creating questions based on materials and documentation.
    You do this by asking questions about the text below:

    {text}

    Create a quiz that will help study for their tests.
    Make sure not to lose any important information.
    QUESTIONS:
    """
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at creating practice questions based on material and documentation.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones (only if necessary) with some more context below.

    {text}

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Load the summarization chain for question generation
    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=False,
                                          question_prompt=PROMPT_QUESTIONS,
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Invoke the question generation chain on the documents
    ques = ques_gen_chain.invoke(document_ques_gen)

    # Load embeddings model for question-answer pairing
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create a vector store using FAISS
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Load the language model for answer generation
    llm_answer_gen = load_llm()

    # Split the generated questions into a list
    ques_list = ques["output_text"].split("\n")

    # Filter out unique questions ending with '?'
    filtered_ques_list = np.unique([element for element in ques_list if element.endswith('?')])

    # Create a retrieval question-answering chain
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever())

    # Iterate over filtered questions, generate answers, and display them
    for question in filtered_ques_list:
        answer = answer_generation_chain.invoke(question)
        st.text(question)
        with st.expander("Press to reveal answer:"):
            st.info(answer["result"].split("Helpful Answer:")[1])

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Quiz with your class notes", page_icon=":books:")

    # Display the main header
    st.header("Start your quiz!")

    # Create a sidebar for user input
    with st.sidebar:
        st.subheader("Enter your HuggingFace Hub API Key:")
        HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your api key", type='password')
        st.subheader("Drop your class notes:")
        pdf_notes = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        button = st.button("Quiz me!")

    # Check if the button is pressed and the API key is provided
    if button and HUGGINGFACEHUB_API_TOKEN:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
        with st.spinner("Crafting the quiz"):
            file_processing(pdf_notes)

if __name__ == '__main__':
    main()
