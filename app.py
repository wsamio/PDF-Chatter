import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function =  len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatter")

    st.header("PDF Chatter")
    st.text_input("Ask questions about the ingested pdf")

    with st.sidebar:
        st.subheader("Ingested Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Ingest'", accept_multiple_files=True)

        if st.button("Ingest"):
            with st.spinner("ingesting"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store


if __name__ == '__main__':
    main()
