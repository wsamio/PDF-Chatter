import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_question, response["answer"]))

    history =  st.session_state.chat_history[::-1]
    for message in history:
        st.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatter")

    st.write(css, unsafe_allow_html =  True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("PDF Chatter :robot_face:")
    user_question = st.text_input("Ask questions about the ingested pdf")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Ingested Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Ingest'", accept_multiple_files = True)

        if st.button("Ingest"):
            with st.spinner("ingesting..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
