import streamlit as st

def main():
    st.set_page_config(page_title="PDF Chatter")

    st.header("PDF Chatter")
    st.text_input("Ask questions about the ingested pdf")

    with st.sidebar:
        st.subheader("Ingested Documents")
        st.file_uploader("Upload your PDFs here and Click on 'Ingest'")
        st.button("Ingest")


if __name__ == '__main__':
    main()
