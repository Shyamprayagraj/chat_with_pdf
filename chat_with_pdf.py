import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = "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"

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Hugging Face API token is missing. Add it to your .env file.")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    pdf_texts = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
        pdf_texts[pdf.name] = text
    return pdf_texts

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store."""
    if not text_chunks:
        raise ValueError("No text chunks available for vectorization.")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Initialize a detailed Hugging Face question-answering pipeline."""
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        use_auth_token=HUGGINGFACE_API_TOKEN,
    )
    return qa_pipeline

def user_input(user_question, pdf_texts):
    """Process user question and return detailed answers."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return

    # Retrieve relevant chunks from the vector store
    docs = new_db.similarity_search(user_question, k=3)  # Retrieve top 3 similar chunks
    if not docs:
        st.write("No relevant context found for your question. Please try rephrasing.")
        return

    # Combine the contents of the top chunks
    combined_context = " ".join([doc.page_content for doc in docs])

    # Display relevant content
    relevant_sections = []
    for doc in docs:
        # Find the exact position of the chunk in the original document
        relevant_sections.append(f"**{doc.page_content}**")

    st.write("### Relevant PDF Sections")
    st.markdown("\n".join(relevant_sections))

    # Initialize and use the Hugging Face QA pipeline
    qa_pipeline = get_conversational_chain()
    try:
        response = qa_pipeline({
            "question": user_question,
            "context": combined_context
        })
        st.write("Reply: ", response["answer"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please try again.")

def main():
    """Streamlit main function."""
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Hugging Face 💁")

    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                    return

                try:
                    # Save the extracted text in session state
                    pdf_texts = get_pdf_text(pdf_docs)
                    st.session_state.pdf_texts = pdf_texts  # Store in session state

                    # Display extracted text
                    st.write("### Extracted PDF Content")
                    for filename, content in pdf_texts.items():
                        st.text_area(f"Extracted Content from {filename}", content, height=200)

                    # Process text and create vector store
                    all_text = " ".join(pdf_texts.values())
                    text_chunks = get_text_chunks(all_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed. You can now ask questions!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

    # Retrieve the stored pdf_texts from session state (if exists)
    pdf_texts = st.session_state.get('pdf_texts', None)

    if pdf_texts:  # Only allow questions if pdf_texts exists
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question, pdf_texts)
    else:
        st.write("Please upload and process PDF files first.")

if __name__ == "__main__":
    main()
