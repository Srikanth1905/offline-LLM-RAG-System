import os
import streamlit as st
import time
import logging
import warnings
from utils import track_processing_time
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, JSONLoader, CSVLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@track_processing_time
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2:1b", verbose=True, temperature=0)

@track_processing_time
@st.cache_data(show_spinner=False)
def process_documents(directory):
    
    jq_schema = "{content: .content, metadata: .metadata, text_content: .text_content}"
    start = time.time()
    all_docs = []
    
    # Load PDFs - enhanced with better metadata tracking
    pdf_loader = DirectoryLoader(
        directory, 
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader, 
        use_multithreading=True, 
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    # Add document identifier to help with debugging
    for i, doc in enumerate(pdf_docs):
        doc.metadata['doc_id'] = f"pdf_{i}"
        doc.metadata['doc_type'] = "pdf"
    all_docs.extend(pdf_docs)
    logger.info(f"PDFs loaded: {len(pdf_docs)} in {time.time() - start:.2f} seconds")
    
    # Load embeddings
    Embedding_time = time.time()
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2", show_progress=True)
    logger.info(f"Embedding Model {st.session_state.embeddings} loaded in {time.time() - Embedding_time:.2f} seconds")
    
    # Load other document types with enhanced metadata
    json_loader = DirectoryLoader(directory, glob="**/*.json", loader_cls=JSONLoader, loader_kwargs={"jq_schema": jq_schema, "text_content": False})
    json_docs = json_loader.load()
    for i, doc in enumerate(json_docs):
        doc.metadata['doc_id'] = f"json_{i}"
        doc.metadata['doc_type'] = "json"
    all_docs.extend(json_docs)
    
    csv_loader = DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs={"encoding": "utf-8"})
    csv_docs = csv_loader.load()
    for i, doc in enumerate(csv_docs):
        doc.metadata['doc_id'] = f"csv_{i}"
        doc.metadata['doc_type'] = "csv"
    all_docs.extend(csv_docs)
    
    excel_loader = DirectoryLoader(directory, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader, loader_kwargs={"mode": "elements"})
    excel_docs = excel_loader.load()
    for i, doc in enumerate(excel_docs):
        doc.metadata['doc_id'] = f"excel_{i}"
        doc.metadata['doc_type'] = "excel"
    all_docs.extend(excel_docs)
    
    docx_loader = DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader)
    docx_docs = docx_loader.load()
    for i, doc in enumerate(docx_docs):
        doc.metadata['doc_id'] = f"docx_{i}"
        doc.metadata['doc_type'] = "docx"
    all_docs.extend(docx_docs)
    
    logger.info(f"Total documents loaded: {len(all_docs)} in {time.time() - start:.2f} seconds")
    
    if not all_docs:
        raise ValueError(f"No documents were loaded from {directory}. Please check the files.")
        
    # Improved document splitting - smaller chunks with more overlap for better cross-document retrieval
    split_start = time.time()
    st.session_state.docs = all_docs
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    split_end = time.time() - split_start
    logger.info(f"Created {len(st.session_state.final_documents)} splits from {len(st.session_state.docs)} documents in {split_end:.2f} seconds")

    # Create vector store
    logger.info("Creating Vector Store:.....")
    vector_time = time.time()
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    logger.info(f"Embeddings created in {time.time() - vector_time:.2f} seconds")
    
    # Track document statistics
    doc_sources = {}
    for doc in st.session_state.docs:
        source = doc.metadata.get('source', 'Unknown')
        doc_sources[source] = doc_sources.get(source, 0) + 1
    
    st.session_state.doc_statistics = {
        "total_docs": len(st.session_state.docs),
        "total_chunks": len(st.session_state.final_documents),
        "doc_sources": doc_sources
    }
    
    st.session_state.current_directory = directory

@track_processing_time
def setup_rag_chain(directory):
    
    if ("current_directory" not in st.session_state or 
        st.session_state.current_directory != directory or 
        "vectors" not in st.session_state):
        process_documents(directory)
    
    # Improved retriever with higher k value for better coverage across multiple documents
    retriever = st.session_state.vectors.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 20, "lambda_mult": 0.5}
    )

    system_prompt = """You are an expert document analysis assistant tasked with answering questions based on information from multiple documents. Follow these guidelines:

1. ONLY use information from the retrieved context.
2. DO NOT add any information from external sources or general knowledge.
3. If the context contains information from multiple documents, synthesize a comprehensive answer that incorporates relevant details from ALL documents.
4. For each key point in your answer, mention which document it came from if that information is available in the context.
5. If the documents contain conflicting information, present both viewpoints and indicate the source.
6. If the information is incomplete or not found in the context, state: "The provided documents don't contain sufficient information about this specific question."
7. Organize your response logically, especially when synthesizing information from multiple sources.

Context:
{context}

Question: {input}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(system_prompt)
    llm = get_llm()
    
    # For streaming, we'll use a different approach - creating a streaming-compatible chain
    return retriever, prompt, llm

@track_processing_time
def get_answer(query, directory):
    try:
        retriever, prompt, llm = setup_rag_chain(directory)
        
        # Get retrieved documents for debugging display
        retrieved_docs = retriever.invoke(query)
        
        with st.expander("Retrieved Documents (Debugging)"):
            st.write(f"Number of retrieved documents: {len(retrieved_docs)}")
            doc_sources = {}
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                doc_sources[source] = doc_sources.get(source, 0) + 1
                st.write(f"Document {i}:")
                st.write(f"Source: {source}")
                st.write(f"Doc ID: {doc.metadata.get('doc_id', 'Unknown')}")
                st.write(f"Content Length: {len(doc.page_content)}")
                st.write(f"Preview: {doc.page_content[:300]}...")
                st.divider()
            
            st.write("Documents by source:", doc_sources)
            
            if not retrieved_docs:
                return "I cannot find any relevant information in the provided documents."
        
        # Build streaming chain
        def format_docs(docs):
            return "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
                               for i, doc in enumerate(docs)])
        
        # Create a streaming chain
        rag_chain_streaming = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Return a streaming generator
        return rag_chain_streaming.stream(query)
    
    except Exception as e:
        st.error(f"Error in get_answer: {str(e)}")
        logger.error(f"Error in get_answer: {str(e)}", exc_info=True)
        raise