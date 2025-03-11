import os
import streamlit as st
import time
from utils import setup_logging, track_processing_time, clear_documents, handle_uploaded_files
from document_processor import process_documents, get_answer
from audio_manager import speak_text, recognize_speech
from pathlib import Path
import logging

logger = setup_logging()

# Constants
LOCAL_DOCS_DIR = "./data"
TEMP_AUDIO_DIR = "./temp_audio"
UPLOAD_DIR = "./uploaded_data"

# Page configuration
st.set_page_config(page_title="LLM RAG Assistant", page_icon="🤖", layout="wide")
st.title("LLM RAG Assistant 🤖")

@track_processing_time
def warm_up_model():
    from ollama import chat
    # Simple warm-up query
    chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': 'Hello'}],
    )

def main():
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)

    if 'model_warmed' not in st.session_state:
        with st.spinner("Initializing model..."):
            warm_up_model()
            st.session_state.model_warmed = True
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("Document Processing")
        doc_source = st.radio("Choose document source:", ["Local Directory", "Upload Files"], help="Select whether to process documents from local directory or upload new files")
        
        current_directory = LOCAL_DOCS_DIR if doc_source == "Local Directory" else UPLOAD_DIR
        
        if doc_source == "Local Directory":
            st.info(f"Documents will be loaded from {LOCAL_DOCS_DIR} directory")
            
            if st.button("Process Local Documents"):
                start_time = time.time()
                with st.spinner("Processing local documents..."):
                    try:
                        process_documents(LOCAL_DOCS_DIR)
                        processing_time = time.time() - start_time
                        st.success(f"Local processing completed in {processing_time:.2f}s")
                    except Exception as e:
                        st.error(f"Error processing local documents: {str(e)}")
        
        else:  
            st.info("Upload your documents here")
            uploaded_files = st.file_uploader("Choose files to upload",accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'md'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Upload & Process"):
                    if uploaded_files:
                        start_time = time.time()
                        with st.spinner("Processing uploaded documents..."):
                            try:
                                clear_documents(UPLOAD_DIR)  
                                handle_uploaded_files(uploaded_files, UPLOAD_DIR)
                                process_documents(UPLOAD_DIR)
                                
                                processing_time = time.time() - start_time
                                st.success(f"Upload processing completed in {processing_time:.2f}s")
                            except Exception as e:
                                st.error(f"Error processing uploaded documents: {str(e)}")
                    else:
                        st.warning("Please upload some documents first!")
            
            with col2:
                if st.button("Clear Uploads"):
                    clear_documents(UPLOAD_DIR)
                    st.success("Uploaded documents cleared!")
        
        if "vectors" in st.session_state:
            st.divider()
            st.subheader("Document Statistics")
            if "doc_statistics" in st.session_state:
                stats = st.session_state.doc_statistics
                st.info(f"Documents processed: {stats['total_docs']}")
                st.info(f"Total chunks: {stats['total_chunks']}")
                
                # Show distribution of document sources
                if st.checkbox("Show document sources"):
                    for source, count in stats['doc_sources'].items():
                        source_name = Path(source).name if isinstance(source, str) else "Unknown"
                        st.info(f"{source_name}: {count} chunks")
            else:
                st.info(f"Documents processed: {len(st.session_state.docs)}")
                st.info(f"Total chunks: {len(st.session_state.final_documents)}")
            
            current_source = "Local Directory" if doc_source == "Local Directory" else "Uploaded Files"
            st.info(f"Current source: {current_source}")
            if "current_directory" in st.session_state:
                st.info(f"Active directory: {st.session_state.current_directory}")
        
        # Add advanced retrieval settings
        st.divider()
        st.subheader("Retrieval Settings")
        retrieval_k = st.slider("Number of chunks to retrieve", min_value=5, max_value=30, value=12, 
                               help="Higher values retrieve more document chunks, which can help with complex queries across multiple documents")

    
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "response_time" in message:
                    st.info(f"Time: {message['response_time']:.2f}s")

    
    user_input = st.chat_input("Ask about your documents...")
    mic_button = st.button("🎙️ Voice Input", key="mic")

    if user_input or mic_button:
        if mic_button:
            with st.spinner("Listening..."):
                user_input = recognize_speech()

        if user_input:
            if not hasattr(st.session_state, 'vectors') or st.session_state.vectors is None:
                st.warning("Please process documents first!")
                return

            start_time = time.time()

            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            try:
                with st.chat_message("assistant") as assistant_msg:
                    # Implement streaming response
                    response_container = st.empty()
                    full_response = ""
                    
                    # Get streaming response generator
                    response_stream = get_answer(user_input, current_directory)
                    
                    # Stream the response
                    for token in response_stream:
                        full_response += token
                        response_container.markdown(full_response + "▌")
                    
                    # Display final response without cursor
                    response_container.markdown(full_response)
                    
                    total_time = time.time() - start_time
                    st.info(f"Response time: {total_time:.2f}s")

                    # Generate audio after the full response is received
                    audio_bytes = speak_text(full_response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")

                    # Add the message to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response, 
                        "response_time": total_time
                    })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error generating response: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()