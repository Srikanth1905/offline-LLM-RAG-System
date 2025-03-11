import os
import time
import logging
from functools import wraps
from pathlib import Path
import shutil

def setup_logging():
    """
    Set up logging configuration
    
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs('log', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('log', 'processing_times.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def track_processing_time(func):
    """
    Decorator to track processing time of functions
    
    Args:
        func: Function to be decorated
        
    Returns:
        function: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            logging.getLogger(__name__).info(f"Function '{func.__name__}' completed in {processing_time:.4f} seconds")
            return result
        
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

def handle_uploaded_files(uploaded_files, upload_dir):
    """
    Handle uploaded files and save them to the upload directory
    
    Args:
        uploaded_files (list): List of uploaded files from Streamlit
        upload_dir (str): Directory to save uploaded files
    """
    os.makedirs(upload_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        safe_filename = Path(uploaded_file.name).name
        file_path = os.path.join(upload_dir, safe_filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

def clear_documents(directory: str):
    """
    Clear documents from directory and reset session state
    
    Args:
        directory (str): Directory to clear
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
    
    # Clear session state
    import streamlit as st
    if "vectors" in st.session_state:
        del st.session_state.vectors
    if "docs" in st.session_state:
        del st.session_state.docs
    if "current_directory" in st.session_state:
        del st.session_state.current_directory