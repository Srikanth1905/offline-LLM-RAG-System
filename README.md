# LLM-RAG Assistant

## Overview

LLM-RAG Assistant is an AI-powered document retrieval chatbot that allows users to search, process, and interact with documents using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). It also includes speech recognition and text-to-speech capabilities for seamless interaction.

## Features

- **Document Processing** - Supports PDF, DOCX, TXT, JSON, and CSV
- **Retrieval-Augmented Generation (RAG)** - Intelligent document-based answering
- **Speech Recognition** - Convert spoken words into queries (via Whisper)
- **Text-to-Speech (TTS)** - Get AI-generated responses in voice format
- **Vector Search with FAISS** - Efficient semantic search for documents
- **Streamlit Web Interface** - Simple and interactive UI

## Tech Stack

- **Python 3.9+**
- **Streamlit** (UI Framework)
- **LangChain** (RAG & Prompt Engineering)
- **FAISS** (Vector Search Engine)
- **Hugging Face Embeddings** (Text Representation)
- **Whisper** (Speech Recognition)
- **PyMuPDF & Docx2txt** (Document Processing)
- **Ollama** 

## Installation & Setup

### Clone the Repository
```
git clone https://github.com/Srikanth1905/offline-LLM-RAG.git
cd offline-LLM-RAG
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run the Application
```
streamlit run main.py
```



## Downloading Models from Ollama

### Install Ollama

#### For macOS & Linux
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

#### For Windows
1. Download the installer from [Ollama’s official website](https://ollama.com/download).
2. Run the installer and follow the setup instructions.

Verify Ollama installation:
```sh
ollama --version
```

### Download a Model
```sh
ollama pull <model_name>
```
Example:
```sh
ollama pull llama3
```
Other models:
```sh
ollama pull mistral
ollama pull gemma
ollama pull phi3
```
Check available models:
```sh
ollama list
```

### Running a Model
```sh
ollama run llama3
```
Programmatically using Python:
```python
import ollama

response = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

print(response['message'])
```

### Managing Models
List installed models:
```sh
ollama list
```
Remove a model:
```sh
ollama rm <model_name>
```
Update a model:
```sh
ollama update <model_name>
```

### Custom Model Fine-tuning
```sh
echo "FROM llama3" > Modelfile
ollama create my-model -f Modelfile
ollama run my-model
```

For more details, visit [Ollama's official documentation](https://ollama.com/docs). 🚀

## Project Structure
```
📂 offline-LLM-RAG
 ┣ 📂 data                 # Local document storage
 ┣ 📂 uploaded_data        # User uploaded files
 ┣ 📂 temp_audio          # Temporary audio storage
 ┣ 📜 main.py             # Streamlit UI
 ┣ 📜 document_processor.py # Document Processing & RAG
 ┣ 📜 audio_manager.py     # Speech Recognition & TTS
 ┣ 📜 utils.py            # Helper functions
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 .gitignore           # Ignored files
 ┣ 📜 README.md            # Project documentation
 ┗ 📜 LICENSE              # License file
```

## How It Works

1. Upload or load documents from a directory.
2. The LLM-RAG pipeline processes and chunks documents.
3. Users can type or use voice commands to ask questions.
4. The system retrieves relevant document chunks and generates answers.
5. AI-generated answers are displayed in text and speech format.

## Usage

- **To process local documents**:
  - Select "Local Directory" and click "Process Documents"  
- **To upload and process new documents**:
  - Click "Upload & Process"  
- **To ask a question**:
  - Type in the chat box or use voice input  

## Customization

- Change the **LLM Model** in `document_processor.py`
  ```python
  return ChatOllama(model="llama3.2:1b", verbose=True, temperature=0)
  ```
- Modify **retrieval settings** in `main.py`
  ```python
  retrieval_k = st.slider("Number of chunks to retrieve", min_value=5, max_value=30, value=12)
  ```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Follow these steps:
1. **Fork** the repository.
2. Create a **new branch**:
   ```
   git checkout -b feature-new-feature
   ```
3. Make your changes & **commit**:
   ```
   git commit -m "Added new feature"
   ```
4. **Push** to your branch:
   ```
   git push origin feature-new-feature
   ```
5. Open a **Pull Request (PR)**.

## Contact

For issues and discussions, please open an **[issue](https://github.com/Srikanth1905/offline-LLM-RAG/issues)** or contact:

## Star this Repo if You Like It! ⭐

