# Fine-Tuning-LLM-on-CSV-data
The provided code is a Python script that performs the following tasks:

1. **Data Preprocessing**:
   - Loads the "zomato-bangalore-dataset" from Kaggle.
   - Extracts the relevant CSV file ("zomato.csv") from the downloaded dataset.
   - Performs data cleaning and preprocessing steps on the "zomato.csv" dataset, including dropping irrelevant columns, handling null values, and filtering the data based on specific criteria (e.g., rate >= 3).
   - Saves the preprocessed dataset as a new CSV file ("file.csv").

2. **Library Installations**:
   - Installs required Python libraries, including Transformers, Accelerate, bitsandbytes, HuggingFace Hub, LangChain, LangChain Community, PyPDF, sentence-transformers, FAISS, and Chroma.

3. **Model Loading and Configuration**:
   - Loads the Mistral-7B-Instruct-v0.1 model from HuggingFace Hub.
   - Configures the model with quantization settings (BitsAndBytes) for efficient inference on GPU or CPU.
   - Sets up a text generation pipeline using the loaded model and tokenizer.

4. **Document Loading and Splitting**:
   - Loads the preprocessed "file.csv" dataset using the CSVLoader from LangChain Community.
   - Splits the documents into smaller chunks using the CharacterTextSplitter.

5. **Embedding and Vector Store**:
   - Creates text embeddings using the HuggingFaceEmbeddings from LangChain Community.
   - Builds a FAISS vector store from the chunked documents and their embeddings.

6. **Conversational Retrieval Chain**:
   - Sets up a ConversationalRetrievalChain using the HuggingFacePipeline LLM (Mistral-7B-Instruct-v0.1) and the FAISS vector store retriever.

7. **Interactive Question-Answering Loop**:
   - Enters an infinite loop that prompts the user to enter a query.
   - Passes the query and chat history to the ConversationalRetrievalChain.
   - Prints the answer obtained from the chain.
   - Appends the query and answer to the chat history for context in subsequent questions.
