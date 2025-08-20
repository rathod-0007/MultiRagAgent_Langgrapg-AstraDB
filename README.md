# Agentic RAG with LangChain and Groq

This project demonstrates an Agentic RAG (Retrieval Augmented Generation) system that leverages **LangChain**, **Groq**, and **Cassandra** as a vector store. The system is designed to intelligently route user questions to the most relevant data source.

For questions about **LLM agents**, **prompt engineering**, and **adversarial attacks**, the system consults a specialized vector database. For general knowledge queries, it performs a **Wikipedia search**, ensuring that the responses are always relevant and accurate.

---

## Getting Started

Follow these steps to set up and run the project.

### Prerequisites

* Python 3.8+
* Google Colab (recommended for easy setup)
* Astra DB account for Cassandra vector store
* Groq API key

### Setup

1.  **Clone the Repository (or use Google Colab)**

    If you're running this locally, clone the repository. If you're using Google Colab, you can simply open the provided notebook.

2.  **Install Dependencies**

    Run the following command to install the required libraries:

    ```bash
    !pip install -r requirements.txt 
    # If you're running this in Colab, you can install them directly
    !pip install -qU langchain-groq wikipedia python-dotenv typing_extensions cassandra-driver cassio langchain-huggingface
    ```

3.  **Configure API Keys and Database**

    Set up your environment variables or use Google Colab's `userdata` feature to store your secrets.

    ```python
    import cassio
    from google.colab import userdata

    ASTRA_DB_APPLICATION_TOKEN = userdata.get('ASTRA_DB_APPLICATION_TOKEN')
    ASTRA_DB_ID = userdata.get('ASTRA_DB_ID')
    groq_api_key = userdata.get('groq_api_key')

    # Initialize Cassio
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    ```

---

## Project Structure and Workflow

The system is built on a conditional routing logic using LangChain's expression language and Groq's function calling capabilities.

### 1. Data Ingestion

The system ingests documentation from specific URLs related to LLM agents and prompt engineering. These documents are split into smaller chunks and then embedded using the `all-MiniLM-L6-v2` model.

### 2. Vector Store

The embedded documents are stored in a Cassandra database, which is configured as a vector store. This allows for efficient semantic search and retrieval.

### 3. Query Routing

A core component of this project is the **query router**. It uses a Groq model with a structured output to classify the user's question into one of two categories: `vectorstore` or `wiki_search`.

* **`vectorstore`**: The question is related to the specialized topics (agents, prompt engineering, etc.). The system retrieves relevant documents from the Cassandra vector store.
* **`wiki_search`**: The question is for general knowledge. The system performs a search using the Wikipedia API.

### 4. Generation

Based on the routing decision, the system retrieves the appropriate information and provides a final answer.

---

## Usage Examples

Here are two examples demonstrating the system's routing capabilities.

### Example 1: Query for Vector Store

This query is related to LLM agents and will be routed to the Cassandra vector store.

```python
inputs_vectorstore = {"question": "What is agent?"}
for output in app.stream(inputs_vectorstore):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value['documents'][0].page_content)
```

### Example 2: Query for Wikipedia Search

This query is for general knowledge and will be routed to the Wikipedia search tool.

```python
inputs_wiki = {"question": "Avengers"}
for output in app.stream(inputs_wiki):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value['documents'][0].page_content)
```
