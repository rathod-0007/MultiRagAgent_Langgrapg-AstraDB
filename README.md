
# Agentic RAG with LangGraph and Groq

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This project demonstrates an **Agentic RAG (Retrieval Augmented Generation) system** using **Langgraph**, **Groq**, and **Cassandra** as a vector store. It intelligently routes user questions to either a vector database containing information about LLM agents, prompt engineering, and adversarial attacks, or to Wikipedia for general knowledge queries.

## Features

- Vector Store Retrieval with Cassandra and HuggingFace embeddings
- LangChain-based document splitting, embeddings, and retrieval
- Groq LLM API for intelligent routing of user queries
- Wikipedia and Arxiv fallback for general knowledge
- Agentic routing: routes questions dynamically to the most relevant datasource

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/yourusername/agentic-rag-langchain-groq.git
cd agentic-rag-langchain-groq
```

### 2. Install Dependencies

```
pip install cassio langchain langchain-huggingface langchain-groq langchain-community
```

### 3. Configure Cassandra (Astra DB)

```python
import cassio

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:..."  # Enter your token
ASTRA_DB_ID = "your_database_id"            # Enter your database ID

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
```

### 4. Set Environment Variable

```python
%env USER_AGENT=MyLangChainBot/1.0
```

## Usage

### Load and Index Documents

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

astra_vector_store.add_documents(doc_splits)
retriever = astra_vector_store.as_retriever()
```

### Configure Groq LLM Router

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(..., description="Route question to vectorstore or Wikipedia")

groq_api_key = "YOUR_GROQ_API_KEY"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = '''You are an expert at routing user questions to vectorstore or Wikipedia.
Use vectorstore for topics like agents, prompt engineering, adversarial attacks.
Otherwise, use Wikipedia.'''.strip()

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

question_router = route_prompt | structured_llm_router
```

### Wikipedia and Arxiv Tools

```python
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
```

### Build Workflow Graph

```python
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = Document(page_content=docs)
    return {"documents": [wiki_results], "question": question}

def route_question(state):
    source = question_router.invoke({"question": state["question"]})
    return "wiki_search" if source.datasource == "wiki_search" else "retrieve"

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve"})
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

app = workflow.compile()
```


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

## References

- [LangChain Documentation](https://www.langchain.com/docs/)
- [Groq LLM API](https://www.groq.com/)
- [HuggingFace Embeddings](https://huggingface.co/docs/transformers/main/en/main_classes/embeddings)
- [Cassandra Vector Store](https://www.datastax.com/)
- [Arxiv API Wrapper](https://arxiv.org/help/api/user-manual)
- [Wikipedia API Wrapper](https://www.mediawiki.org/wiki/API:Main_page)



## Author

**Rathod Pavan Kumar Naik**  
- Email: rathodpavan2292@gmail.com  
- GitHub: [rathod-0007](https://github.com/rathod-0007)
