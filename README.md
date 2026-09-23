# Video Processing RAG + Multi-Agent App on Databricks

This repository is a practical end-to-end example of building an AI application on top of video data using Databricks. It follows the workflow described in the blog:

- A Sample LLM based Video Processing RAG App: Introduction
- https://medium.com/@junshan0/a-sample-llm-based-video-processing-rag-app-introduction-1cb3f5883eb8?sk=1d6066d0807553d1eb28b7d58e50acd3

The project goes from raw video files in S3 to a searchable, AI-enriched knowledge base, then exposes those capabilities via RAG, MCP tools, and a multi-agent workflow.

---

## Why this project exists

The goal is to show how traditional data engineering concepts map into modern AI application engineering:

- cloud storage and ingestion
- metadata and schema design
- streaming data processing
- chunking unstructured content
- AI-based enrichment and summarization
- semantic vector search
- retrieval-augmented generation
- tool calling via MCP
- orchestration through a multi-agent supervisor

This is not just a demo chatbot. It is a full architecture pattern for creating AI-enabled data products over unstructured media.

---

## High-level flow

The project follows this progression:

Ingest → Process → Enrich → Search → Retrieve → Generate → Tool → Agent → Multi-Agent

At a high level:

- S3 stores raw video files
- Databricks ingests those files and stores raw metadata
- video chunks are created and enriched with captions and metadata
- semantic search indexes those chunks
- a RAG app retrieves relevant context and passes it to an LLM
- MCP tools expose the search and summarization capabilities
- a multi-agent supervisor composes those capabilities into a richer workflow

---

## Folder and file overview

This workspace includes notebooks, Python agents, MCP tooling, and a small app scaffold.

### Notebooks

- section 1.ipynb — Databricks workspace setup and catalog/schema provisioning
- section 2.ipynb — S3 bucket setup and external location configuration
- section 3.ipynb — streaming ingest flow from S3 into Databricks
- section 4-1.ipynb — video chunking logic
- section 4-2.ipynb — caption and metadata generation
- section 4-3.ipynb — vector search index creation and sync
- section 5.ipynb — RAG application implementation
- section 6.ipynb — MCP tool packaging for UC functions
- section 7.ipynb — multi-agent orchestration example

### Python files

- server.py — FastMCP server exposing search and summarization tools
- rag_agent.py — ChatAgent implementation using Databricks vector search + model serving
- multi_agent.py — wrapper for a multi-agent supervisor model
- rag_mcp_agent.py — agent that calls MCP tools through the Databricks MCP client
- supervisor.py — supervisor agent that orchestrates specialist tools
- specialists.py — tool-calling specialist implementations

### App scaffold

- app/app.py — simple Streamlit sketch for a video chatbot
- app/app.yaml — app deployment config for Databricks Apps
- app/requirements.txt — app dependencies

---

## Project architecture in practice

The repo uses a Databricks lakehouse + AI architecture:

- Catalog: `video_ai`
- Schemas:
  - `video_ai.bronze` for raw metadata and ingest state
  - `video_ai.silver` for chunked and enriched content
  - `video_ai.ai` for AI functions and service artifacts
- Volumes under `video_ai.bronze` for raw or staged video files
- Delta tables for chunk metadata and enriched content
- Vector search index for semantic retrieval over chunk captions
- MLflow model packaging for agent and RAG variants

The project also demonstrates the move from code-first experimentation to reusable AI tools and orchestration.

---

## Notebook flow by section

### Section 1: Set up Databricks

Creates the workspace foundation:

- Unity Catalog catalog named `video_ai`
- medallion schemas (`bronze`, `silver`, `gold`, `ai`)
- volume for raw videos
- initial workspace organization for the project

### Section 2: Set up S3 access

Shows how to connect Databricks to an S3 bucket:

- external location creation
- storage credential setup
- file listing validation
- permission grants for workshop participants or groups

### Section 3: Streaming ingestion

Builds the ingestion pipeline:

- reads `.mp4` files from S3 via Auto Loader / CloudFiles
- captures file metadata like path, size, and modified time
- writes Bronze metadata rows
- stores chunk/checkpoint state in a volume
- writes to a Delta table such as `video_ai.bronze.video_files`

### Section 4-1: Chunking

Segments videos by time windows with overlap:

- 5-minute chunk size
- 15-second overlap
- chunk metadata includes `chunk_id`, `video_id`, `start_time`, `end_time`, `duration`, `chunk_index`
- state is tracked in `video_ai.silver.video_chunks`

### Section 4-2: Captions and metadata

Transforms each chunk into AI-ready text:

- transcript generation
- caption summarization
- topic and category classification
- keywords and entity extraction
- language and content metadata
- processing timestamps

These results are stored in `video_ai.silver.video_chunk_content`.

### Section 4-3: Search index

Creates a vector search index from enriched chunk content:

- enables Delta Change Data Feed
- sets a non-null primary key on `chunk_id`
- creates a vector index on `video_ai.silver.video_chunk_content`
- uses `caption` as the embedding source column
- syncs and tests retrieval quality

### Section 5: RAG application

Implements a retrieval-augmented generation flow:

- query vector search by user question
- build a context window from the top matches
- pass the context to an LLM endpoint
- return grounded answers with citations/timestamps

The implementation is centered around the `video_rag_agent` pattern and the `retrieve` / `build_context` / `answer` flow.

### Section 6: MCP tools

Exposes the project actions as UC functions and MCP-callable tools:

- `search_video`
- `get_video_metadata`
- `get_video_chunk`
- `find_topic`
- `summarize_video`
- `query_video_transcript`

This layer turns data and AI capabilities into reusable tools for other agents or clients.

### Section 7: Multi-agent flow

Creates a higher-level orchestration layer:

- a supervisor agent delegates work to specialists
- specialist agents call MCP tools to retrieve, summarize, and compare results
- the final answer combines multiple evidence sources and citations

---

## Python agent components

### `rag_agent.py`

This defines a `ChatAgent` subclass that:

- queries the vector search index
- gathers candidate chunks
- builds prompt context
- calls a model serving endpoint
- returns an assistant message with the answer

### `multi_agent.py`

This wraps the `supervise` function as a `ChatAgent` model.

### `rag_mcp_agent.py`

This version uses `DatabricksMCPClient` to connect to the MCP tools exposed by the workspace, then calls them via the model as tool use.

### `supervisor.py`

This is the orchestration layer:

- receives a user request
- decides which specialist tool to call
- iterates through tool calls until enough information exists
- returns the final synthesized response

### `specialists.py`

Defines three specialist agents:

- `video_search_agent` — finds relevant segments
- `data_agent` — extracts structured facts
- `summary_agent` — synthesizes and compares results

---

## MCP server

The repo includes `server.py`, which creates a lightweight FastMCP server:

- `search_video(query, k=4)`
- `summarize_video(video_id)`

This server calls Databricks SQL UDFs such as `video_ai.ai.search_video(...)` and `video_ai.ai.summarize_video(...)` and exposes them as MCP tools.

The server is designed to make the project’s capabilities available to external agents and tool-call systems.

---

## App scaffold

The folder `app/` contains a small application starter:

- `app.py` — a Streamlit example for a video chatbot
- `app.yaml` — Databricks app configuration
- `requirements.txt` — minimal dependencies

This is a simple front-end wrapper around the model-serving or agent call flow.

---

## Environment and prerequisites

Before running the project, you will typically need:

- Databricks workspace access with the right permissions
- Unity Catalog enabled and usable in the workspace
- access to an S3 bucket containing video files
- a valid storage credential or IAM configuration for S3
- a SQL warehouse if using SQL execution patterns
- model serving endpoints for LLMs and embeddings
- `ffprobe` available when measuring video durations on the driver or local path

---

## Recommended execution order

1. Start with Section 1 and Section 2 to create the catalog, schemas, and S3 access.
2. Run Section 3 to ingest raw videos into the Bronze layer.
3. Run Section 4-1 and Section 4-2 to chunk and enrich the videos.
4. Run Section 4-3 to create the vector search index.
5. Run Section 5 to build and test the RAG app.
6. Run Section 6 to expose the capabilities as UC/MCP tools.
7. Run Section 7 to test the multi-agent orchestration pattern.
8. Use the Python files and app scaffold to package or deploy the final solution.

---

## Key takeaways

This project demonstrates an important architectural transition:

- from raw data engineering to AI-powered data engineering
- from static notebooks to reusable tools and agents
- from searchable files to a video knowledge system
- from one-off LLM prompts to multi-step orchestration

The practical lesson is that data engineering skills remain central: the same ideas of ingestion, metadata, transformation, indexing, and governance are still foundational, but now they are used to support AI-native applications.

---

## References

- Blog series: A Sample LLM based Video Processing RAG App: Introduction
- Databricks Unity Catalog documentation
- Databricks Vector Search documentation
- Databricks model serving / MLflow documentation
- MCP tool usage patterns for Databricks and agent workflows

---

## License

This repository is intended as a learning, experimental, and demonstration project for building a video-processing RAG and multi-agent workflow on Databricks.
