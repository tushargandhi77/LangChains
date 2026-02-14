# LangChains Learning Repository

![Langchains](https://img.shields.io/badge/Lang%20Chains-Langchains-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models-orange)

This repository is a hands-on LangChain learning workspace. It is organized by concept so you can learn one building block at a time (models, prompts, chains, retrievers, tools, agents, RAG, etc.), then combine them.

## Repository Goal

- Build practical intuition for core LangChain abstractions.
- Compare multiple model providers (OpenAI, Gemini, Anthropic, Hugging Face).
- Practice end-to-end LLM patterns: prompt -> model -> parser -> chain -> retriever -> tool -> agent.
- Keep reusable mini examples for revision.

## Tech Stack

- Python
- LangChain ecosystem packages
- Google Gemini, OpenAI, Anthropic, Hugging Face integrations
- FAISS and Chroma vector stores
- Streamlit UI samples
- Jupyter notebooks for guided experiments

## Project Structure

```text
LangChains/
|-- README.md
|-- requirements.txt
|-- test.py
|-- .env
|-- .gitignore
|-- LangChain Models/
|-- LangChain Prompts/
|-- LangChain Output Parser/
|-- LangChain Structured Output/
|-- LangChain Chains/
|-- LangChain Runnables/
|-- LangChain Document Loaders/
|-- LangChain Text Splitters/
|-- LangChain Vector Stores/
|-- LangChain Retrievers/
|-- LangChain Tools/
|-- LangChain Tool Calling/
|-- LangChain AI Agents/
|-- LangChain Chat with Youtube Video RAG/
```

## Setup and Run

1. Create and activate virtual environment.
2. Install dependencies.
3. Configure API keys in `.env`.
4. Run scripts folder by folder.

Example:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python test.py
```

## Environment Variables

Your scripts use providers that usually need these keys in `.env`:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN`

Also used in some local embedding/model files:

- `HF_HOME` (for local Hugging Face cache directory)

## Dependency Notes (`requirements.txt`)

- `langchain`, `langchain-core`: Core framework primitives.
- `langchain-openai`, `openai`: OpenAI model and embedding support.
- `langchain-google-genai`, `google-generativeai`: Gemini chat and embeddings.
- `langchain-anthropic`: Claude model integration.
- `langchain-huggingface`, `transformers`, `huggingface-hub`: Hugging Face APIs and local pipelines.
- `python-dotenv`: Load environment variables from `.env`.
- `numpy`, `scikit-learn`: Numeric helpers and cosine similarity demo.
- `youtube_transcript_api`: Transcript ingestion for YouTube RAG.

## Detailed Module Guide

## 1) Root Files

- `test.py`: quick sanity check that prints installed LangChain version.
- `.gitignore`: excludes `venv` and `.env`.
- `requirements.txt`: curated dependencies for all demos.

## 2) LangChain Models

### `LangChain Models/LLMS`
- `LLM_DEMO.py`: basic non-chat OpenAI completion model invocation.

### `LangChain Models/ChatModels`
- `chatmodel_openai.py`: OpenAI chat model demo with temperature and token settings.
- `chatmodel_gemini.py`: Gemini chat model invocation.
- `chatmodel_antropic.py`: Anthropic Claude chat invocation.
- `chatmodel_hf_api.py`: Hugging Face hosted inference endpoint through LangChain.
- `chatmodel_local.py`: local Hugging Face pipeline wrapped as a chat model.

### `LangChain Models/EmbeddedModels`
- `Embedding_OpenAI.py`: query embedding with OpenAI embeddings.
- `Embedding_Gemini.py`: query embedding with Gemini embedding model.
- `Embedding_Gemini_document.py`: document embeddings for multiple texts.
- `Embedding_hf_local.py`: local Hugging Face embeddings for documents.

### `LangChain Models/Document Similarity tool`
- `document_similarity.py`: embeds multiple documents and a query, then uses cosine similarity (`sklearn`) to find best match.

## 3) LangChain Prompts

- `prompt_generator.py`: creates and saves a reusable `PromptTemplate` into `template.json`.
- `template.json`: serialized prompt template used by UI scripts.
- `prompt_ui.py`: Streamlit app that fills template and sends prompt to Gemini.
- `prompt_ui_chains.py`: same app pattern using chain syntax (`template | model`).
- `messages.py`: demonstrates system/human message objects and model output as `AIMessage`.
- `chat_prompt_template.py`: `ChatPromptTemplate` with runtime variables.
- `message_placeholder.py`: demonstrates `MessagesPlaceholder` with preloaded chat history.
- `chat_history.txt`: sample prior conversation injected into prompt.
- `chatbot.py`: console chatbot loop with chat history memory list.

## 4) LangChain Output Parser

- `stroutputparser.py`: manual two-step prompt pipeline without parser chaining.
- `stroutputparser1.py`: same idea implemented with `StrOutputParser` and chain composition.
- `jsonoutputparser.py`: enforces JSON response shape with `JsonOutputParser`.
- `pydanticoutputparser.py`: typed structured output using Pydantic parser.
- `structuredouputparser.py`: schema-driven output via `ResponseSchema` + `StructuredOutputParser`.

## 5) LangChain Structured Output

- `with_structured_ouput_typedict.py`: structured extraction using `TypedDict` schema.
- `with_structured_output_pydantic.py`: structured extraction using Pydantic model schema.
- `with_structured_output_json.py`: structured extraction using raw JSON schema dict.
- `with_structured_output_hf.py`: attempts structured extraction with Hugging Face chat model.
- `pydantic_demo.py`: standalone Pydantic validation demo.
- `json_schema.json`: example JSON schema file.

## 6) LangChain Chains

- `simple_chains.py`: simple `prompt -> model -> parser` chain.
- `sequential_chain.py`: multi-step sequential summarization chain.
- `parallel_chain.py`: parallel notes + quiz generation, then merged in a final prompt.
- `conditional_chain.py`: sentiment classification then conditional branching (`RunnableBranch`) for positive/negative response generation.

## 7) LangChain Runnables

- `runnable_sequence.py`: explicit `RunnableSequence` construction.
- `runnable_parallel.py`: parallel generation of tweet and LinkedIn post.
- `runnable_passthrough.py`: pass generated joke and compute word count in parallel.
- `runnable_lambda.py`: `RunnableLambda` function usage within chain.
- `runnable_branch.py`: branch summarization only when report exceeds threshold.
- `langchain_dummy_chain.ipynb`: custom toy chain classes to understand chain mechanics.
- `langchain_standardize.ipynb`: notebook standardizing runnable interface from scratch.

## 8) LangChain Document Loaders

- `text_loader.py`: loads text file and summarizes with model.
- `pdf_loader.py`: loads PDF pages and inspects metadata.
- `csv_loader.py`: loads CSV rows as documents.
- `directory_loader.py`: bulk load PDFs from folder (`lazy_load` demonstration).
- `webBase_loader.py`: loads webpage text and performs QA over page content.
- Data files:
  - `cricket.txt`
  - `Social_Network_Ads.csv`
  - `dl-curriculum.pdf`
  - `books/...pdf`

## 9) LangChain Text Splitters

- `text_structure_based.py`: recursive splitting for generic prose.
- `length_based.py`: fixed-length `CharacterTextSplitter` for PDF content.
- `markdown_splitting.py`: markdown-aware splitting.
- `python_code_splitting.py`: code-aware splitting using Python language mode.
- `semantic_meaning_based.py`: semantic chunking using embedding-based boundaries.

## 10) LangChain Vector Stores

- `langchain_chroma.ipynb`: end-to-end Chroma usage:
  - create docs
  - add/search/filter/update/delete
  - persist to local `chroma_db/`
- `chroma_db/`: persisted Chroma collection data.

## 11) LangChain Retrievers

- `vector_store_retriever.py`: create Chroma retriever from embedded docs.
- `Wikipedia_retriever.py`: fetch top-k results from Wikipedia retriever.
- `MMR_Retriever.py`: retriever with MMR search for diversity.
- `Multi_Query_Retriever.py`: compare plain similarity vs multi-query expansion retrieval.
- `Contextual_Compression_Retriever.py`: compression retriever with `LLMChainExtractor`.

## 12) LangChain Tools

- `custom_tool.py`: convert typed Python function into tool with `@tool`.
- `structured_tool.py`: `StructuredTool.from_function` with explicit args schema.
- `base_custom_tool.py`: custom class inheriting `BaseTool`.
- `toolkit.py`: bundles reusable math tools in toolkit style.
- `buildin_tool_duckduckgo.py`: built-in DuckDuckGo search tool.
- `buildin_tool_shell_tool.py`: built-in shell command tool.

## 13) LangChain Tool Calling

- `tool_calling.py`: model tool binding, tool call extraction, and manual tool execution.
- `tool_calling.ipynb`: message-driven tool-calling workflow.
- `curency_converter.ipynb`: multi-tool workflow for currency conversion using live API + calculation.

## 14) LangChain AI Agents

- `agents_in_langchain.ipynb`: ReAct-style agent setup with custom tool(s), DuckDuckGo search, LangChain Hub prompt, and `AgentExecutor` invocation.

## 15) LangChain Chat with Youtube Video RAG

- `rag_langchain.ipynb`: complete RAG pipeline over YouTube transcript:
  - ingestion via `youtube_transcript_api`
  - text splitting
  - embedding + FAISS indexing
  - retrieval
  - augmentation prompt construction
  - answer generation with Gemini
  - chain-based RAG variant (`RunnableParallel` + parser)

## Learning Path (Recommended)

1. `LangChain Models` -> understand model interfaces.
2. `LangChain Prompts` -> control model behavior.
3. `LangChain Output Parser` + `Structured Output` -> enforce response formats.
4. `LangChain Chains` + `Runnables` -> compose workflows.
5. `Document Loaders` + `Text Splitters` -> prepare unstructured data.
6. `Vector Stores` + `Retrievers` -> retrieval systems.
7. `Tools` + `Tool Calling` -> external action support.
8. `AI Agents` + `YouTube RAG` -> full intelligent application patterns.

## Revision Checklist

Use this before interviews or project implementation:

- Can I explain difference between `LLM` vs `ChatModel` usage in this repo?
- Can I build a chain with `|` composition and with explicit `RunnableSequence`?
- Can I enforce structured output with `JsonOutputParser`, Pydantic parser, and `with_structured_output`?
- Can I choose correct splitter type (length, recursive, markdown, semantic) for data type?
- Can I build vector store + retriever + prompt augmentation flow for RAG?
- Can I define tools in all three styles (`@tool`, `StructuredTool`, `BaseTool`)?
- Can I explain tool calling vs agent execution differences?

## Notes and Improvement Ideas

- Some scripts use older model IDs or provider settings; update model names as APIs evolve.
- Some files contain minor typos in prompts/filenames (safe to refactor for clarity).
- Consider adding a single unified `run_examples.md` with expected outputs and troubleshooting.
- Consider splitting provider-specific env requirements by section for cleaner onboarding.

## License

See `LICENSE`.
