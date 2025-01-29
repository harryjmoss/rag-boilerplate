"""Boilerplate code for using the RAG pattern with LangChain."""

import os
from functools import partial

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from rag_boilerplate.config import get_settings

settings = get_settings()


def get_embeddings_model_huggingface(
    embeddings_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> HuggingFaceEmbeddings:
    """
    Get the HuggingFace embeddings model.

    Args:
        embeddings_model (str, optional): Huggingface embeddings model.
            Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        HuggingFaceEmbeddings: HuggingFace embeddings model.

    """
    return HuggingFaceEmbeddings(model_name=embeddings_model)


def get_embeddings_model_openai() -> OpenAIEmbeddings:
    """
    Get the OpenAI embeddings model.

    Returns:
        OpenAIEmbeddings: OpenAI embeddings model.

    """
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )


def load_blog_post() -> list[Document]:
    """
    Load a blog post from the web.

    The blog post is the LLM Powered Autonomous Agents blog post by Lilian Weng.
    This is suggested by the LangChain docs.

    Returns:
        List[Document]: A list of documents comprising the blog post.

    """
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
    )
    return loader.load()


def chunk_and_store_vectors(vector_store: PGVector, docs: list[Document]) -> None:
    """
    Chunk text and store vectors in the vector store.

    Args:
        vector_store (PGVector): PGVector vector store.
        docs (List[Document]): List of documents to store.

    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    _ = vector_store.add_documents(documents=all_splits)


def build_rag_prompt() -> ChatPromptTemplate:
    """
    Build a prompt for the RAG model from the default huggingface template.

    Returns:
        ChatPromptTemplate: A prompt template for the RAG models

    """
    prompt_string = """You are an assistant for question-answering tasks.
     Use the following pieces of retrieved context to answer the question.
     If you don't know the answer, just say that you don't know.
     Use three sentences maximum and keep the answer concise.
     Question: {question}
     Context: {context}
     Answer:
    """
    return ChatPromptTemplate.from_template(prompt_string)


# Define state for application
class State(TypedDict):
    """Define the state for the langchain graph."""

    question: str
    context: list[Document]
    answer: str


# Define appication steps
def retrieve(state: State, vector_store: PGVector) -> dict[str, list[Document]]:
    """
    Retrieve documents from the vector store.

    Args:
        state (State): Application state.
        vector_store (PGVector): PGVector vector store.

    Returns:
        dict[str, List[Document]]: Retrieved documents.

    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State, llm: ChatOpenAI, prompt: ChatPromptTemplate) -> dict[str, str]:
    """
    Generate a response from the LLM model.

    Args:
        llm (ChatOpenAI): OpenAI language model.
        state (State): Application state.
        prompt (str): Prompt for the LLM model.

    Returns:
        dict[str, str]: Generated response.

    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def disable_langsmith_logging() -> None:
    """Disable LangSmith logging."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = "null"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""


def main() -> None:
    """Run the main application."""
    disable_langsmith_logging()
    language_model = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY.get_secret_value())
    logger.info("Loading embeddings model...")
    embeddings_model = get_embeddings_model_huggingface()

    logger.info("Setting up vector store connection.")
    vector_store = PGVector(
        embeddings=embeddings_model,
        collection_name="my_docs",
        connection=f"postgresql+psycopg://{settings.DB_USER}:{settings.DB_PASSWORD.get_secret_value()}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}",
    )
    retrieved_docs = load_blog_post()
    logger.info("Chunking and storing vectors...")
    chunk_and_store_vectors(vector_store, retrieved_docs)

    prompt = build_rag_prompt()

    retrieve_docs = partial(retrieve, vector_store=vector_store)
    generate_response = partial(generate, llm=language_model, prompt=prompt)
    graph_builder = StateGraph(State).add_sequence(
        [("retrieve_step", retrieve_docs), ("generate_step", generate_response)]
    )
    graph_builder.add_edge(START, "retrieve_step")
    logger.info("Compiling graph...")
    graph = graph_builder.compile()

    question = {"question": "What is Task Decomposition?"}
    logger.info(f"Invoking graph with question: {question['question']}")
    response = graph.invoke(question)
    logger.info(f"Model answer: {response["answer"]}")


if __name__ == "__main__":
    main()
