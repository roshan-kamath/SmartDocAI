import os
import gc
import shutil
import time

from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader,
    UnstructuredWordDocumentLoader, JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"

_vectordb = None
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def release_vectordb():
    global _vectordb
    if _vectordb is not None:
        try:
            _vectordb._client.close()
        except Exception:
            pass
        _vectordb = None
    gc.collect()


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf":  PyPDFLoader,
        ".csv":  CSVLoader,
        ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
        ".doc":  UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".json": lambda p: JSONLoader(p, jq_schema=".[]", text_content=False),
        ".md":   lambda p: TextLoader(p, encoding="utf-8"),
    }
    loader_cls = loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path).load()


def ingest_document(file_path: str):
    global _vectordb

    release_vectordb()
    time.sleep(0.4)

    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            pass

    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    _vectordb = Chroma.from_documents(
        chunks, get_embeddings(), persist_directory=CHROMA_DIR
    )
    return len(chunks)


def get_qa_chain():
    global _vectordb

    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings()
        )

    retriever = _vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 18, "lambda_mult": 0.65}
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly accurate document assistant. You answer questions ONLY using the document context provided below.

STRICT RULES:
1. Answer directly and concisely — no preamble, no filler phrases
2. Use bullet points for lists, steps, or enumerations
3. Always include exact numbers, dates, names, and figures when they appear in the context
4. Never fabricate or assume information not present in the context
5. Never say "based on the context", "the document states", or "according to the passage"
6. Never say "I don't have enough information" if relevant content exists in the context — use what's there
7. If the topic is genuinely absent from the document, say exactly: "This topic is not covered in the uploaded document."
8. Format your answer clearly — short paragraphs or bullets depending on what fits best

DOCUMENT CONTEXT:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        sections = []
        for i, doc in enumerate(docs):
            sections.append(f"[Section {i+1}]\n{doc.page_content.strip()}")
        return "\n\n".join(sections)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain