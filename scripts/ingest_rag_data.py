import os
import sys
import uuid
import argparse
import logging
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
try:
    from langchain.storage import LocalFileStore, create_kv_docstore
except ModuleNotFoundError:
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
try:
    from langchain.retrievers import ParentDocumentRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import PromptTemplate
from src.agents.llm_factory import get_llm, get_embeddings
from src.utils.logging_config import setup_logging
from src import config

logger = logging.getLogger("eeg_agent.ingest")

def generate_summary(doc_text: str, llm) -> str:
    prompt = PromptTemplate.from_template(
        "Please provide a concise but comprehensive summary of the methodologies, parameters, and key findings in the following scientific paper excerpt:\n\n{text}\n\nSummary:"
    )
    chain = prompt | llm
    
    # To avoid context limits, we just summarize the configured character threshold
    max_chars = int(config.get_val("ingestion.articles.summary_max_chars"))
    summary = chain.invoke({"text": doc_text[:max_chars]})
    content = summary.content
    if isinstance(content, list):
        extracted = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                extracted.append(part["text"])
            elif isinstance(part, str):
                extracted.append(part)
            elif hasattr(part, "text"):
                extracted.append(part.text)
        return "".join(extracted)
    return str(content)


def ingest_scientific_papers(articles_dir, db_dir, embeddings, llm, force=False):
    logger.info("Processing Scientific Papers (PDF)...")
    if not os.path.exists(articles_dir):
        logger.warning("Articles directory '%s' does not exist.", articles_dir)
        return
        
    pdf_files = [os.path.join(articles_dir, f) for f in os.listdir(articles_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.info("No PDF files found in articles/.")
        return

    vector_store = Chroma(
        collection_name="neuroimage_methods",
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    for pdf_path in pdf_files:
        file_basename = os.path.basename(pdf_path)
        
        # Check if already ingested to support incremental ingestion
        if not force:
            res = vector_store.get(ids=[f"{file_basename}_chunk_0"])
            if res and res.get('ids'):
                logger.info("Skipping %s (already ingested).", file_basename)
                continue

        logger.info("Ingesting %s with Layout-Aware Chunking...", file_basename)
        
        # Standard fast and memory-safe PDF loading
        from langchain_community.document_loaders import PyPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            chunk_size = int(config.get_val("ingestion.articles.chunk_size"))
            chunk_overlap = int(config.get_val("ingestion.articles.chunk_overlap"))
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)
        except Exception as e:
            logger.error("Error loading %s: %s", pdf_path, e)
            continue

        if not chunks:
            continue
            
        # Generate Global Summary from the full text
        full_text = "\n".join([c.page_content for c in chunks])
        logger.info("Generating Methods Summary using LLM for %s...", file_basename)
        try:
            summary = generate_summary(full_text, llm)
        except Exception as e:
            logger.warning("Failed to generate summary: %s", e)
            summary = "Summary unavailable."
            
        # Inject metadata into every chunk
        for chunk in chunks:
            chunk.metadata['global_summary'] = summary
            chunk.metadata['source_type'] = 'Scientific Paper'
            
        # Assign deterministic IDs to prevent duplicates
        ids = [f"{file_basename}_chunk_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)
        logger.info("Successfully added %d chunks for %s.", len(chunks), file_basename)


def ingest_books(books_dir, db_dir, embeddings, force=False):
    logger.info("Processing Books (PDF)...")
    if not os.path.exists(books_dir):
        logger.warning("Books directory '%s' does not exist.", books_dir)
        return
        
    pdf_files = [os.path.join(books_dir, f) for f in os.listdir(books_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.info("No book PDF files found in books/.")
        return

    vector_store = Chroma(
        collection_name="neuroimage_methods",
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    for pdf_path in pdf_files:
        file_basename = os.path.basename(pdf_path)
        
        # Check if already ingested to support incremental ingestion
        if not force:
            res = vector_store.get(ids=[f"{file_basename}_chunk_0"])
            if res and res.get('ids'):
                logger.info("Skipping book %s (already ingested).", file_basename)
                continue

        logger.info("Ingesting book %s...", file_basename)
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            chunk_size = int(config.get_val("ingestion.books.chunk_size"))
            chunk_overlap = int(config.get_val("ingestion.books.chunk_overlap"))
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)
        except Exception as e:
            logger.error("Error loading book %s: %s", pdf_path, e)
            continue

        if not chunks:
            continue

        # For books, generating LLM summaries for every chunk or full book is impractical.
        # We store the book title/filename as the global summary reference.
        book_title = file_basename.replace(".pdf", "").replace("_", " ")
        summary = f"Reference textbook: {book_title}"

        for chunk in chunks:
            chunk.metadata['global_summary'] = summary
            chunk.metadata['source_type'] = 'Book'

        # Assign deterministic IDs to prevent duplicates
        ids = [f"{file_basename}_chunk_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)
        logger.info("Successfully added %d chunks for book %s.", len(chunks), file_basename)


def ingest_api_docs(api_docs_dir, db_dir, embeddings, force=False):
    logger.info("Processing API Documentation (TXT/MD)...")
    if not os.path.exists(api_docs_dir):
        logger.warning("API docs directory '%s' does not exist.", api_docs_dir)
        return
        
    txt_loader = DirectoryLoader(api_docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(api_docs_dir, glob="**/*.md", loader_cls=TextLoader)
    
    docs = txt_loader.load() + md_loader.load()
    
    if not docs:
        logger.info("No TXT/MD files found in mne_python_docs/.")
        return
        
    logger.info("Found %d API documentation files. Initializing parent document splitting...", len(docs))
    
    # Hierarchical indexing splitters from config
    parent_size = int(config.get_val("ingestion.api_docs.parent_chunk_size"))
    parent_overlap = int(config.get_val("ingestion.api_docs.parent_chunk_overlap"))
    child_size = int(config.get_val("ingestion.api_docs.child_chunk_size"))
    child_overlap = int(config.get_val("ingestion.api_docs.child_chunk_overlap"))

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=parent_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap)
    
    # Pre-split into parent documents to assign deterministic chunk-level IDs
    parent_docs = parent_splitter.split_documents(docs)
    logger.info("Split %d files into %d parent documents.", len(docs), len(parent_docs))
    
    vectorstore = Chroma(
        collection_name="neuroimage_api",
        embedding_function=embeddings,
        persist_directory=db_dir
    )
    
    store_dir = os.path.join(db_dir, "docstore")
    os.makedirs(store_dir, exist_ok=True)
    fs = LocalFileStore(store_dir)
    store = create_kv_docstore(fs)
    
    # We pass parent_splitter=None because we already split them manually
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=None,
    )
    
    # Generate deterministic parent IDs based on filename and content hash
    ids = []
    filtered_docs = []
    for doc in parent_docs:
        source_path = doc.metadata.get('source', '')
        if not source_path:
            continue
        file_basename = os.path.basename(source_path)
        
        # Generate stable content hash
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:12]
        parent_id = f"api_parent_{file_basename}_{content_hash}"
        
        # Check if already present in docstore to support incremental ingestion
        if not force:
            values = store.mget([parent_id])
            if values and values[0] is not None:
                logger.debug("Skipping parent document chunk %s (already ingested).", parent_id)
                continue
                
        ids.append(parent_id)
        filtered_docs.append(doc)

    if filtered_docs:
        logger.info("Ingesting %d new parent document chunks in batches...", len(filtered_docs))
        batch_size = 500
        for start_idx in range(0, len(filtered_docs), batch_size):
            end_idx = min(start_idx + batch_size, len(filtered_docs))
            batch_docs = filtered_docs[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            logger.info("Ingesting parent documents batch [%d-%d/%d]...", start_idx, end_idx, len(filtered_docs))
            retriever.add_documents(batch_docs, ids=batch_ids)
        logger.info("Successfully added %d new API document chunks using Hierarchical Indexing.", len(filtered_docs))
    else:
        logger.info("All API documents are already ingested.")


def main():
    load_dotenv(override=True)
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Ingest RAG documents into ChromaDB.")
    parser.add_argument(
        "--category",
        choices=["all", "articles", "books", "api"],
        default="all",
        help="Specify the category of documents to ingest (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of already processed documents"
    )
    args = parser.parse_args()
    
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_docs"))
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_data"))
    
    articles_dir = os.path.join(docs_dir, "articles")
    books_dir = os.path.join(docs_dir, "books")
    api_docs_dir = os.path.join(docs_dir, "mne_python_docs")
    
    # Create subdirectories if they don't exist
    for d in [docs_dir, articles_dir, books_dir, api_docs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info("Created directory %s.", d)
        
    logger.info("Initializing LLM and Embedding Models...")
    llm = get_llm(agent_type="text")
    embeddings = get_embeddings()
    
    # 1. Scientific Papers (Layout Chunking + Summaries)
    if args.category in ["all", "articles"]:
        ingest_scientific_papers(articles_dir, db_dir, embeddings, llm, force=args.force)
    
    # 2. Books (Recursive Chunking)
    if args.category in ["all", "books"]:
        ingest_books(books_dir, db_dir, embeddings, force=args.force)
    
    # 3. API Documentation (Hierarchical Indexing)
    if args.category in ["all", "api"]:
        ingest_api_docs(api_docs_dir, db_dir, embeddings, force=args.force)
    
    logger.info("Successfully populated Dual-Strategy RAG Database at %s", db_dir)

if __name__ == "__main__":
    main()
