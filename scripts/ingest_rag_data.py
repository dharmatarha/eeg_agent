import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.prompts import PromptTemplate
from src.agents.llm_factory import get_llm, get_embeddings

def generate_summary(doc_text: str, llm) -> str:
    prompt = PromptTemplate.from_template(
        "Please provide a concise but comprehensive summary of the methodologies, parameters, and key findings in the following scientific paper excerpt:\n\n{text}\n\nSummary:"
    )
    chain = prompt | llm
    # To avoid context limits, we just summarize the first 10,000 characters (typically Abstract + Intro + Methods)
    summary = chain.invoke({"text": doc_text[:10000]})
    return summary.content

def ingest_scientific_papers(articles_dir, db_dir, embeddings, llm):
    print("\n--- Processing Scientific Papers (PDF) ---")
    if not os.path.exists(articles_dir):
        print("Articles directory does not exist.")
        return
        
    pdf_files = [os.path.join(articles_dir, f) for f in os.listdir(articles_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in articles/.")
        return

    vector_store = Chroma(
        collection_name="neuroimage_methods",
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    for pdf_path in pdf_files:
        print(f"Ingesting {os.path.basename(pdf_path)} with Layout-Aware Chunking...")
        # Layout aware chunking by title/section
        try:
            loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="fast", chunking_strategy="by_title")
            chunks = loader.load()
        except Exception as e:
            print(f"Warning: UnstructuredLoader failed, falling back to basic loader for {pdf_path}. Error: {e}")
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            chunks = splitter.split_documents(docs)

        if not chunks:
            continue
            
        # Generate Global Summary from the full text
        full_text = "\n".join([c.page_content for c in chunks])
        print("Generating Methods Summary using LLM...")
        try:
            summary = generate_summary(full_text, llm)
        except Exception as e:
            print(f"Warning: Failed to generate summary: {e}")
            summary = "Summary unavailable."
            
        # Inject metadata into every chunk
        for chunk in chunks:
            chunk.metadata['global_summary'] = summary
            chunk.metadata['source_type'] = 'Scientific Paper'
            
        vector_store.add_documents(chunks)
        print(f"Successfully added {len(chunks)} chunks for {os.path.basename(pdf_path)}.")

def ingest_books(books_dir, db_dir, embeddings):
    print("\n--- Processing Books (PDF) ---")
    if not os.path.exists(books_dir):
        print("Books directory does not exist.")
        return
        
    pdf_files = [os.path.join(books_dir, f) for f in os.listdir(books_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No book PDF files found in books/.")
        return

    vector_store = Chroma(
        collection_name="neuroimage_methods",
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    for pdf_path in pdf_files:
        print(f"Ingesting book {os.path.basename(pdf_path)}...")
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            chunks = splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading book {pdf_path}: {e}")
            continue

        if not chunks:
            continue

        # For books, generating LLM summaries for every chunk or full book is impractical.
        # We store the book title/filename as the global summary reference.
        book_title = os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ")
        summary = f"Reference textbook: {book_title}"

        for chunk in chunks:
            chunk.metadata['global_summary'] = summary
            chunk.metadata['source_type'] = 'Book'

        vector_store.add_documents(chunks)
        print(f"Successfully added {len(chunks)} chunks for book {os.path.basename(pdf_path)}.")

def ingest_api_docs(api_docs_dir, db_dir, embeddings):
    print("\n--- Processing API Documentation (TXT/MD) ---")
    if not os.path.exists(api_docs_dir):
        print("API docs directory does not exist.")
        return
        
    txt_loader = DirectoryLoader(api_docs_dir, glob="**/*.txt")
    md_loader = DirectoryLoader(api_docs_dir, glob="**/*.md")
    
    docs = txt_loader.load() + md_loader.load()
    
    if not docs:
        print("No TXT/MD files found in mne_python_docs/.")
        return
        
    print(f"Found {len(docs)} API documentation files. Initializing ParentDocumentRetriever...")
    
    # Hierarchical indexing
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # Small child chunks for strict function-level matching
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    vectorstore = Chroma(
        collection_name="neuroimage_api",
        embedding_function=embeddings,
        persist_directory=db_dir
    )
    
    store_dir = os.path.join(db_dir, "docstore")
    os.makedirs(store_dir, exist_ok=True)
    store = LocalFileStore(store_dir)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(docs, ids=None)
    print(f"Successfully added {len(docs)} API documents using Hierarchical Indexing.")

def main():
    load_dotenv()
    
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_docs"))
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_data"))
    
    articles_dir = os.path.join(docs_dir, "articles")
    books_dir = os.path.join(docs_dir, "books")
    api_docs_dir = os.path.join(docs_dir, "mne_python_docs")
    
    # Create subdirectories if they don't exist
    for d in [docs_dir, articles_dir, books_dir, api_docs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory {d}.")
        
    print("Initializing LLM and Embedding Models...")
    llm = get_llm(agent_type="text", temperature=0.0)
    embeddings = get_embeddings()
    
    # 1. Scientific Papers (Layout Chunking + Summaries)
    ingest_scientific_papers(articles_dir, db_dir, embeddings, llm)
    
    # 2. Books (Recursive Chunking)
    ingest_books(books_dir, db_dir, embeddings)
    
    # 3. API Documentation (Hierarchical Indexing)
    ingest_api_docs(api_docs_dir, db_dir, embeddings)
    
    print(f"\nSuccessfully populated Dual-Strategy RAG Database at {db_dir}")

if __name__ == "__main__":
    main()
