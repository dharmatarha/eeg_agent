import pytest
from unittest.mock import patch, MagicMock
import os
from scripts.ingest_rag_data import ingest_scientific_papers

@patch("os.path.exists")
@patch("os.listdir")
@patch("scripts.ingest_rag_data.Chroma")
@patch("scripts.ingest_rag_data.generate_summary")
@patch("scripts.ingest_rag_data.config")
def test_ingest_scientific_papers_docling_success(mock_config, mock_summary, mock_chroma, mock_listdir, mock_exists):
    # Set up mocks
    mock_exists.return_value = True
    mock_listdir.return_value = ["article_1.pdf"]
    
    mock_vector_store = MagicMock()
    # Mock vectorstore.get() to return empty, indicating file needs ingestion
    mock_vector_store.get.return_value = {"ids": []}
    mock_chroma.return_value = mock_vector_store
    
    mock_summary.return_value = "Test Summary of Methods."
    
    # Mock config values
    mock_config.get_val.return_value = "1000"
    
    # Mock Docling conversion and HybridChunker
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_result.document = mock_doc
    mock_converter.convert.return_value = mock_result
    
    mock_chunker = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.text = "This is the text of chunk 1"
    
    # Mock headings and provenance inside chunk metadata
    mock_meta = MagicMock()
    mock_meta.headings = ["Introduction"]
    
    # Mock doc_items/provenance
    mock_item = MagicMock()
    mock_prov = MagicMock()
    mock_prov.page_no = 2
    mock_item.prov = [mock_prov]
    mock_meta.doc_items = [mock_item]
    
    mock_chunk.meta = mock_meta
    mock_chunker.chunk.return_value = [mock_chunk]
    
    with patch("docling.document_converter.DocumentConverter", return_value=mock_converter), \
         patch("docling.chunking.HybridChunker", return_value=mock_chunker):
         
        # Run ingestion
        ingest_scientific_papers(
            articles_dir="/data/articles",
            db_dir="/data/chroma",
            embeddings=MagicMock(),
            llm=MagicMock(),
            force=False
        )
        
    # Verify that Docling convert was called
    mock_converter.convert.assert_called_once_with("/data/articles/article_1.pdf")
    
    # Verify that vector_store.add_documents was called with the document containing headings and summary metadata
    assert mock_vector_store.add_documents.called
    added_docs = mock_vector_store.add_documents.call_args[0][0]
    assert len(added_docs) == 1
    assert "Article: article 1" in added_docs[0].page_content
    assert "Section: Introduction" in added_docs[0].page_content
    assert "Pages: 2" in added_docs[0].page_content
    assert added_docs[0].metadata["global_summary"] == "Test Summary of Methods."
    assert added_docs[0].metadata["source_type"] == "Scientific Paper"


@patch("os.path.exists")
@patch("os.listdir")
@patch("scripts.ingest_rag_data.Chroma")
@patch("scripts.ingest_rag_data.generate_summary")
@patch("scripts.ingest_rag_data.config")
@patch("langchain_community.document_loaders.PyPDFLoader")
def test_ingest_scientific_papers_docling_fallback(mock_pdf_loader, mock_config, mock_summary, mock_chroma, mock_listdir, mock_exists):
    # Set up mocks
    mock_exists.return_value = True
    mock_listdir.return_value = ["article_2.pdf"]
    
    mock_vector_store = MagicMock()
    mock_vector_store.get.return_value = {"ids": []}
    mock_chroma.return_value = mock_vector_store
    
    mock_summary.return_value = "Fallback Summary."
    mock_config.get_val.return_value = "1000"
    
    # Mock PyPDFLoader docs
    mock_loader_instance = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "This is fallback content."
    mock_doc.metadata = {}
    mock_loader_instance.load.return_value = [mock_doc]
    mock_pdf_loader.return_value = mock_loader_instance
    
    # Force docling to raise an error to trigger fallback
    with patch("docling.document_converter.DocumentConverter", side_effect=Exception("Docling crashed")):
        ingest_scientific_papers(
            articles_dir="/data/articles",
            db_dir="/data/chroma",
            embeddings=MagicMock(),
            llm=MagicMock(),
            force=False
        )
        
    # Verify fallback PDF Loader was called
    mock_pdf_loader.assert_called_once_with("/data/articles/article_2.pdf")
    assert mock_vector_store.add_documents.called
    added_docs = mock_vector_store.add_documents.call_args[0][0]
    assert len(added_docs) > 0
    assert added_docs[0].metadata["global_summary"] == "Fallback Summary."
    assert added_docs[0].metadata["source_type"] == "Scientific Paper"
