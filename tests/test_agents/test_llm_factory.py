import os
import pytest
from unittest.mock import patch
from src.agents.llm_factory import get_llm, get_embeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def test_get_llm_vllm_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1"

def test_get_llm_vllm_multimodal(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = get_llm(agent_type="multimodal")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "llava-hf/llava-1.5-7b-hf"

def test_get_llm_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-1.5-pro"

def test_get_embeddings_vllm_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    embeddings = get_embeddings()
    assert isinstance(embeddings, OpenAIEmbeddings)

def test_get_embeddings_local(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    with patch("langchain_huggingface.HuggingFaceEmbeddings.__init__", return_value=None):
        embeddings = get_embeddings()
        assert isinstance(embeddings, HuggingFaceEmbeddings)

def test_get_embeddings_gemini(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    embeddings = get_embeddings()
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
