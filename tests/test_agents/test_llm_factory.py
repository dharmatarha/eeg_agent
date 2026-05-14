import os
import pytest
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
    embeddings = get_embeddings()
    assert isinstance(embeddings, HuggingFaceEmbeddings)

def test_get_embeddings_gemini(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    embeddings = get_embeddings()
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
