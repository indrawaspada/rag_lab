#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

print("Testing imports...")

try:
    import os
    print("✓ os")
    import bs4
    print("✓ bs4")
    import requests
    print("✓ requests")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✓ RecursiveCharacterTextSplitter")
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("✓ ChatOpenAI, OpenAIEmbeddings")
    from langchain_community.vectorstores import FAISS
    print("✓ FAISS")
    from langchain_community.document_loaders import WebBaseLoader
    print("✓ WebBaseLoader")
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("✓ ChatMessageHistory")
    from langchain_core.prompts import PromptTemplate
    print("✓ PromptTemplate")
    from langchain_classic.memory import ConversationBufferMemory
    print("✓ ConversationBufferMemory")
    from langchain_classic.chains import ConversationalRetrievalChain
    print("✓ ConversationalRetrievalChain")
    print("\n✅ All imports successful!")
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
