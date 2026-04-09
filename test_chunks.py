#!/usr/bin/env python3
"""Quick test of text chunking on snapy.ai"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

def test_website(url: str):
    """Test website chunking"""
    print(f"\nTesting: {url}")
    print("=" * 60)
    
    try:
        # Fetch HTML
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        print(f"✓ Fetched {len(response.text)} bytes of HTML")
        
        # Save to temp file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
            f.write(response.text)
            temp_path = f.name
        
        # Load with BeautifulSoup
        loader = BSHTMLLoader(temp_path)
        documents = loader.load()
        os.unlink(temp_path)
        
        # Check extracted content
        total_text = "\n".join([doc.page_content for doc in documents])
        print(f"✓ Extracted {len(documents)} documents")
        print(f"✓ Total text: {len(total_text)} characters")
        print(f"\nFirst 300 chars of content:\n{total_text[:300]}...\n")
        
        # Split chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        print(f"✓ Generated {len(chunks)} chunks")
        print(f"  Chunk size config: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
        
        if chunks:
            sizes = [len(c.page_content) for c in chunks]
            print(f"  Average chunk size: {sum(sizes) // len(sizes)} chars")
            print(f"  Min chunk size: {min(sizes)} chars")
            print(f"  Max chunk size: {max(sizes)} chars")
            
            print(f"\nChunk preview:")
            for i, chunk in enumerate(chunks[:3], 1):
                content = chunk.page_content[:100]
                print(f"  [{i}] {content}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_website("https://snapy.ai")
