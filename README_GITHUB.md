# RAG (Retrieval-Augmented Generation) Projects

Koleksi project RAG menggunakan LangChain dan OpenAI untuk berbagai use case.

## Struktur Project

```
rag/
├── 2_vector_db/              # Vector database implementations
├── 3_building_simple_rag_pipeline/   # Basic RAG chatbots
│   ├── memory_based.py       # Website chatbot dengan memory
│   ├── website_chatbot.py    # Web scraping RAG pipeline
│   ├── csv_bot.py           # CSV data RAG
│   └── tesla_motors_data.csv # Sample dataset
├── 4_advanced_RAG/           # Advanced RAG techniques
│   ├── multi-vector.py      # Multi-vector RAG
│   └── query-expansion.py   # Query expansion for RAG
├── 5_RAG_Evaluation/         # RAG evaluation metrics
├── 6_sql_rag/               # SQL database RAG
│   └── sql_rag_project.py   # Tesla database SQL agent
├── 7_multimedia_pdf/         # PDF processing
├── 8_deploying_st_app/       # Streamlit deployment
├── 9_prompt_caching/         # Prompt caching with OpenAI/Claude
└── 10_other_projects/        # Miscellaneous projects
```

## Requirements

- Python 3.13+
- Virtual Environment (ragenv)
- OpenAI API Key
- Pinecone API Key (optional)

## Installation

1. **Activate Virtual Environment:**
```bash
cd rag
.\ragenv\Scripts\Activate.ps1
```

2. **Install Dependencies:**
```bash
pip install -r Requirements.txt
```

3. **Setup Environment Variables:**
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Quick Start

### Website Chatbot
```bash
python 3_building_simple_rag_pipeline\memory_based.py
```

### CSV Bot
```bash
python 3_building_simple_rag_pipeline\csv_bot.py
```

### SQL RAG Agent
```bash
python 6_sql_rag\sql_rag_project.py
```

### Query Expansion
```bash
python 4_advanced_RAG\query-expansion.py
```

## Features

- ✅ Website scraping and RAG
- ✅ CSV data RAG
- ✅ SQL database RAG
- ✅ Multi-vector retrieval
- ✅ Query expansion
- ✅ Memory-based conversations
- ✅ PDF processing
- ✅ Prompt caching

## Key Technologies

- **LangChain**: Framework untuk LLM applications
- **OpenAI**: GPT-4o-mini and Embeddings
- **FAISS**: Vector similarity search
- **BeautifulSoup**: Web scraping
- **Streamlit**: Web deployment

## Project Status

✅ Basic RAG implementations working
✅ Advanced RAG techniques implemented
✅ Database integrations complete
🔄 Deployment optimization in progress

## Contributing

Feel free to fork dan submit pull requests untuk improvements!

## License

MIT License - feel free to use this project for your own purposes.

## Contact

For questions and support, please open an issue on GitHub.
