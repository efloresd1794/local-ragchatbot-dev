# Research Paper Navigator - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system for querying and understanding research papers, built with LangChain, ChromaDB, and local LLMs via Ollama.

## Project Status: Phase 1 - POC & Experimentation ✅

This repository contains the **proof-of-concept and experimentation phase** of the RAG chatbot development. The notebooks demonstrate core functionality, compare different approaches, and establish comprehensive evaluation metrics.

**Phase 2** (Production Implementation) will follow, transforming these notebooks into a production-ready chatbot with modular architecture, API endpoints, and deployment capabilities.

---

## 📚 Phase 1: What's Been Built

### Notebooks Overview

#### 1. `01_rag_poc.ipynb` - Core RAG Pipeline
**Purpose**: Establish baseline RAG functionality and test end-to-end workflow

**Features**:
- PDF ingestion and parsing (PyMuPDF)
- Recursive chunking strategy (512 chars, 128 overlap)
- Embedding generation (sentence-transformers/all-MiniLM-L6-v2)
- Vector storage (ChromaDB)
- Three retrieval methods:
  - Dense (vector similarity)
  - Sparse (BM25)
  - Hybrid (Reciprocal Rank Fusion)
- LLM generation (Ollama llama3.1:8b)
- Basic evaluation metrics
- Tested with sample papers (Attention Is All You Need, BERT)

**Key Learnings**:
- Hybrid retrieval outperforms single methods
- Recursive chunking preserves context better than fixed-size
- 384-dim embeddings provide good balance of speed/quality

---

#### 2. `02_multi_embedding_comparison.ipynb` - Model Comparison
**Purpose**: Compare multiple embedding models across all retrieval methods

**Features**:
- Dual embedding model testing:
  - `sentence-transformers/all-MiniLM-L6-v2` (fast baseline)
  - `BAAI/bge-small-en-v1.5` (better quality, same size)
- Side-by-side retrieval comparison (dense, sparse, hybrid)
- Performance benchmarking (latency, accuracy)
- Overlap analysis between models
- Batch testing with aggregate statistics
- Visual comparisons (timing charts, distributions)

**Key Learnings**:
- BGE models show better semantic understanding
- Dense retrieval time varies significantly by embedding model
- BM25 provides consistent baseline across embeddings
- Hybrid approach benefits from model diversity

---

#### 3. `03_rag_evaluation_metrics.ipynb` - Comprehensive Evaluation
**Purpose**: Production-grade evaluation framework for RAG quality and performance

**Metrics Implemented**:

**Performance**:
- Latency tracking (avg, median, p95, p99)
- Success rate monitoring
- Throughput measurement
- SLA compliance checking

**Quality**:
- **Faithfulness**: Answer grounding in context (0.0-1.0)
  - N-gram overlap detection
  - Sentence-level verification
  - Hallucination identification
- **Answer Relevance**: Query-answer alignment (0.0-1.0)
  - Key term coverage
  - Semantic similarity
- **Context Relevance**: Retrieval quality (0.0-1.0)
  - Source document scoring
  - Query-context matching
- **Contradiction Detection**: Identify conflicting claims
  - Negation pattern analysis

**Error Tracking**:
- Error categorization by type
- Failure rate analysis
- Edge case identification
- Detailed logging

**Features**:
- 17-query test dataset (factual, conceptual, edge cases)
- Automated hallucination detection
- A/B testing framework
- Comprehensive reporting (JSON, CSV, visualizations)
- Configurable thresholds (faithfulness > 0.6, relevance > 0.5)

**Key Learnings**:
- Hallucination detection is critical for research applications
- Context relevance strongly correlates with answer quality
- Edge case handling ("don't know" responses) improves trustworthiness
- Automated metrics enable rapid iteration

---

## 🏗️ Technology Stack

### Core Framework
- **LangChain**: RAG orchestration, document processing, chains
- **ChromaDB**: Vector database for embeddings storage
- **Ollama**: Local LLM inference (llama3.1:8b)

### Embeddings & Retrieval
- **HuggingFace Transformers**: Embedding models
  - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
  - BAAI/bge-small-en-v1.5 (384 dims)
- **BM25Okapi**: Sparse retrieval (keyword matching)

### Document Processing
- **PyMuPDF**: High-quality PDF parsing
- **LangChain Text Splitters**: Recursive chunking

### Data & Visualization
- **NumPy, Pandas**: Data analysis
- **Matplotlib, Seaborn**: Visualization
- **tqdm**: Progress tracking

---

## 📁 Project Structure

```
dev-chatbot-rag/
├── notebooks/
│   ├── 01_rag_poc.ipynb                    # Core RAG pipeline
│   ├── 02_multi_embedding_comparison.ipynb # Embedding model comparison
│   └── 03_rag_evaluation_metrics.ipynb     # Comprehensive evaluation
├── data/
│   ├── raw/                                # Input PDFs
│   ├── processed/                          # Processed documents
│   ├── vector_store/                       # ChromaDB collections
│   └── eval_results/                       # Evaluation outputs (JSON, CSV)
├── src/                                    # (Phase 2 - Production code)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.11+**
2. **Ollama** installed with llama3.1:8b model
   ```bash
   ollama pull llama3.1:8b
   ```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd dev-chatbot-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community langchain-ollama
pip install chromadb sentence-transformers
pip install PyMuPDF rank-bm25
pip install numpy pandas matplotlib seaborn tqdm jupyter
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ and open:
# - 01_rag_poc.ipynb (start here)
# - 02_multi_embedding_comparison.ipynb
# - 03_rag_evaluation_metrics.ipynb
```

### Sample Data

The notebooks automatically download sample papers from ArXiv:
- Attention Is All You Need (Transformer paper)
- BERT: Pre-training of Deep Bidirectional Transformers

To use your own PDFs, place them in `data/raw/`.

---

## 📊 Key Results from Phase 1

### Performance Benchmarks
- **Average Latency**: ~2-4 seconds per query
- **Success Rate**: 95%+ on well-formed queries
- **Retrieval Accuracy**: Hybrid > Dense > Sparse

### Quality Metrics
- **Faithfulness**: 0.75 avg (on factual questions)
- **Answer Relevance**: 0.82 avg
- **Context Relevance**: 0.71 avg
- **Hallucination Rate**: <5% (with proper prompting)

### Optimal Configuration (Identified)
- **Embedding**: BAAI/bge-small-en-v1.5
- **Retrieval**: Hybrid (α=0.5)
- **Chunk Size**: 512 chars with 128 overlap
- **Top-K**: 5 documents
- **LLM Temperature**: 0.1 (factual responses)

---

## 🎯 Next Steps: Phase 2 - Production Implementation

### Planned Features

#### 1. **Modular Architecture**
- Refactor notebooks into `src/` modules
- Separation of concerns (ingestion, retrieval, generation, evaluation)
- Configuration management (YAML/environment variables)
- Dependency injection for testability

#### 2. **API & Interface**
- FastAPI backend with RESTful endpoints
- WebSocket support for streaming responses
- Gradio/Streamlit UI for user interaction
- Authentication & rate limiting

#### 3. **Advanced Features**
- **Multi-document support**: Query across paper collections
- **Citation tracking**: Link answers to specific paper sections
- **Conversation memory**: Multi-turn dialogue support
- **Query refinement**: Suggest follow-up questions
- **Export functionality**: Save Q&A sessions

#### 4. **Production Infrastructure**
- Docker containerization
- Logging & monitoring (Prometheus, Grafana)
- Continuous evaluation pipeline
- A/B testing framework
- Database migrations (Alembic)

#### 5. **Scalability**
- Batch document processing
- Vector store optimization
- GPU acceleration for embeddings
- Caching layer (Redis)
- Async processing queues

#### 6. **Quality Assurance**
- Unit & integration tests (pytest)
- CI/CD pipeline (GitHub Actions)
- Pre-commit hooks
- Code coverage (>80%)
- Type hints & linting

#### 7. **Deployment**
- Cloud deployment options (AWS, GCP, Azure)
- Kubernetes orchestration
- Auto-scaling configuration
- Health checks & circuit breakers

---

## 📖 Documentation Roadmap (Phase 2)

- API documentation (OpenAPI/Swagger)
- Architecture diagrams
- Deployment guide
- User manual
- Contributing guidelines
- Performance tuning guide

---

## 🤝 Contributing

This is currently in POC phase. Contributions welcome after Phase 2 production implementation begins.

---

## 📄 License

TBD

---

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)

---

**Built by**: Edson Flores
**Last Updated**: 2025-11-14
**Status**: Phase 1 Complete ✅ | Phase 2 In Planning 🚧
