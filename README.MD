# 📚 ScholarAI - AI-Powered Research Companion

> An experimental AI research assistant prototype designed to explore retrieval-augmented generation (RAG) techniques for academic paper discovery and analysis .

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)

---

## 🎯 Problem Statement

Researchers spend countless hours reading papers, extracting key information, and trying to connect ideas across multiple sources. Current tools either:
- Require expensive API subscriptions (ChatGPT, Claude)
- Lack domain-specific understanding of scientific content
- Generate hallucinated or inaccurate information

ScholarAI explores a free, local-first approach to building a research assistant powered by models trained on scientific literature.

---

## ✨ Features

### 📖 Paper Management
- 🔍 **Search ArXiv** - Find relevant papers using natural language queries
- 📥 **Auto-download & Parse** - Automatic PDF download and text extraction
- 🏷️ **Smart Organization** - Tag, categorize, and track reading status
- 📊 **Library Management** - Keep all your research in one place

### 🤖 AI-Powered Analysis
- 💬 **Question Answering** - Ask questions about your paper library and get grounded answers
- 📝 **Intelligent Summaries** - Extractive summaries that preserve accuracy
- 🔗 **Source Attribution** - Every answer is backed by specific paper citations
- 🎯 **Confidence Scoring** - Real relevance scores (0-10) for answer quality

### 🔬 Research-Grade Technology
- **SPECTER2 Embeddings** - Trained on 100M+ scientific papers by AllenAI
- **BGE Reranking** - State-of-the-art reranking for maximum relevance
- **Extractive Approach** - Designed to reduce hallucinations by avoiding text generation
- **100% Free & Local** - All models run on your computer

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                  ┌─────▼────┐
    │ Database │                  │   RAG    │
    │ (SQLite) │                  │  System  │
    └──────────┘                  └─────┬────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────▼──────┐     ┌──────▼──────┐    ┌──────▼──────┐
              │  SPECTER2  │     │    ChromaDB  │    │     BGE     │
              │ Embeddings │     │ Vector Store │    │  Reranker   │
              └────────────┘     └──────────────┘    └─────────────┘
```

### Key Components

1. **Data Collection Layer**
   - ArXiv API integration for paper discovery
   - PyMuPDF for PDF parsing and text extraction
   - Automatic metadata extraction

2. **Storage Layer**
   - **SQLite Database** - Paper metadata, notes, reading status
   - **ChromaDB Vector Store** - Semantic embeddings for similarity search

3. **AI/ML Layer**
   - **SPECTER2** - Scientific paper embeddings (AllenAI)
   - **BGE Reranker** - Cross-encoder for relevance scoring
   - **Extractive QA** - Sentence extraction (no generation)

4. **Interface Layer**
   - Streamlit web application
   - Real-time search and Q&A
   - Interactive visualization

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- 8GB RAM minimum (16GB recommended)
- 5GB disk space (for models and data)
- Windows, macOS, or Linux

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/scholarai.git
cd scholarai

# 2. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create data directories
mkdir -p data/papers data/database data/embeddings

# 5. Run the application
streamlit run app/streamlit_app.py
```

**First run:** Models will download automatically (~500MB). This only happens once!

---

## 💡 Usage

### 1. Search and Add Papers

```
1. Navigate to "🔍 Search Papers"
2. Enter your research query (e.g., "graph neural networks")
3. Browse results and click "Add to Library"
4. Paper is automatically downloaded, parsed, and indexed
```

### 2. Ask Questions

```
1. Go to "💬 Ask Questions"
2. Select a paper or "All papers"
3. Ask: "What methodology do they use?"
4. Get extractive answer with confidence score and sources
```

### 3. Generate Summaries

```
1. Open "📊 My Library"
2. Expand a paper
3. Click "🤖 Summarize"
4. Receive extractive summary of key contributions
```

---

## 🔬 Technical Deep Dive

### Why SPECTER2?

SPECTER2 (Scientific Paper Embeddings) was specifically trained on scientific papers, unlike generic models:

- **Training Data**: 100M+ scientific papers
- **Created by**: Allen Institute for AI
- **Used by**: Semantic Scholar, academic search engines
- **Advantage**: Understands citations, technical terminology, paper structure

### Extractive vs Generative QA

| Approach | Accuracy | Hallucination | Speed |
|----------|----------|---------------|-------|
| **Generative** (FLAN-T5, GPT) | Medium | High risk | Slow |
| **Extractive** (Our approach) | High | Zero | Fast |

**Our Choice:** Extractive approach extracts actual sentences from papers rather than generating new text. Goal: Provide fully grounded answers extracted directly from source papers.

### Two-Stage Retrieval

1. **Stage 1 - Broad Retrieval**
   - SPECTER2 semantic search retrieves top 20 candidate chunks
   - Fast approximate search using cosine similarity

2. **Stage 2 - Precise Reranking**
   - BGE CrossEncoder scores each candidate (0-10)
   - Selects top 5-6 most relevant chunks
   - Uses only highest-quality context

**Result:** Much better answer quality than single-stage retrieval.




---

## 🛠️ Technology Stack

### Backend
- **Python 3.10+** - Core language
- **SentenceTransformers** - Embedding models
- **ChromaDB** - Vector database
- **SQLAlchemy** - SQL ORM
- **PyMuPDF** - PDF parsing

### Frontend
- **Streamlit** - Web interface
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation

### ML/AI
- **SPECTER2** - Scientific embeddings
- **BGE Reranker** - Cross-encoder reranking
- **PyTorch** - Deep learning framework

---

## 🚧 Current Limitations & Future Work

### Known Limitations

1. **Model Quality**
   - Extractive approach limited to sentence-level extraction
   - Summary quality depends on paper structure
   - Works best with well-formatted papers

2. **Performance**
   - First-time model download takes 5-10 minutes
   - CPU inference slower than GPU
   - Large papers (100+ pages) take longer

3. **Scope**
   - Currently supports ArXiv only
   - English-language papers primarily
   - PDF parsing can fail on complex layouts

### Planned Improvements

- [ ] Multi-source support (Google Scholar, PubMed, Semantic Scholar)
- [ ] GPU acceleration for faster inference
- [ ] Fine-tuned models on specific domains
- [ ] Citation network visualization
- [ ] Collaborative features (shared libraries)
- [ ] Export to reference managers (Zotero, Mendeley)
- [ ] Advanced query language (boolean operators, filters)
- [ ] Paper recommendation system
- [ ] Mobile-responsive design

---

## 🤝 Contributing

This is a learning project and contributions are welcome! Areas where help is needed:

- **Model optimization** - Quantization, distillation for speed
- **UI/UX improvements** - Better visualization, mobile support
- **Testing** - Unit tests, integration tests
- **Documentation** - More examples, tutorials
- **New features** - See roadmap above



---



## 👤 Author

**Your Name**
- GitHub: [@vhagar2001](https://github.com/vhagar2001/scholarai)
- LinkedIn: [Mansouri Mohamed Fadhel](https://www.linkedin.com/in/mohamed-fadhel-mansouri/)
- Email: mohamedfadhelmansouri01@gmail.com

---

## 🙏 Acknowledgments

- **Allen Institute for AI** - For SPECTER2 and inspiring this project
- **BAAI** - For the BGE reranker model
- **ArXiv** - For providing open access to research papers
- **Open-source community** - For the amazing tools that made this possible

---
## 🧪 Development Stage

ScholarAI is currently:

- 🧠 A personal research experiment in applied NLP and RAG systems
- 🏗️ Under active architectural iteration
- 🔍 Being evaluated for retrieval quality and system performance
- 🧪 Not production-ready

The focus at this stage is learning, experimentation, and architectural validation rather than production deployment.
---



## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover it.

---

**Built with ❤️ for researchers, by a researcher**

*Last updated: February 2026*