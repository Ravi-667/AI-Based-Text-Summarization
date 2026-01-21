# AI-Based Text Summarization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Tests](https://github.com/Ravi-667/AI-Based-Text-Summarization/workflows/CI%2FCD%20Pipeline/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-check_CI-blue)

An intelligent text summarization system using state-of-the-art Natural Language Processing (NLP) techniques. This project implements both **extractive** and **abstractive** summarization methods to condense long documents such as news articles, research papers, and other text-heavy content.

## ✨ Features

- 📊 **Extractive Summarization** - Selects the most important sentences using BERT embeddings
- ✨ **Abstractive Summarization** - Generates new paraphrased summaries using T5 models
- 🎯 **Multiple Scoring Methods** - TF-IDF, TextRank, LexRank, and combined approaches
- 📈 **Evaluation Metrics** - ROUGE, BLEU, semantic similarity, and readability scores
- 🌐 **Web Interface** - Interactive Streamlit UI for easy use
- 💻 **CLI Application** - Command-line interface for batch processing
- 📁 **Multi-format Support** - Process TXT, PDF, and DOCX files
- ⚡ **GPU Acceleration** - Automatic GPU detection and utilization
- 🧪 **Comprehensive Tests** - Unit tests for all major components

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ravi-667/AI-Based-Text-Summarization.git
cd AI-Based-Text-Summarization
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Usage

#### 🌐 Web Interface (Recommended)

Launch the interactive Streamlit web app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### 💻 Command Line Interface

**Extractive Summarization:**
```bash
python main.py --input data/samples/sample_news.txt --method extractive --ratio 0.3
```

**Abstractive Summarization:**
```bash
python main.py --input data/samples/sample_research.txt --method abstractive --max-length 150
```

**Compare Both Methods:**
```bash
python main.py --input article.txt --method both --evaluate
```

**Custom Options:**
```bash
python main.py --input document.txt \
  --method extractive \
  --num-sentences 5 \
  --scoring combined \
  --output summary.txt \
  --evaluate
```

#### 🐍 Python API

```python
from src import ExtractiveSummarizer, AbstractiveSummarizer

# Extractive summarization
ext_summarizer = ExtractiveSummarizer()
summary = ext_summarizer.generate_summary(text, ratio=0.3)

# Abstractive summarization
abs_summarizer = AbstractiveSummarizer()
summary = abs_summarizer.generate_summary(text, max_length=150)

# Evaluate summary quality
from src import evaluate_summary
metrics = evaluate_summary(summary, original_text)
print(metrics)
```

### 📋 Recommended Input Guidelines

**Input Lengths:**
- **Minimum:** 1 sentence (will work but may not summarize)
- **Optimal:** 5+ sentences for best summarization
- **Maximum:** No hard limit (chunking handles long documents)

**Which Method to Choose:**
- **Extractive:** Best for factual accuracy, preserves original wording
- **Abstractive:** Best for brevity, generates new phrasing
- **Both:** Compare and choose the better summary

## 📊 How It Works

### Extractive Summarization (BERT-based)

1. **Preprocessing** - Clean and tokenize text into sentences
2. **Encoding** - Generate BERT embeddings for each sentence
3. **Scoring** - Apply ranking algorithms (TF-IDF, TextRank, LexRank)
4. **Selection** - Extract top-ranked sentences
5. **Ordering** - Maintain original sentence order

**Scoring Methods:**
- **TF-IDF**: Term frequency-inverse document frequency
- **TextRank**: Graph-based ranking using sentence similarity
- **LexRank**: Similar to TextRank with threshold-based graph
- **Combined**: Weighted combination of multiple methods

### Abstractive Summarization (T5-based)

1. **Preprocessing** - Clean and prepare input text
2. **Encoding** - Tokenize with T5 tokenizer (add "summarize:" prefix)
3. **Generation** - Use T5 model with beam search
4. **Decoding** - Convert tokens back to text
5. **Post-processing** - Clean and format output

**Generation Parameters:**
- Beam search for better quality
- Length control (min/max tokens)
- No-repeat n-grams to avoid repetition
- Temperature and penalty tuning

## 🏗️ Project Structure

```
AI-Based-Text-Summarization/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   ├── preprocessing.py         # Text preprocessing
│   ├── model_loader.py          # Model management
│   ├── sentence_ranker.py       # Sentence scoring algorithms
│   ├── extractive_summarizer.py # BERT extractive summarization
│   ├── abstractive_summarizer.py# T5 abstractive summarization
│   └── evaluation.py            # Evaluation metrics
├── tests/
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_extractive.py       # Extractive tests
│   └── test_abstractive.py      # Abstractive tests
├── data/
│   ├── samples/                 # Sample documents
│   └── outputs/                 # Generated summaries
├── main.py                      # CLI application
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Models**: Choose BERT/DistilBERT, T5-small/base/large
- **Summarization**: Adjust ratios, lengths, beam sizes
- **Processing**: Batch sizes, token limits, chunking
- **Device**: GPU/CPU selection

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L) - Overlap with reference
- **BLEU Score** - N-gram precision
- **Semantic Similarity** - TF-IDF cosine similarity
- **Readability** - Sentence/word length analysis
- **Compression Ratio** - Size reduction percentage

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_extractive.py -v
```

## 📦 Requirements

**Core Dependencies:**
- Python 3.8+
- transformers (HuggingFace)
- PyTorch
- NLTK
- scikit-learn
- rouge-score

**Optional:**
- Streamlit (web interface)
- Flask (API server)
- PyPDF2 (PDF support)
- python-docx (Word support)

See `requirements.txt` for complete list.

## 💡 Examples

### Sample Output

**Original Text (150 words):** "Artificial Intelligence continues to transform industries..."

**Extractive Summary (45 words):** "Artificial Intelligence continues to transform industries worldwide. Recent breakthroughs in deep learning have led to remarkable achievements in natural language processing. Education and workforce development are critical to preparing society for the AI revolution."

**Abstractive Summary (38 words):** "AI is transforming industries through machine learning and deep learning. While raising ethical concerns about privacy and bias, it promises benefits in climate change, drug discovery, and space exploration."

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace** for Transformers library
- **Google** for BERT and T5 models
- **NLTK** for NLP utilities
- **Streamlit** for web framework

## 📧 Contact

Project Link: [https://github.com/Ravi-667/AI-Based-Text-Summarization](https://github.com/Ravi-667/AI-Based-Text-Summarization)

## 🗺️ Roadmap

- [ ] Multi-document summarization
- [ ] Fine-tuning on domain-specific datasets
- [ ] RESTful API deployment
- [ ] Support for more languages
- [ ] Real-time summarization
- [ ] Summary quality scoring without reference
- [ ] Hierarchical summarization for very long documents

---

**Built with ❤️ using Python, BERT, and T5**