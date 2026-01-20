"""
Configuration file for AI-Based Text Summarization System
"""

# Model Configurations
BERT_MODEL = "bert-base-uncased"
BERT_MODEL_DISTILLED = "distilbert-base-uncased"  # Lighter alternative
T5_MODEL_SMALL = "t5-small"
T5_MODEL_BASE = "t5-base"
T5_MODEL_LARGE = "t5-large"

# Default models to use
DEFAULT_EXTRACTIVE_MODEL = BERT_MODEL_DISTILLED
DEFAULT_ABSTRACTIVE_MODEL = T5_MODEL_SMALL

# Summarization Parameters
EXTRACTIVE_RATIO = 0.3  # Extract 30% of sentences
EXTRACTIVE_MIN_SENTENCES = 3
EXTRACTIVE_MAX_SENTENCES = 10

ABSTRACTIVE_MAX_LENGTH = 150  # Maximum summary length in tokens
ABSTRACTIVE_MIN_LENGTH = 50   # Minimum summary length in tokens
ABSTRACTIVE_LENGTH_PENALTY = 2.0
ABSTRACTIVE_NUM_BEAMS = 4     # Beam search width
ABSTRACTIVE_EARLY_STOPPING = True

# Processing Configuration
BATCH_SIZE = 8
MAX_INPUT_LENGTH = 512  # Maximum tokens for BERT/T5
CHUNK_OVERLAP = 50      # Token overlap when chunking long documents

# Text Preprocessing
REMOVE_STOPWORDS = False  # For extractive summarization scoring
MIN_SENTENCE_LENGTH = 5   # Minimum words in a sentence
MAX_SENTENCE_LENGTH = 100 # Maximum words in a sentence

# Device Configuration
DEVICE = "auto"  # Options: "auto", "cuda", "cpu"
USE_GPU = True   # Set to False to force CPU usage

# Logging
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/summarization.log"

# Paths
MODEL_CACHE_DIR = "models/cache"
OUTPUT_DIR = "data/outputs"
SAMPLE_DATA_DIR = "data/samples"

# Evaluation
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
CALCULATE_BLEU = True

# Web Interface
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
ENABLE_CORS = True
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
