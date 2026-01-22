"""
Hugging Face Spaces Streamlit App
Simplified interface for text summarization
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import get_text_stats

# Page config
st.set_page_config(
    page_title="AI Text Summarization",
    page_icon="📝",
    layout="wide"
)

# Initialize session state for models
if 'extractive_model' not in st.session_state:
    st.session_state.extractive_model = None
if 'abstractive_model' not in st.session_state:
    st.session_state.abstractive_model = None


@st.cache_resource
def load_extractive_model():
    """Load extractive summarizer (cached)"""
    return ExtractiveSummarizer(device='cpu')


@st.cache_resource
def load_abstractive_model():
    """Load abstractive summarizer (cached)"""
    return AbstractiveSummarizer(device='cpu')


# Title
st.title("📝 AI-Based Text Summarization")
st.markdown("Generate summaries using state-of-the-art NLP models")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    method = st.selectbox(
        "Summarization Method",
        ["Extractive (BERT)", "Abstractive (T5)", "Both"],
        help="Choose the summarization approach"
    )
    
    st.divider()
    
    if method in ["Extractive (BERT)", "Both"]:
        st.subheader("Extractive Options")
        ext_method = st.selectbox(
            "Scoring Method",
            ["combined", "tfidf", "textrank", "lexrank"],
            help="Algorithm for sentence selection"
        )
        ratio = st.slider("Summary Ratio", 0.1, 0.8, 0.3, 0.05,
                         help="Fraction of original text to keep")
    
    if method in ["Abstractive (T5)", "Both"]:
        st.subheader("Abstractive Options")
        max_length = st.slider("Max Length", 50, 500, 150, 10,
                               help="Maximum summary length in tokens")
        min_length = st.slider("Min Length", 10, 200, 50, 10,
                               help="Minimum summary length in tokens")
    
    st.divider()
    st.markdown("### 📊 About")
    st.info("""
    **Models:**
    - Extractive: BERT embeddings
    - Abstractive: T5-base
    
    **GPU:** Auto-detected
    """)

# Main content
tab1, tab2 = st.tabs(["📄 Single Document", "📁 Batch Processing"])

with tab1:
    # Text input
    text_input = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Paste your text here... (Minimum 50 words recommended)",
        help="Enter the text you want to summarize"
    )
    
    # Sample text button
    if st.button("Load Sample Text"):
        text_input = """
        Artificial intelligence (AI) is transforming industries worldwide. Machine learning, 
        a subset of AI, enables computers to learn from data without explicit programming. 
        Deep learning uses neural networks with multiple layers to solve complex problems. 
        Natural language processing (NLP) allows machines to understand and generate human language. 
        These technologies are being applied in healthcare for disease diagnosis, in finance for 
        fraud detection, in autonomous vehicles for navigation, and in many other domains. 
        The rapid advancement of AI is creating new opportunities while also raising important 
        ethical questions about privacy, bias, and the future of work.
        """
        st.rerun()
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not text_input or len(text_input.strip()) < 50:
            st.error("⚠️ Please enter at least 50 characters of text")
        else:
            with st.spinner("Generating summary... This may take a moment for the first request."):
                # Get original stats
                original_stats = get_text_stats(text_input)
                
                # Display original stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Words", original_stats['word_count'])
                with col2:
                    st.metric("Sentences", original_stats['sentence_count'])
                with col3:
                    st.metric("Avg Word Length", f"{original_stats['avg_word_length']:.1f}")
                
                st.divider()
                
                # Generate summaries
                if method == "Extractive (BERT)":
                    model = load_extractive_model()
                    summary = model.generate_summary(
                        text_input,
                        ratio=ratio,
                        method=ext_method
                    )
                    
                    st.subheader("📊 Extractive Summary")
                    st.success(summary)
                    
                    # Summary stats
                    summary_stats = get_text_stats(summary)
                    compression = (1 - summary_stats['word_count'] / original_stats['word_count']) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Summary Words", summary_stats['word_count'])
                    with col2:
                        st.metric("Compression", f"{compression:.1f}%")
                    with col3:
                        st.metric("Method", ext_method.upper())
                
                elif method == "Abstractive (T5)":
                    model = load_abstractive_model()
                    summary = model.generate_summary(
                        text_input,
                        max_length=max_length,
                        min_length=min_length
                    )
                    
                    st.subheader("✨ Abstractive Summary")
                    st.success(summary)
                    
                    # Summary stats
                    summary_stats = get_text_stats(summary)
                    compression = (1 - summary_stats['word_count'] / original_stats['word_count']) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Summary Words", summary_stats['word_count'])
                    with col2:
                        st.metric("Compression", f"{compression:.1f}%")
                    with col3:
                        st.metric("Model", "T5-Base")
                
                else:  # Both
                    ext_model = load_extractive_model()
                    abs_model = load_abstractive_model()
                    
                    ext_summary = ext_model.generate_summary(
                        text_input,
                        ratio=ratio,
                        method=ext_method
                    )
                    
                    abs_summary = abs_model.generate_summary(
                        text_input,
                        max_length=max_length,
                        min_length=min_length
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Extractive (BERT)")
                        st.info(ext_summary)
                        ext_stats = get_text_stats(ext_summary)
                        ext_comp = (1 - ext_stats['word_count'] / original_stats['word_count']) * 100
                        st.metric("Compression", f"{ext_comp:.1f}%")
                    
                    with col2:
                        st.subheader("✨ Abstractive (T5)")
                        st.success(abs_summary)
                        abs_stats = get_text_stats(abs_summary)
                        abs_comp = (1 - abs_stats['word_count'] / original_stats['word_count']) * 100
                        st.metric("Compression", f"{abs_comp:.1f}%")

with tab2:
    st.info("📁 Batch processing feature coming soon! For now, use the API endpoint.")
    st.code("""
# Batch processing via API
curl -X POST "https://YOUR-SPACE-URL.hf.space/api/v1/batch" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["text1", "text2"], "method": "extractive"}'
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit | Powered by 🤗 Hugging Face</p>
    <p><a href='/docs'>API Documentation</a> | <a href='https://github.com/Ravi-667/AI-Based-Text-Summarization'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
