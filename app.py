"""
Web Interface for AI-Based Text Summarization using Streamlit
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.evaluation import SummaryEvaluator
from src.utils import read_document, get_text_stats, create_directories


# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'extractive_summarizer' not in st.session_state:
    create_directories()
    st.session_state.extractive_summarizer = None
    st.session_state.abstractive_summarizer = None
    st.session_state.evaluator = SummaryEvaluator()


def load_models(device='auto'):
    """Load summarization models."""
    with st.spinner("Loading models... This may take a minute."):
        if st.session_state.extractive_summarizer is None:
            st.session_state.extractive_summarizer = ExtractiveSummarizer(device=device)
        if st.session_state.abstractive_summarizer is None:
            st.session_state.abstractive_summarizer = AbstractiveSummarizer(device=device)


def main():
    """Main web application."""
    
    # Title and description
    st.title("📝 AI-Based Text Summarization")
    st.markdown("""
    Generate intelligent summaries using state-of-the-art NLP models.
    Choose between **Extractive** (BERT) or **Abstractive** (T5) summarization.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Summarization Method",
        ["Extractive (BERT)", "Abstractive (T5)", "Both"],
        help="Extractive: Selects important sentences. Abstractive: Generates new text."
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        help="Choose computation device. Auto will use GPU if available."
    )
    
    # Input method
    input_method = st.sidebar.radio(
        "Input Method",
        ["Text Box", "File Upload"]
    )
    
    st.sidebar.markdown("---")
    
    # Extractive options
    if method in ["Extractive (BERT)", "Both"]:
        st.sidebar.subheader("Extractive Options")
        
        extractive_ratio = st.sidebar.slider(
            "Extraction Ratio",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="Percentage of sentences to extract"
        )
        
        scoring_method = st.sidebar.selectbox(
            "Scoring Method",
            ["combined", "tfidf", "textrank", "lexrank"]
        )
    
    # Abstractive options
    if method in ["Abstractive (T5)", "Both"]:
        st.sidebar.subheader("Abstractive Options")
        
        summary_length = st.sidebar.select_slider(
            "Summary Length",
            options=["Short", "Medium", "Long"],
            value="Medium"
        )
        
        length_configs = {
            "Short": {"max_length": 80, "min_length": 30},
            "Medium": {"max_length": 150, "min_length": 50},
            "Long": {"max_length": 250, "min_length": 100}
        }
    
    st.sidebar.markdown("---")
    
    # Show evaluation
    show_eval = st.sidebar.checkbox("Show Evaluation Metrics", value=True)
    
    # Main content area
    text_input = None
    
    if input_method == "Text Box":
        text_input = st.text_area(
            "Enter text to summarize:",
            height=300,
            placeholder="Paste your text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_upload" + Path(uploaded_file.name).suffix)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text_input = read_document(str(temp_path))
                temp_path.unlink()  # Delete temp file
                
                st.success(f"✓ Loaded {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Summarize button
    if st.button("🚀 Generate Summary", type="primary"):
        
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please provide input text")
            return
        
        # Load models
        load_models(device)
        
        # Get original stats
        original_stats = get_text_stats(text_input)
        
        # Display original text stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", original_stats['word_count'])
        with col2:
            st.metric("Sentences", original_stats['sentence_count'])
        with col3:
            st.metric("Reading Time", f"{original_stats['reading_time_minutes']} min")
        
        st.markdown("---")
        
        # Extractive Summarization
        if method in ["Extractive (BERT)", "Both"]:
            st.subheader("📊 Extractive Summary (BERT)")
            
            with st.spinner("Generating extractive summary..."):
                extractive_summary = st.session_state.extractive_summarizer.generate_summary(
                    text_input,
                    ratio=extractive_ratio,
                    method=scoring_method
                )
            
            st.markdown(f"**Summary:**")
            st.info(extractive_summary)
            
            # Stats
            ext_stats = get_text_stats(extractive_summary)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", ext_stats['word_count'])
            with col2:
                st.metric("Compression", f"{(1 - ext_stats['word_count']/original_stats['word_count'])*100:.1f}%")
            with col3:
                st.metric("Reading Time", f"{ext_stats['reading_time_minutes']} min")
            
            # Evaluation
            if show_eval:
                with st.spinner("Calculating evaluation metrics..."):
                    eval_results = st.session_state.evaluator.evaluate_summary(
                        extractive_summary,
                        text_input
                    )
                
                with st.expander("📈 Evaluation Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Semantic Similarity", f"{eval_results['semantic_similarity']:.3f}")
                    with col2:
                        st.metric("Readability Score", f"{eval_results['readability']['readability_score']:.3f}")
            
            st.markdown("---")
        
        # Abstractive Summarization
        if method in ["Abstractive (T5)", "Both"]:
            st.subheader("✨ Abstractive Summary (T5)")
            
            config = length_configs[summary_length]
            
            with st.spinner("Generating abstractive summary..."):
                abstractive_summary = st.session_state.abstractive_summarizer.generate_summary(
                    text_input,
                    max_length=config['max_length'],
                    min_length=config['min_length']
                )
            
            st.markdown(f"**Summary:**")
            st.success(abstractive_summary)
            
            # Stats
            abs_stats = get_text_stats(abstractive_summary)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", abs_stats['word_count'])
            with col2:
                st.metric("Compression", f"{(1 - abs_stats['word_count']/original_stats['word_count'])*100:.1f}%")
            with col3:
                st.metric("Reading Time", f"{abs_stats['reading_time_minutes']} min")
            
            # Evaluation
            if show_eval:
                with st.spinner("Calculating evaluation metrics..."):
                    eval_results = st.session_state.evaluator.evaluate_summary(
                        abstractive_summary,
                        text_input
                    )
                
                with st.expander("📈 Evaluation Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Semantic Similarity", f"{eval_results['semantic_similarity']:.3f}")
                    with col2:
                        st.metric("Readability Score", f"{eval_results['readability']['readability_score']:.3f}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application uses:
    - **BERT** for extractive summarization
    - **T5** for abstractive summarization
    - **ROUGE** metrics for evaluation
    
    Built with Streamlit and HuggingFace Transformers
    """)


if __name__ == "__main__":
    main()
