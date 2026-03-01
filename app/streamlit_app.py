"""
ScholarAI - Streamlit Web Interface
Main application interface for the research companion
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.arxiv_scraper import ArXivScraper
from src.data_collection.pdf_parser import PDFParser
from src.database.operations import DatabaseManager
from src.rag.qa_chain_free import ScholarRAGFree

# Page configuration
st.set_page_config(
    page_title="ScholarAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()

if 'scraper' not in st.session_state:
    st.session_state.scraper = ArXivScraper()

if 'parser' not in st.session_state:
    st.session_state.parser = PDFParser()

# RAG initialization
if 'rag' not in st.session_state:
    st.session_state.rag = None

if st.session_state.rag is None and 'rag_initialized' not in st.session_state:
    with st.spinner("Initializing AI system... (first time: 1-2 minutes)"):
        try:
            st.session_state.rag = ScholarRAGFree()
            st.session_state.rag_initialized = True
            st.success("AI system ready")
        except Exception as e:
            st.session_state.rag = None
            st.session_state.rag_initialized = False
            st.session_state.rag_error = str(e)
            st.error(f"Could not initialize AI system: {str(e)[:100]}")


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("# ScholarAI")
        st.markdown("Research Assistant")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Home", "Search Papers", "Library", "Ask Questions", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Statistics
        stats = st.session_state.db.get_statistics()
        st.markdown("### Statistics")
        st.metric("Total Papers", stats['total_papers'])
        st.metric("Unread", stats['unread_papers'])
        st.metric("Notes", stats['total_notes'])
    
    # Main content
    if page == "Home":
        show_home()
    elif page == "Search Papers":
        show_search()
    elif page == "Library":
        show_library()
    elif page == "Ask Questions":
        show_qa()
    elif page == "Settings":
        show_settings()


def show_home():
    """Home page"""
    st.markdown('<p class="main-header">Welcome to ScholarAI</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Your Research Assistant
    
    ScholarAI helps you:
    - **Discover** relevant papers from ArXiv
    - **Organize** your research library
    - **Understand** papers with AI summaries
    - **Ask questions** and get instant answers
    - **Connect** ideas across multiple papers
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Quick Actions")
        if st.button("Search Papers", use_container_width=True):
            st.session_state.current_page = "Search Papers"
            st.rerun()
        if st.button("View Library", use_container_width=True):
            st.session_state.current_page = "Library"
            st.rerun()
    
    with col2:
        st.markdown("### Your Progress")
        stats = st.session_state.db.get_statistics()
        st.info(f"{stats['total_papers']} papers in library")
        st.info(f"{stats['total_notes']} notes created")
        st.info(f"{stats['total_summaries']} summaries generated")
    
    with col3:
        st.markdown("### Recent Activity")
        recent_papers = st.session_state.db.get_all_papers(limit=3)
        if recent_papers:
            for paper in recent_papers:
                st.markdown(f"- {paper.title[:50]}...")
        else:
            st.markdown("*No papers yet. Start by searching.*")


def show_search():
    """Search and add papers"""
    st.markdown('<p class="main-header">Search Papers</p>', unsafe_allow_html=True)
    
    # Search form
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search ArXiv", placeholder="e.g., Graph Neural Networks, Transformers")
    with col2:
        max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching ArXiv..."):
                try:
                    papers = st.session_state.scraper.search_papers(query, max_results)
                    st.session_state.search_results = papers
                    st.success(f"Found {len(papers)} papers")
                except Exception as e:
                    st.error(f"Search failed: {e}")
        else:
            st.warning("Please enter a search query")
    
    # Display results
    if hasattr(st.session_state, 'search_results'):
        st.markdown("---")
        st.markdown("### Results")
        
        for i, paper in enumerate(st.session_state.search_results):
            with st.expander(f"{i+1}. {paper['title']}", expanded=(i==0)):
                st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                st.markdown(f"**Published:** {paper['published'][:10]}")
                st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
                st.markdown(f"**Abstract:** {paper['abstract'][:300]}...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Add to Library", key=f"add_{i}"):
                        with st.spinner("Processing..."):
                            try:
                                filepath = st.session_state.scraper.download_paper(paper)
                                
                                if filepath:
                                    parsed = st.session_state.parser.parse_paper(filepath)
                                    
                                    paper['pdf_path'] = filepath
                                    paper['full_text'] = parsed.get('full_text', '')
                                    paper['sections'] = parsed.get('sections', {})
                                    
                                    db_paper = st.session_state.db.add_paper(paper)
                                    
                                    if st.session_state.rag and paper['full_text']:
                                        with st.spinner("Indexing for AI..."):
                                            try:
                                                chunks_added = st.session_state.rag.add_paper(paper)
                                                st.success(f"Added to library ({chunks_added} chunks indexed)")
                                            except Exception as e:
                                                st.warning(f"Added to library but AI indexing failed: {e}")
                                    else:
                                        st.success("Added to library")
                                else:
                                    st.error("Failed to download PDF")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                with col2:
                    st.markdown(f"[View on ArXiv]({paper['pdf_url']})")


def show_library():
    """Library view"""
    st.markdown('<p class="main-header">Library</p>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["All", "Unread", "Reading", "Read"])
    with col2:
        search_library = st.text_input("Search", placeholder="Title or keyword")
    with col3:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Title"])
    
    # Get papers
    if search_library:
        papers = st.session_state.db.search_papers(search_library)
    else:
        papers = st.session_state.db.get_all_papers(limit=100)
    
    if status_filter != "All":
        papers = [p for p in papers if p.read_status == status_filter.lower()]
    
    st.markdown(f"### {len(papers)} Papers")
    
    # Display papers
    for paper in papers:
        with st.expander(f"{paper.title}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Authors:** {', '.join(paper.authors[:5])}")
                st.markdown(f"**Published:** {paper.published_date}")
                
                if paper.abstract:
                    st.markdown("**Abstract:**")
                    st.markdown(paper.abstract[:400] + "...")
                
                notes = st.session_state.db.get_paper_notes(paper.id)
                if notes:
                    st.markdown("**Notes:**")
                    for note in notes:
                        st.info(f"{note.content}")
            
            with col2:
                # Status
                new_status = st.selectbox(
                    "Status",
                    ["unread", "reading", "read"],
                    index=["unread", "reading", "read"].index(paper.read_status),
                    key=f"status_{paper.id}"
                )
                if new_status != paper.read_status:
                    st.session_state.db.update_paper(paper.id, {'read_status': new_status})
                    st.rerun()
                
                # Actions
                is_indexed = False
                if st.session_state.rag:
                    try:
                        results = st.session_state.rag.collection.get(where={"arxiv_id": paper.arxiv_id})
                        is_indexed = len(results['ids']) > 0
                    except:
                        pass
                
                if not is_indexed and st.session_state.rag:
                    if st.button("Index for AI", key=f"index_{paper.id}"):
                        if paper.full_text:
                            with st.spinner("Indexing..."):
                                try:
                                    paper_data = {
                                        'arxiv_id': paper.arxiv_id,
                                        'title': paper.title,
                                        'authors': paper.authors,
                                        'full_text': paper.full_text
                                    }
                                    chunks = st.session_state.rag.add_paper(paper_data)
                                    st.success(f"Indexed ({chunks} chunks)")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                elif is_indexed:
                    st.success("AI-ready")
                
                if st.button("Add Note", key=f"note_{paper.id}"):
                    st.session_state[f'show_note_input_{paper.id}'] = True
                
                if st.session_state.get(f'show_note_input_{paper.id}', False):
                    note_text = st.text_area("Note:", key=f"note_text_{paper.id}")
                    if st.button("Save", key=f"save_note_{paper.id}"):
                        if note_text:
                            st.session_state.db.add_note(paper.id, note_text)
                            st.success("Saved")
                            st.session_state[f'show_note_input_{paper.id}'] = False
                            st.rerun()
                
                if st.button("Summarize", key=f"sum_{paper.id}"):
                    if st.session_state.rag:
                        try:
                            results = st.session_state.rag.collection.get(where={"arxiv_id": paper.arxiv_id})
                            if not results['ids']:
                                st.warning("Not indexed. Click 'Index for AI' first")
                            else:
                                with st.spinner("Generating..."):
                                    summary = st.session_state.rag.summarize_paper(paper.arxiv_id)
                                    st.session_state.db.add_summary(paper.id, summary)
                                    st.markdown("**Summary:**")
                                    st.info(summary)
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                if st.button("Delete", key=f"del_{paper.id}"):
                    st.session_state[f'confirm_delete_{paper.id}'] = True
                
                if st.session_state.get(f'confirm_delete_{paper.id}', False):
                    st.warning("Are you sure?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", key=f"yes_{paper.id}"):
                            session = st.session_state.db.get_session()
                            session.delete(paper)
                            session.commit()
                            session.close()
                            st.success("Deleted")
                            st.rerun()
                    with col_no:
                        if st.button("No", key=f"no_{paper.id}"):
                            st.session_state[f'confirm_delete_{paper.id}'] = False
                            st.rerun()


def show_qa():
    """Question answering"""
    st.markdown('<p class="main-header">Ask Questions</p>', unsafe_allow_html=True)
    
    if st.session_state.rag is None:
        st.error("AI system not available")
        
        if hasattr(st.session_state, 'rag_error'):
            with st.expander("Show error details"):
                st.code(st.session_state.rag_error)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retry Initialization", type="primary", use_container_width=True):
                with st.spinner("Initializing..."):
                    try:
                        st.session_state.rag = ScholarRAGFree()
                        st.session_state.rag_initialized = True
                        st.success("System ready")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")
        
        return
    
    papers = st.session_state.db.get_all_papers()
    
    if not papers:
        st.warning("No papers in library")
        if st.button("Search Papers"):
            st.rerun()
        return
    
    # Select paper
    paper_options = ["All papers"] + [f"{p.title[:50]}... ({p.arxiv_id})" for p in papers]
    selected_paper = st.selectbox("Ask about:", paper_options)
    
    # Question
    question = st.text_area(
        "Your question:", 
        placeholder="e.g., What methodology do they use?",
        height=100
    )
    
    if st.button("Get Answer", type="primary", use_container_width=True):
        if question:
            with st.spinner("Processing..."):
                try:
                    arxiv_id = None
                    if selected_paper != "All papers":
                        arxiv_id = selected_paper.split("(")[-1].strip(")")
                    
                    result = st.session_state.rag.answer_question(question, arxiv_id)
                    
                    st.markdown("---")
                    st.markdown("### Answer")
                    st.info("Extractive - Direct quotes from papers")
                    st.markdown(result['answer'])
                    
                    if result.get('sources'):
                        st.markdown("### Sources")
                        for source in result['sources']:
                            st.markdown(f"- {source}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence_score', 0)}/10")
                    with col2:
                        st.metric("Chunks Used", result.get('context_used', 0))
                    with col3:
                        st.metric("Method", "Extractive")
                    
                    if result.get('relevance_scores'):
                        with st.expander("Relevance scores"):
                            for i, score in enumerate(result['relevance_scores'], 1):
                                st.markdown(f"Chunk {i}: {score}/10")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question")
    
    # Quick questions
    st.markdown("---")
    st.markdown("### Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Main contributions?", use_container_width=True):
            st.session_state.quick_question = "What are the main contributions?"
    
    with col2:
        if st.button("Methodology?", use_container_width=True):
            st.session_state.quick_question = "What methodology was used?"
    
    with col3:
        if st.button("Key findings?", use_container_width=True):
            st.session_state.quick_question = "What are the key findings?"


def show_settings():
    """Settings"""
    st.markdown('<p class="main-header">Settings</p>', unsafe_allow_html=True)
    
    st.markdown("### Research-Grade Models")
    
    st.markdown("""
    **SPECTER2** (AllenAI)
    - Trained on 100M+ scientific papers
    - Used by Semantic Scholar
    
    **BGE Reranker** (BAAI)  
    - State-of-the-art reranking
    - Production-grade accuracy
    
    **Extractive Method**
    - No text generation
    - Direct quotes from papers
    - Zero hallucinations
    """)
    
    st.markdown("---")
    st.markdown("### System Information")
    
    if st.session_state.rag:
        stats = st.session_state.rag.get_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Chunks", stats['total_chunks'])
            st.metric("Papers", stats['unique_papers'])
        with col2:
            st.metric("Device", stats['device'].upper())
    
    st.markdown("---")
    st.markdown("### Maintenance")
    
    if st.button("Clear All Data"):
        st.warning("This will delete all papers and embeddings")


if __name__ == "__main__":
    main()