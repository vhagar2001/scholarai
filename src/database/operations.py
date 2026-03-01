

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict
from datetime import datetime
import json

from .models import Base, Paper, Tag, Note, Summary, Collection, Citation, SearchHistory


class DatabaseManager:
    """Manage database operations"""
    
    def __init__(self, database_url: str = "sqlite:///data/database/scholarai.db"):
        """
        Initialize database connection
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    # ==================== PAPER OPERATIONS ====================
    
    def add_paper(self, paper_data: Dict, session: Optional[Session] = None) -> Paper:
        """
        Add a new paper to database
        
        Args:
            paper_data: Dictionary with paper information
            session: Optional existing session
            
        Returns:
            Created Paper object
        """
        close_session = False
        if session is None:
            session = self.get_session()
            close_session = True
        
        try:
            # Check if paper already exists
            existing = session.query(Paper).filter_by(
                arxiv_id=paper_data.get('arxiv_id')
            ).first()
            
            if existing:
                print(f"Paper {paper_data.get('arxiv_id')} already exists")
                return existing
            
            # Parse dates if they're strings
            published = paper_data.get('published')
            if isinstance(published, str):
                published = datetime.fromisoformat(published.replace('Z', '+00:00'))
            
            updated = paper_data.get('updated')
            if isinstance(updated, str):
                updated = datetime.fromisoformat(updated.replace('Z', '+00:00'))
            
            # Create paper
            paper = Paper(
                arxiv_id=paper_data.get('arxiv_id'),
                title=paper_data.get('title'),
                authors=paper_data.get('authors', []),
                abstract=paper_data.get('abstract'),
                published_date=published,
                updated_date=updated,
                categories=paper_data.get('categories', []),
                primary_category=paper_data.get('primary_category'),
                pdf_path=paper_data.get('pdf_path'),
                pdf_url=paper_data.get('pdf_url'),
                full_text=paper_data.get('full_text'),
                sections=paper_data.get('sections', {})
            )
            
            session.add(paper)
            session.commit()
            session.refresh(paper)
            
            print(f"✓ Added paper: {paper.title[:60]}")
            return paper
            
        finally:
            if close_session:
                session.close()
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """Get paper by ArXiv ID"""
        session = self.get_session()
        try:
            return session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
        finally:
            session.close()
    
    def get_paper_by_id(self, paper_id: int) -> Optional[Paper]:
        """Get paper by database ID"""
        session = self.get_session()
        try:
            return session.query(Paper).filter_by(id=paper_id).first()
        finally:
            session.close()
    
    def get_all_papers(self, limit: int = 100) -> List[Paper]:
        """Get all papers"""
        session = self.get_session()
        try:
            return session.query(Paper).order_by(desc(Paper.created_at)).limit(limit).all()
        finally:
            session.close()
    
    def search_papers(self, query: str, limit: int = 20) -> List[Paper]:
        """
        Search papers by title or abstract
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching papers
        """
        session = self.get_session()
        try:
            return session.query(Paper).filter(
                (Paper.title.contains(query)) | 
                (Paper.abstract.contains(query))
            ).limit(limit).all()
        finally:
            session.close()
    
    def update_paper(self, paper_id: int, updates: Dict) -> Optional[Paper]:
        """Update paper fields"""
        session = self.get_session()
        try:
            paper = session.query(Paper).filter_by(id=paper_id).first()
            if paper:
                for key, value in updates.items():
                    if hasattr(paper, key):
                        setattr(paper, key, value)
                session.commit()
                session.refresh(paper)
            return paper
        finally:
            session.close()
    
    # ==================== TAG OPERATIONS ====================
    
    def add_tag(self, name: str, color: Optional[str] = None) -> Tag:
        """Add a new tag"""
        session = self.get_session()
        try:
            existing = session.query(Tag).filter_by(name=name).first()
            if existing:
                return existing
            
            tag = Tag(name=name, color=color)
            session.add(tag)
            session.commit()
            session.refresh(tag)
            return tag
        finally:
            session.close()
    
    def add_tag_to_paper(self, paper_id: int, tag_name: str) -> bool:
        """Add a tag to a paper"""
        session = self.get_session()
        try:
            paper = session.query(Paper).filter_by(id=paper_id).first()
            tag = session.query(Tag).filter_by(name=tag_name).first()
            
            if not tag:
                tag = Tag(name=tag_name)
                session.add(tag)
            
            if paper and tag not in paper.tags:
                paper.tags.append(tag)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def get_papers_by_tag(self, tag_name: str) -> List[Paper]:
        """Get all papers with a specific tag"""
        session = self.get_session()
        try:
            tag = session.query(Tag).filter_by(name=tag_name).first()
            return tag.papers if tag else []
        finally:
            session.close()
    
    # ==================== NOTE OPERATIONS ====================
    
    def add_note(self, paper_id: int, content: str, note_type: str = 'general') -> Note:
        """Add a note to a paper"""
        session = self.get_session()
        try:
            note = Note(
                paper_id=paper_id,
                content=content,
                note_type=note_type
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            return note
        finally:
            session.close()
    
    def get_paper_notes(self, paper_id: int) -> List[Note]:
        """Get all notes for a paper"""
        session = self.get_session()
        try:
            return session.query(Note).filter_by(paper_id=paper_id).all()
        finally:
            session.close()
    
    # ==================== SUMMARY OPERATIONS ====================
    
    def add_summary(
        self, 
        paper_id: int, 
        content: str, 
        summary_type: str = 'full',
        model_used: str = 'claude-3'
    ) -> Summary:
        """Add an AI-generated summary"""
        session = self.get_session()
        try:
            summary = Summary(
                paper_id=paper_id,
                content=content,
                summary_type=summary_type,
                model_used=model_used
            )
            session.add(summary)
            session.commit()
            session.refresh(summary)
            return summary
        finally:
            session.close()
    
    def get_paper_summary(self, paper_id: int, summary_type: str = 'full') -> Optional[Summary]:
        """Get a specific summary for a paper"""
        session = self.get_session()
        try:
            return session.query(Summary).filter_by(
                paper_id=paper_id,
                summary_type=summary_type
            ).first()
        finally:
            session.close()
    
    # ==================== UTILITY OPERATIONS ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        session = self.get_session()
        try:
            return {
                'total_papers': session.query(Paper).count(),
                'total_tags': session.query(Tag).count(),
                'total_notes': session.query(Note).count(),
                'total_summaries': session.query(Summary).count(),
                'unread_papers': session.query(Paper).filter_by(read_status='unread').count(),
            }
        finally:
            session.close()


# Example usage
if __name__ == "__main__":
    db = DatabaseManager()
    
    # Get statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print("="*40)
    for key, value in stats.items():
        print(f"{key}: {value}")