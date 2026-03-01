"""
Database Models
SQLAlchemy models for storing paper metadata and user annotations
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, 
    ForeignKey, Table, JSON, Float
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


# Many-to-many relationship for paper tags
paper_tags = Table(
    'paper_tags',
    Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)


class Paper(Base):
    """Store paper metadata and content"""
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    title = Column(String(500), nullable=False)
    authors = Column(JSON)  # List of author names
    abstract = Column(Text)
    published_date = Column(DateTime)
    updated_date = Column(DateTime)
    categories = Column(JSON)  # ArXiv categories
    primary_category = Column(String(50))
    
    # File information
    pdf_path = Column(String(500))
    pdf_url = Column(String(500))
    
    # Extracted content
    full_text = Column(Text)
    sections = Column(JSON)  # Dict of section_name: content
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Reading status
    read_status = Column(String(20), default='unread')  # unread, reading, read
    
    # Relationships
    tags = relationship('Tag', secondary=paper_tags, back_populates='papers')
    notes = relationship('Note', back_populates='paper', cascade='all, delete-orphan')
    summaries = relationship('Summary', back_populates='paper', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Paper(arxiv_id='{self.arxiv_id}', title='{self.title[:50]}...')>"


class Tag(Base):
    """Tags for organizing papers"""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    color = Column(String(7))  # Hex color code
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    papers = relationship('Paper', secondary=paper_tags, back_populates='tags')
    
    def __repr__(self):
        return f"<Tag(name='{self.name}')>"


class Note(Base):
    """User notes on papers"""
    __tablename__ = 'notes'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'), nullable=False)
    content = Column(Text, nullable=False)
    note_type = Column(String(50))  # general, highlight, question, critique
    page_number = Column(Integer)  # If note refers to specific page
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    paper = relationship('Paper', back_populates='notes')
    
    def __repr__(self):
        return f"<Note(paper_id={self.paper_id}, type='{self.note_type}')>"


class Summary(Base):
    """AI-generated summaries of papers"""
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'), nullable=False)
    summary_type = Column(String(50))  # abstract, full, methodology, results
    content = Column(Text, nullable=False)
    model_used = Column(String(100))  # Which AI model generated this
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    paper = relationship('Paper', back_populates='summaries')
    
    def __repr__(self):
        return f"<Summary(paper_id={self.paper_id}, type='{self.summary_type}')>"


class Collection(Base):
    """Collections/projects for grouping papers"""
    __tablename__ = 'collections'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Collection(name='{self.name}')>"


class Citation(Base):
    """Track citations between papers"""
    __tablename__ = 'citations'
    
    id = Column(Integer, primary_key=True)
    citing_paper_id = Column(Integer, ForeignKey('papers.id'))
    cited_paper_id = Column(Integer, ForeignKey('papers.id'))
    context = Column(Text)  # Context where citation appears
    
    def __repr__(self):
        return f"<Citation(citing={self.citing_paper_id}, cited={self.cited_paper_id})>"


class SearchHistory(Base):
    """Track user search queries"""
    __tablename__ = 'search_history'
    
    id = Column(Integer, primary_key=True)
    query = Column(String(500), nullable=False)
    results_count = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchHistory(query='{self.query}')>"