"""
ArXiv Paper Scraper
Searches and downloads papers from ArXiv API
"""

import arxiv
import os
from typing import List, Dict, Optional
from pathlib import Path
import time


class ArXivScraper:
    """Handle searching and downloading papers from ArXiv"""
    
    def __init__(self, download_dir: str = "data/papers"):
        """
        Initialize the scraper
        
        Args:
            download_dir: Directory to save downloaded PDFs
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def search_papers(
        self, 
        query: str, 
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """
        Search for papers on ArXiv
        
        Args:
            query: Search query (e.g., "Graph Neural Networks")
            max_results: Maximum number of results to return
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            
        Returns:
            List of paper metadata dictionaries
        """
        print(f"Searching ArXiv for: '{query}'...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        papers = []
        for result in search.results():
            paper_data = {
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.isoformat(),
                'updated': result.updated.isoformat(),
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category,
            }
            papers.append(paper_data)
            
        print(f"Found {len(papers)} papers")
        return papers
    
    def download_paper(self, paper: Dict, filename: Optional[str] = None) -> str:
        """
        Download a single paper PDF
        
        Args:
            paper: Paper metadata dictionary (must contain 'arxiv_id' and 'pdf_url')
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to downloaded PDF
        """
        arxiv_id = paper['arxiv_id']
        
        if filename is None:
            # Clean title for filename
            clean_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_'))
            clean_title = clean_title[:50]  # Limit length
            filename = f"{arxiv_id}_{clean_title}"
        
        filepath = self.download_dir / f"{filename}.pdf"
        
        # Skip if already downloaded
        if filepath.exists():
            print(f"Paper already exists: {filepath}")
            return str(filepath)
        
        print(f"Downloading: {paper['title'][:60]}...")
        
        try:
            # Use arxiv library to download
            paper_obj = next(arxiv.Search(id_list=[arxiv_id]).results())
            paper_obj.download_pdf(dirpath=str(self.download_dir), filename=f"{filename}.pdf")
            print(f"Downloaded to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"Error downloading {arxiv_id}: {e}")
            return None
    
    def download_papers(self, papers: List[Dict], delay: float = 3.0) -> List[str]:
        """
        Download multiple papers with rate limiting
        
        Args:
            papers: List of paper metadata dictionaries
            delay: Seconds to wait between downloads (be nice to ArXiv!)
            
        Returns:
            List of file paths to downloaded PDFs
        """
        filepaths = []
        
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}]")
            filepath = self.download_paper(paper)
            if filepath:
                filepaths.append(filepath)
            
            # Rate limiting
            if i < len(papers):
                time.sleep(delay)
        
        print(f"\n✓ Downloaded {len(filepaths)} papers successfully")
        return filepaths
    
    def search_and_download(
        self, 
        query: str, 
        max_results: int = 10
    ) -> tuple[List[Dict], List[str]]:
        """
        Convenience method: search and download papers in one go
        
        Args:
            query: Search query
            max_results: Maximum number of papers to download
            
        Returns:
            Tuple of (paper_metadata_list, filepath_list)
        """
        papers = self.search_papers(query, max_results)
        filepaths = self.download_papers(papers)
        return papers, filepaths


# Example usage
if __name__ == "__main__":
    scraper = ArXivScraper()
    
    # Search for papers
    papers, paths = scraper.search_and_download(
        query="Graph Neural Networks",
        max_results=5
    )
    
    # Print results
    print("\n" + "="*60)
    print("DOWNLOADED PAPERS:")
    print("="*60)
    for paper, path in zip(papers, paths):
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"File: {path}")