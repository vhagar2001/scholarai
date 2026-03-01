

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Optional


class PDFParser:
    """Parse PDF papers and extract structured content"""
    
    def __init__(self):
        self.section_headers = [
            'abstract', 'introduction', 'related work', 'methodology',
            'method', 'approach', 'experiments', 'results', 'discussion',
            'conclusion', 'references', 'background'
        ]
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        
        sections = {}
        text_lower = text.lower()
        
        # Find abstract
        abstract_match = re.search(
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|1\.|keywords))',
            text_lower,
            re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()
        
        # Try to find other sections by headers
        for i, header in enumerate(self.section_headers):
            # Look for section header
            pattern = rf'\n\s*(?:\d+\.?\s*)?{header}\s*\n(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:{")|(?:".join(self.section_headers[i+1:])})|$)'
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            
            if match:
                sections[header] = match.group(1).strip()[:2000]  # Limit length
        
        # If no sections found, treat full text as content
        if not sections:
            sections['full_text'] = text[:5000]  # First 5000 chars
        
        return sections
    
    def extract_metadata(self, text: str) -> Dict[str, any]:
        """
        Extract metadata from paper text
        
        
        """
        metadata = {}
        
        # Extract title 
        lines = text.split('\n')
        potential_title = [line.strip() for line in lines[:10] if len(line.strip()) > 20]
        if potential_title:
            metadata['extracted_title'] = potential_title[0]
        
        # Count pages
        metadata['text_length'] = len(text)
        metadata['estimated_pages'] = len(text) // 3000  # Rough estimate
        
        # Look for arxiv ID
        arxiv_match = re.search(r'arXiv:(\d+\.\d+)', text)
        if arxiv_match:
            metadata['arxiv_id'] = arxiv_match.group(1)
        
        return metadata
    
    def parse_paper(self, pdf_path: str) -> Dict:
        """
        Complete parsing pipeline for a paper
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with all extracted information
        """
        print(f"Parsing: {Path(pdf_path).name}")
        
        # Extract and clean text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            return {'error': 'Failed to extract text'}
        
        clean_text = self.clean_text(raw_text)
        
        # Extract sections and metadata
        sections = self.extract_sections(clean_text)
        metadata = self.extract_metadata(clean_text)
        
        result = {
            'filepath': pdf_path,
            'full_text': clean_text,
            'sections': sections,
            'metadata': metadata,
            'success': True
        }
        
        print(f"✓ Extracted {len(sections)} sections, {len(clean_text)} characters")
        return result
    
    def parse_multiple(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Parse multiple PDFs
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of parsing results
        """
        results = []
        
        for i, path in enumerate(pdf_paths, 1):
            print(f"\n[{i}/{len(pdf_paths)}]")
            result = self.parse_paper(path)
            results.append(result)
        
        print(f"\n✓ Parsed {len(results)} papers")
        return results


# Example usage
if __name__ == "__main__":
    parser = PDFParser()
    
    # Parse a single paper
    import os
    papers_dir = "data/papers"
    
    if os.path.exists(papers_dir):
        pdf_files = [
            os.path.join(papers_dir, f) 
            for f in os.listdir(papers_dir) 
            if f.endswith('.pdf')
        ][:3]  # Just first 3
        
        if pdf_files:
            results = parser.parse_multiple(pdf_files)
            
            # Show first result
            if results:
                print("\n" + "="*60)
                print("SAMPLE PARSED PAPER:")
                print("="*60)
                r = results[0]
                print(f"\nFile: {r['filepath']}")
                print(f"Text length: {r['metadata'].get('text_length', 0)} chars")
                print(f"Sections found: {list(r['sections'].keys())}")
                if 'abstract' in r['sections']:
                    print(f"\nAbstract preview:")
                    print(r['sections']['abstract'][:300] + "...")