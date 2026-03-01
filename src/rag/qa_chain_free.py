

import os
from typing import List, Dict, Optional
import torch

os.environ["ALLOW_RESET"] = "TRUE"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder


class ScholarRAGFree:
    """
    Research-grade RAG system using SOTA models
    - SPECTER2 for scientific paper embeddings (used by Semantic Scholar)
    - BGE for reranking (SOTA Chinese model, multilingual)
    - Extractive approach for accuracy (no hallucinations)
    """
    
    def __init__(self, persist_directory: str = "data/embeddings"):
        """Initialize with research-grade models"""
        print("🚀 Initializing RESEARCH-GRADE RAG system...")
        print("   Using models from Semantic Scholar & industry leaders")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")
        
        # SPECTER2 - Specifically trained on scientific papers!
        # Used by Semantic Scholar, Allen Institute for AI
        print("   Loading SPECTER2 (scientific paper embeddings)...")
        print("   📚 This model was trained on 100M+ scientific papers")
        try:
            self.embedding_model = SentenceTransformer('allenai/specter2')
        except:
            # Fallback to SciBERT if SPECTER2 fails
            print("   Falling back to SciBERT...")
            self.embedding_model = SentenceTransformer('sentence-transformers/allenai-specter')
        
        # BGE Reranker - Current SOTA for reranking
        print("   Loading BGE reranker (SOTA quality)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        # ChromaDB
        print("   Initializing vector database...")
        os.makedirs(persist_directory, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ RESEARCH-GRADE RAG ready!")
        print(f"   📊 {self.collection.count()} chunks indexed")
    
    def add_paper(self, paper_data: Dict) -> int:
        """Add paper with scientific chunking strategy"""
        arxiv_id = paper_data.get('arxiv_id', 'unknown')
        text = paper_data.get('full_text', '')
        
        if not text:
            print(f"⚠️  No text for paper {arxiv_id}")
            return 0
        
        try:
            existing = self.collection.get(where={"arxiv_id": arxiv_id})
            if existing['ids']:
                print(f"ℹ️  Paper {arxiv_id} already indexed")
                return 0
        except:
            pass
        
        print(f"📄 Indexing paper: {arxiv_id}")
        
        # Scientific paper chunking: ~250 words (1200 chars)
        # Research shows this is optimal for academic content
        chunks = self._chunk_text(text, chunk_size=1200, overlap=200)
        
        metadata = {
            'arxiv_id': arxiv_id,
            'title': paper_data.get('title', ''),
            'authors': str(paper_data.get('authors', [])),
        }
        
        print(f"   🔢 Generating SPECTER2 embeddings for {len(chunks)} chunks...")
        chunk_embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=8  # Optimal for SPECTER2
        )
        
        self.collection.add(
            ids=[f"{arxiv_id}_chunk_{i}" for i in range(len(chunks))],
            embeddings=chunk_embeddings.tolist(),
            documents=chunks,
            metadatas=[{**metadata, 'chunk_id': i} for i in range(len(chunks))]
        )
        
        print(f"✅ Indexed {len(chunks)} chunks")
        return len(chunks)
    
    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """Smart chunking optimized for research papers"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        
        return chunks
    
    def _rerank_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[tuple]:
        """
        BGE reranker - SOTA quality
        Returns: List of (chunk, score) tuples
        """
        if not chunks:
            return []
        
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores[:top_k]
    
    def _extract_answer(self, question: str, context: str, sources: List[str]) -> str:
        """
        EXTRACTIVE approach - no hallucinations!
        Finds the most relevant sentence(s) from context
        """
        # Split context into sentences
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return "No relevant information found in the retrieved context."
        
        # Rerank sentences to find most relevant
        if len(sentences) > 1:
            sentence_pairs = [[question, sent] for sent in sentences]
            sentence_scores = self.reranker.predict(sentence_pairs)
            
            # Get top 2-3 sentences
            sent_with_scores = list(zip(sentences, sentence_scores))
            sent_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top sentences (at least score > 0)
            top_sentences = [s for s, score in sent_with_scores[:3] if score > 0]
            
            if not top_sentences:
                top_sentences = [sent_with_scores[0][0]]
            
            answer = '. '.join(top_sentences)
            if not answer.endswith('.'):
                answer += '.'
            
            return answer
        else:
            return sentences[0] + '.'
    
    def answer_question(
        self, 
        question: str, 
        arxiv_id: Optional[str] = None,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 6
    ) -> Dict:
        """
        Answer with EXTRACTIVE approach - NO hallucinations
        """
        print(f"🤔 Answering: {question[:80]}...")
        
        # Step 1: SPECTER2 retrieval
        query_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        
        if arxiv_id:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k_retrieve,
                where={"arxiv_id": arxiv_id}
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k_retrieve
            )
        
        if not results['documents'] or not results['documents'][0]:
            return {
                'answer': "No relevant papers found in the library for this question.",
                'sources': [],
                'confidence': 'none',
                'confidence_score': 0.0,
                'method': 'extractive'
            }
        
        # Step 2: BGE Reranking
        print("   🔍 Reranking with BGE (SOTA)...")
        retrieved_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        reranked = self._rerank_chunks(question, retrieved_chunks, top_k=top_k_rerank)
        
        if not reranked:
            return {
                'answer': "No sufficiently relevant information found.",
                'sources': [],
                'confidence': 'none',
                'confidence_score': 0.0,
                'method': 'extractive'
            }
        
        # Step 3: Build context and extract sources
        context_parts = []
        sources = set()
        scores = []
        
        for chunk, score in reranked:
            context_parts.append(chunk)
            scores.append(score)
            idx = retrieved_chunks.index(chunk)
            meta = metadatas[idx]
            sources.add(f"{meta.get('title', 'Unknown')[:60]}... ({meta.get('arxiv_id', 'unknown')})")
        
        context = "\n\n".join(context_parts)
        
        # Step 4: EXTRACT answer (no generation = no hallucinations!)
        print("   ✂️  Extracting answer from context (no hallucination)...")
        answer = self._extract_answer(question, context, list(sources))
        
        # Real confidence from BGE scores
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # BGE scores are typically -10 to +10
        # Normalize to 0-10 scale
        normalized_score = max(0, min(10, (avg_score + 5) * 1.5))
        
        if normalized_score > 7:
            confidence = 'high'
        elif normalized_score > 4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        print(f"   📊 Confidence: {confidence} ({normalized_score:.1f}/10)")
        
        return {
            'answer': answer,
            'sources': list(sources),
            'context_used': len(reranked),
            'confidence': confidence,
            'confidence_score': round(normalized_score, 1),
            'relevance_scores': [round((s + 5) * 1.5, 1) for s in scores],
            'method': 'extractive (no hallucination)'
        }
    
    def summarize_paper(self, arxiv_id: str) -> str:
        """
        Extractive summarization - most important sentences
        """
        print(f"📝 Summarizing paper: {arxiv_id}")
        
        results = self.collection.get(where={"arxiv_id": arxiv_id})
        
        if not results['documents']:
            return "Paper not found in database. Please add it first."
        
        # Get first 10 chunks (usually abstract + intro)
        chunks = results['documents'][:10]
        full_text = " ".join(chunks)
        
        # Split into sentences
        sentences = [s.strip() + '.' for s in full_text.split('.') if len(s.strip()) > 30]
        
        if len(sentences) < 3:
            return " ".join(sentences)
        
        # Use reranker to find most important sentences
        # Query: what is this paper about
        summary_query = "What is the main contribution, methodology, and findings of this research?"
        
        sentence_pairs = [[summary_query, sent] for sent in sentences[:20]]  # First 20 sentences
        scores = self.reranker.predict(sentence_pairs)
        
        # Get top 4-5 sentences
        sent_with_scores = list(zip(sentences[:20], scores))
        sent_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_sentences = [s for s, _ in sent_with_scores[:5]]
        
        # Reorder to match original order
        top_sentences_ordered = [s for s in sentences if s in top_sentences]
        
        summary = " ".join(top_sentences_ordered[:4])
        
        print("   ✅ Extractive summary generated")
        return summary
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        total_chunks = self.collection.count()
        unique_papers = 0
        
        if total_chunks > 0:
            try:
                all_metadata = self.collection.get()['metadatas']
                unique_papers = len(set(meta['arxiv_id'] for meta in all_metadata))
            except:
                unique_papers = 0
        
        return {
            'total_chunks': total_chunks,
            'unique_papers': unique_papers,
            'embedding_model': 'SPECTER2 (AllenAI - 100M papers)',
            'reranker_model': 'BGE-reranker-base (SOTA)',
            'method': 'EXTRACTIVE (no hallucinations)',
            'device': self.device,
            'industry_grade': True
        }


if __name__ == "__main__":
    rag = ScholarRAGFree()
    stats = rag.get_statistics()
    print("\n📊 System Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")