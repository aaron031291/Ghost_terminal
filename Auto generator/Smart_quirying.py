from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticKnowledgeBase:
    """Knowledge base with semantic search capabilities."""
    
    def __init__(self):
        """Initialize semantic knowledge base."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_items = []
        self.embeddings = None
        
    def add_item(self, item: Dict[str, Any]):
        """Add knowledge item to the base."""
        self.knowledge_items.append(item)
        # Invalidate embeddings cache
        self.embeddings = None
        
    def _get_embeddings(self):
        """Get or compute embeddings for all items."""
        if self.embeddings is None:
            # Convert knowledge items to text for embedding
            texts = [json.dumps(item) for item in self.knowledge_items]
            self.embeddings = self.model.encode(texts)
        return self.embeddings
        
    def query(self, query: str, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Query knowledge base with semantic search.
        
        Args:
            query: Query string
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of matching knowledge items with scores
        """
        if not self.knowledge_items:
            return []
            
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Get all embeddings
        embeddings = self._get_embeddings()
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get items above threshold
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append({
                    "item": self.knowledge_items[i],
                    "relevance": float(similarity),
                })
                
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
