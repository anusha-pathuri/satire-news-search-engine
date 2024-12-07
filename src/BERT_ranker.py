from sentence_transformers import SentenceTransformer, util
from ranker import Ranker
from typing import Dict

class HybridRanker(Ranker):
    def __init__(self, base_ranker: Ranker, raw_text_dict: Dict[int, str],
                 bi_encoder_model_name: str, alpha: float = 0.5, rerank_top_k: int = 10) -> None:
        """
        Initializes a HybridRanker that combines base ranking with BERT reranking.

        Args:
            base_ranker: Base ranker (e.g., BM25) that implements Ranker interface
            raw_text_dict: Dictionary mapping document IDs to their raw text
            bi_encoder_model_name: Name of the BERT model to use for reranking
            alpha: Weight for combining scores (alpha * base_score + (1-alpha) * bert_score)
            rerank_top_k: Number of top documents to rerank
        """
        self.base_ranker = base_ranker
        self.raw_text_dict = raw_text_dict
        self.alpha = alpha
        self.rerank_top_k = rerank_top_k
        self.model = SentenceTransformer(bi_encoder_model_name)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Performs hybrid ranking by combining base ranker scores with BERT reranking.

        Args:
            query: The query string

        Returns:
            Sorted list of (doc_id, score) tuples, with highest scores first
        """
        # Handle edge cases
        if not query or not isinstance(query, str):
            return []

        try:
            # Get initial base rankings
            base_results = self.base_ranker.query(query)
            if not base_results:
                return []

            # Only rerank top k documents
            top_k_results = base_results[:self.rerank_top_k]

            # Encode query and top k documents
            query_embedding = self.model.encode(query)
            docs = [self.raw_text_dict[docid] for docid, _ in top_k_results]
            doc_embeddings = self.model.encode(docs)

            # Calculate hybrid scores for top k
            reranked_scores = []
            for i, (docid, base_score) in enumerate(top_k_results):
                bert_score = float(util.cos_sim(query_embedding, doc_embeddings[i]))
                final_score = self.alpha * base_score + (1 - self.alpha) * bert_score
                reranked_scores.append((docid, final_score))

            # Sort by combined score and append remaining documents
            reranked_scores.sort(key=lambda x: x[1], reverse=True)
            return reranked_scores + base_results[self.rerank_top_k:]

        except Exception:
            return []
