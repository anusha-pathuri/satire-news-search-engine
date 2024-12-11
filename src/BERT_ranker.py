"""TODO: Fix this, it's not working as expected.
Results were generated using the implementation of HybridRanker in 05_BERT_opt.ipynb.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict
import torch
from sentence_transformers import SentenceTransformer, util
from src.ranker import Ranker


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device)
        
    return torch.tensor(array, dtype=torch.float32).to(device)


class HybridRanker(Ranker):
    def __init__(self, base_ranker: Ranker, bi_encoder_model_name: str, alpha: float = 0.5, rerank_top_k: int = 10, 
                 raw_text_dict: Dict[int, str] = None, encoded_docs: np.ndarray = None, row_to_docid: list[int] = None) -> None:
        """
        Initializes a HybridRanker that combines base ranking with BERT reranking.

        Args:
            base_ranker: Base ranker (e.g., BM25) that implements Ranker interface
            raw_text_dict: Dictionary mapping document IDs to their raw text
            bi_encoder_model_name: Name of the BERT model to use for reranking
            alpha: Weight for combining scores (alpha * base_score + (1-alpha) * bert_score)
            rerank_top_k: Number of top documents to rerank
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")
        
        assert raw_text_dict is not None or (encoded_docs is not None and row_to_docid is not None), \
            "At least one of raw_text_dict or (encoded_docs, row_to_docid) must be provided."

        self.base_ranker = base_ranker
        self.alpha = alpha
        self.rerank_top_k = rerank_top_k
        
        # In order to use dot-product for computing similarity between embeddings,
        # we must normalize them so that each sentence embedding is of length 1
        self.model = SentenceTransformer(bi_encoder_model_name)
        self.model.to(device)
        
        self.raw_text_dict = raw_text_dict
        
        # Encode documents
        if encoded_docs is None:
            self.docids = list(raw_text_dict.keys())
            self.doc_embeddings = self.model.encode([raw_text_dict[docid] for docid in self.docids],
                                                    normalize_embeddings=True,
                                                    convert_to_tensor=True,
                                                    device=device)
            print(self.doc_embeddings.shape)    
        else:
            self.doc_embeddings = util.normalize_embeddings(array_to_tensor(encoded_docs))
            self.docids = row_to_docid
        
        self.docid_to_row = {docid: row for row, docid in enumerate(self.docids)}
        

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
            print(top_k_results)

            # Encode query
            query_embedding = self.biencoder_model.encode(query, 
                                                          normalize_embeddings=True,
                                                          convert_to_tensor=True,
                                                          device=device)            

            # Calculate hybrid scores for top k
            reranked_scores = []
            for i, (docid, base_score) in enumerate(top_k_results):
                row = self.docid_to_row[docid]
                bert_score = float(util.cos_sim(query_embedding, self.doc_embeddings[row]))
                print(f"Doc ID: {docid}, Base Score: {base_score}, BERT Score: {bert_score}")
                
                # Linear interpolation of BM25 and BERT scores
                final_score = self.alpha * base_score + (1 - self.alpha) * bert_score
                
                reranked_scores.append((docid, final_score))

            # Sort by combined score and append remaining documents
            reranked_scores.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_scores + base_results[self.rerank_top_k:]

        except Exception:
            return []
