from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from src.indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """

    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: RelevanceScorer, raw_text_dict: dict[int, str] = None,
                 score_top_k: Optional[int] = 100) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
            score_top_k: The number of top documents to score and return; by default, 100
                if None, all documents are scored
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.score_top_k = score_top_k
        self.stopwords = stopwords or set()
        self.raw_text_dict = raw_text_dict or dict()
        self.docs_word_counts = defaultdict(dict)  # {docid: {word: count}} for only the terms of queries seen so far

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        query_tokens = self.tokenize(query)

        query_word_counts = defaultdict(int)
        filtered_tokens = 0
        candidate_docs = []
        docs_word_counts = defaultdict(dict)  # {docid: {word: count}} for only the query terms

        for q_token in query_tokens:
            if q_token in self.index.vocabulary and q_token not in self.stopwords:
                if q_token not in query_word_counts:  # first time seeing this term
                    # Fetch a list of possible documents from the index
                    for docid, count, *_ in self.index.get_postings(q_token):
                        candidate_docs.append(docid)
                        # Store the word count for the query term in the document
                        docs_word_counts[docid][q_token] = count
                
                query_word_counts[q_token] += 1
            else:
                filtered_tokens += 1

        query_word_counts[None] = filtered_tokens  # stopwords and unknown words
        
        # Update the document word counts with terms from the current query
        for docid, doc_word_counts in docs_word_counts.items():
            self.docs_word_counts[docid].update(doc_word_counts)

        # Filter candidates by selecting the top k that have the most query terms 
        candidate_docs = Counter(candidate_docs)
        if self.score_top_k:
            candidate_docs = dict(candidate_docs.most_common(self.score_top_k))

        # Run RelevanceScorer (like BM25 from below classes)
        document_scores = [
            (docid, self.scorer.score(docid, docs_word_counts[docid], query_word_counts))
            for docid in candidate_docs
        ]

        # Return **sorted** results in the format [(100, 0.5), (10, 0.2), ...]
        return sorted(document_scores, key=lambda x: x[1], reverse=True)


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']  # doc length importance
        self.k1 = parameters['k1']  # doc TF scaling
        self.k3 = parameters['k3']  # query TF scaling
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # Get necessary information from index
        index_stats = self.index.get_statistics()
        n_docs = index_stats["number_of_documents"]
        avdl = index_stats["mean_document_length"]
        doc_len = self.index.get_doc_metadata(docid)["length"]
        
        # Find the dot product of the word count vector of the document and the word count vector of the query
        tf_norm_denom = 1 - self.b + self.b * doc_len / avdl

        # For all query parts, compute the TF and IDF to get a score
        score = 0
        for q_term, query_tf in query_word_counts.items():
            if q_term and q_term in self.index.vocabulary:  # if not stopword and present in index
                doc_tf = doc_word_counts.get(q_term, 0)  # document TF
                if doc_tf > 0:
                    # A variant form of IDF
                    df = self.index.get_term_metadata(q_term)["doc_frequency"]
                    idf = np.log((n_docs - df + 0.5) / (df + 0.5))
                    # A variant form of (normalized) document TF
                    tf_norm = (
                        ((self.k1 + 1) * doc_tf) /
                        (self.k1 * tf_norm_denom + doc_tf)
                    )
                    # Normalized query TF
                    qtf_norm = (
                        ((self.k3 + 1) * query_tf) /
                        (self.k3 + query_tf)
                    )
                    score += (idf * tf_norm * qtf_norm)

        return score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # Get necessary information from index
        index_stats = self.index.get_statistics()
        n_docs = index_stats["number_of_documents"]

        # For all query parts, compute a score
        score = 0
        for q_term in query_word_counts:
            if q_term and q_term in self.index.vocabulary:  # if not stopword and present in index
                doc_tf = doc_word_counts.get(q_term, 0)  # document TF
                if doc_tf > 0:
                    tf = np.log(doc_tf + 1)
                    df = self.index.get_term_metadata(q_term)["doc_frequency"] 
                    idf = np.log(n_docs / df) + 1
                    score += (tf * idf)

        return score


if __name__ == "__main__":
    pass