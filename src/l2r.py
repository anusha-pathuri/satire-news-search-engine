import lightgbm
import pandas as pd
from collections import Counter

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.document_preprocessor import Tokenizer
from src.indexing import InvertedIndex, BasicInvertedIndex
from src.ranker import *


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor

        # Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of
                documents and their relevance scores for that query.
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relevance_to_query_1), (docid_2, relevance_to_query_2), ...]

        Returns:
            A tuple containing the training data in the form of three lists: X, Y, and qgroups
                X: A list of feature vectors for each query-document pair
                y: A list of relevance scores for each query-document pair
                qgroups: A list of the number of documents retrieved for each query
        """
        # Handle empty input case
        if not query_to_document_relevance_scores:
            return [], [], []

        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # For each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features
        for query, doc_relevance_scores in query_to_document_relevance_scores.items():
            if not query or not doc_relevance_scores:
                continue

            # Tokenize the query
            query_parts = self.document_preprocessor.tokenize(query)
            # Don't remove stopwords here
            # Query length in L2R feature extractor is for the entire query, including stopwords

            # Accumulate the token counts for each document's title and content
            doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

            # For each of the documents, generate its features, then append
            # the features and relevance score to the lists to be returned
            query_docs_features = []
            query_docs_relevance = []
            for docid, relevance in doc_relevance_scores:
                features = self.feature_extractor.generate_features(
                    docid, doc_term_counts.get(docid, {}), title_term_counts.get(docid, {}), query_parts
                )
                query_docs_features.append(features)
                query_docs_relevance.append(relevance)

            if query_docs_features:
                X.extend(query_docs_features)
                y.extend(query_docs_relevance)
                qgroups.append(len(query_docs_relevance))

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}  # {docid: {word: count}} for only the query terms

        for term in set(query_parts):  # only consider unique query terms
            if term in index.vocabulary:  # exclude filtered terms
                for docid, freq in index.get_postings(term):
                    if docid not in doc_term_counts:
                        doc_term_counts[docid] = {}

                    doc_term_counts[docid][term] = freq

        return doc_term_counts

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # Convert the relevance data into the right format for training data preparation
        df = pd.read_csv(training_data_filename, encoding='utf-8')

        training_data = df[df.docid.isin(self.document_index.document_metadata)]\
            .groupby('query')\
            .apply(lambda x: list(zip(x['docid'].astype(int), x['rel'].astype(int))))\
            .to_dict()

        # Prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        X, y, qgroups = self.prepare_training_data(training_data)

        # Train the model
        self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Return a prediction made using the LambdaMART model
        return self.model.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        if not query:
            return []

        # Tokenize the query
        query_parts = self.document_preprocessor.tokenize(query)
        query_parts = [token for token in query_parts if token not in self.stopwords]

        # If none of the query terms are in the index, return an empty list
        if not any(term in self.document_index.vocabulary for term in query_parts):
            return []

        # Retrieve potentially-relevant documents
        # 1. Using BM25, or
        # 2. Using the bi-encoder ranker
        initial_ranking = self.ranker.query(query)

        if not initial_ranking:
            return []

        doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

        if not doc_term_counts and not title_term_counts:
            return []

        # Take top 100 documents for re-ranking
        top_100_docs = initial_ranking[:100]

        # Prepare features for top 100 documents
        X = [self.feature_extractor.generate_features(docid,
                                                      doc_term_counts.get(docid, {}),
                                                      title_term_counts.get(docid, {}),
                                                      query_parts)
            for docid, _ in top_100_docs]

        # Re-rank top 100 documents using the L2R model
        reranked_scores = self.predict(X)

        # Combine re-ranked documents with their new scores
        reranked_docs = list(zip([docid for docid, _ in top_100_docs], reranked_scores))
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Combine re-ranked documents with the rest
        final_ranking = reranked_docs + initial_ranking[100:]

        return final_ranking


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str]) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
        """
        # Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords or set()

        # Initialize RelevanceScorer objects to support the methods below
        self.doc_tf_scorer = TF(document_index)
        self.doc_tf_idf_scorer = TF_IDF(document_index)
        self.title_tf_scorer = TF(title_index)
        self.title_tf_idf_scorer = TF_IDF(title_index)
        self.bm25_scorer = BM25(document_index)
        self.pivoted_normalization_scorer = PivotedNormalization(document_index)

    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid).get("length", 0)

    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid).get("length", 0)

    def get_doc_tf(self, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score with respect to the document body.

        Args:
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        return self.doc_tf_scorer.score(docid, word_counts, Counter(query_parts))

    def get_doc_tf_idf(self, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score with respect to the document body.

        Args:
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        return self.doc_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))

    def get_title_tf(self, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score with respect to the document title.

        Args:
            docid: The id of the document
            word_counts: The words in the document's title mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        return self.title_tf_scorer.score(docid, word_counts, Counter(query_parts))

    def get_title_tf_idf(self, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score with respect to the document title.

        Args:
            docid: The id of the document
            word_counts: The words in the document's title mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        return self.title_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        return self.bm25_scorer.score(docid, doc_word_counts, Counter(query_parts))

    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        return self.pivoted_normalization_scorer.score(docid, doc_word_counts, Counter(query_parts))

    def get_sarcasm_score(self, docid: int) -> float:
        """
        Gets the sarcasm score for a document.

        Args:
            docid: The id of the document

        Returns:
            The sarcasm score of the document, or 0.0 if not available
        """
        return self.document_index.get_doc_metadata(docid).get('sarcasm_score', 0.0)

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        feature_vector = []

        # Document Length
        feature_vector.append(self.get_article_length(docid))

        # Title Length
        feature_vector.append(self.get_title_length(docid))

        # Query Length
        feature_vector.append(len(query_parts))

        # TF (document)
        feature_vector.append(self.get_doc_tf(docid, doc_word_counts, query_parts))

        # TF-IDF (document)
        feature_vector.append(self.get_doc_tf_idf(docid, doc_word_counts, query_parts))

        # TF (title)
        feature_vector.append(self.get_title_tf(docid, title_word_counts, query_parts))

        # TF-IDF (title)
        feature_vector.append(self.get_title_tf_idf(docid, title_word_counts, query_parts))

        # BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # Sarcasm Score
        feature_vector.append(self.get_sarcasm_score(docid))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
            # "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # Fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like):
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # Generate the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)