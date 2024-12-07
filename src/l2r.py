from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer


# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        # X = []
        # y = []
        # qgroups = []

        # # TODO: For each query and the documents that have been rated for relevance to that query,
        # #       process these query-document pairs into features

        #     # TODO: Accumulate the token counts for each document's title and content here

        #     # TODO: For each of the documents, generate its features, then append
        #     #       the features and relevance score to the lists to be returned

        #     # Make sure to keep track of how many scores we have for this query

        # return X, y, qgroups

        if not query_to_document_relevance_scores:
            return [], [], []

        X = []
        y = []
        qgroups = []

        for query, doc_relevance_scores in query_to_document_relevance_scores.items():
            if not query or not doc_relevance_scores:
                continue

            query_parts = self.document_preprocessor.tokenize(query)
            query_parts = [token for token in query_parts if token not in self.stopwords]

            if not query_parts:
                continue

            doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

            query_docs_features = []
            query_docs_relevance = []

            for docid, relevance in doc_relevance_scores:
                features = self.feature_extractor.generate_features(
                    docid,
                    doc_term_counts.get(docid, {}),
                    title_term_counts.get(docid, {}),
                    query_parts,
                    query  # Pass the original query string here
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
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        #       create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}
        for term in query_parts:
            postings = index.get_postings(term)
            for docid, freq in postings:
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
        # TODO: Convert the relevance data into the right format for training data preparation

        # TODO: Prepare the training data by featurizing the query-doc pairs and
        #       getting the necessary datastructures

        # TODO: Train the model
        df = pd.read_csv(training_data_filename, encoding='latin-1')

        training_data = df.groupby('query').apply(lambda x: list(zip(x['docid'].astype(int), x['rel'].astype(int)))).to_dict()

        X, y, qgroups = self.prepare_training_data(training_data)

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
        # TODO: Return a prediction made using the LambdaMART model
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)

    # TODO (HW5): Implement MMR diversification for a given list of documents and their cosine similarity scores
    @staticmethod
    def maximize_mmr(thresholded_search_results: list[tuple[int, float]], similarity_matrix: np.ndarray,
                     list_docs: list[int], mmr_lambda: int) -> list[tuple[int, float]]:
        """
        Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm
        on the list.
        It should return a list of the same length with the same overall documents but with different document ranks.

        Args:
            thresholded_search_results: The thresholded search results
            similarity_matrix: Precomputed similarity scores for all the thresholded search results
            list_docs: The list of documents following the indexes of the similarity matrix
                       If document 421 is at the 5th index (row, column) of the similarity matrix,
                       it should be on the 5th index of list_docs.
            mmr_lambda: The hyperparameter lambda used to measure the MMR scores of each document

        Returns:
            A list containing tuples of the documents and their MMR scores when the documents were added to S
        """
        # NOTE: This algorithm implementation requires some amount of planning as you need to maximize
        #       the MMR at every step.
        #       1. Create an empty list S
        #       2. Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        #       3. Move that element from R and append it to S
        #       4. Repeat 2 & 3 until there are no more remaining elements in R to be processed

        # 1. Create an empty list S
        S = []

        # Create mapping from docid to its position in list_docs for quick lookups
        doc_to_idx = {docid: idx for idx, docid in enumerate(list_docs)}

        # Create a copy of thresholded_search_results as our R set
        R = thresholded_search_results.copy()

        # 2 & 3 & 4. Repeat until R is empty:
        # - Find element with maximum MMR in R (but not in S)
        # - Move that element from R and append it to S
        while R:
            max_mmr_score = float('-inf')
            max_mmr_doc = None
            max_mmr_idx = None

            # Find element with maximum MMR
            for idx, (docid, rel_score) in enumerate(R):
                doc_idx = doc_to_idx[docid]

                # Calculate MMR = λ*rel(di) - (1-λ)*max(sim(di,dj))
                relevance_term = mmr_lambda * rel_score

                if S:
                    # Get max similarity to docs in S
                    similarities = [similarity_matrix[doc_idx][doc_to_idx[selected_doc]]
                                for selected_doc, _ in S]
                    diversity_term = (1 - mmr_lambda) * max(similarities)
                else:
                    diversity_term = 0

                mmr_score = relevance_term - diversity_term

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_doc = docid
                    max_mmr_idx = idx

            # Move element from R to S
            S.append((max_mmr_doc, max_mmr_score))
            R.pop(max_mmr_idx)

        return S

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None, mmr_lambda:float=1, mmr_threshold:int=100) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query

            HW4:
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown

            HW5:
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking

        # TODO: Construct the feature vectors for each query-document pair in the top 100

        # TODO: Use your L2R model to rank these top 100 documents

        # TODO: Sort posting_lists based on scores

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked

        # TODO (HW5): Run MMR diversification for appropriate values of lambda

        # TODO (HW5): Get the threholded part of the search results, aka top t results and
        #      keep the rest separate

        # TODO (HW5): Get the document similarity matrix for the thresholded documents using vector_ranker
        #      Preserve the input list of documents to be used in the MMR function

        # TODO (HW5): Run the maximize_mmr function with appropriate arguments

        # TODO (HW5): Add the remaining search results back to the MMR diversification results

        # TODO: Return the ranked documents
        if not query:
            return []

        query_parts = self.document_preprocessor.tokenize(query)
        query_parts = [token for token in query_parts if token not in self.stopwords]

        if not query_parts:
            return []

        # Use the ranker to get initial candidates
        initial_ranking = self.ranker.query(
            query,
            pseudofeedback_num_docs=pseudofeedback_num_docs,
            pseudofeedback_alpha=pseudofeedback_alpha,
            pseudofeedback_beta=pseudofeedback_beta
        )

        if not initial_ranking:
            return []

        # Get term counts for feature generation
        doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

        if not doc_term_counts and not title_term_counts:
            return []

        # Take top 100 documents for re-ranking
        top_100_docs = initial_ranking[:100]

        # Prepare features for top 100 documents
        X = [self.feature_extractor.generate_features(
            docid,
            doc_term_counts.get(docid, {}),
            title_term_counts.get(docid, {}),
            query_parts,
            query  # Pass original query for cross-encoder scoring
        ) for docid, _ in top_100_docs]

        # Re-rank top 100 documents using the L2R model
        reranked_scores = self.predict(X)

        # Combine re-ranked documents with their new scores
        reranked_docs = list(zip([docid for docid, _ in top_100_docs], reranked_scores))
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Combine re-ranked documents with the rest
        final_ranking = reranked_docs + initial_ranking[100:]

        # Only apply MMR diversification if lambda < 1
        if mmr_lambda < 1:
            # Get thresholded part of search results
            thresholded_results = final_ranking[:mmr_threshold]
            remaining_results = final_ranking[mmr_threshold:]

            # Extract document IDs for similarity matrix calculation
            thresholded_docs = [docid for docid, _ in thresholded_results]

            # Get similarity matrix using the ranker's get_document_similarities method
            # Note: This assumes ranker has this method - you may need to implement it
            similarity_matrix = self.ranker.document_similarity(thresholded_docs)

            # Run MMR diversification
            diversified_results = self.maximize_mmr(
                thresholded_results,
                similarity_matrix,
                thresholded_docs,
                mmr_lambda
            )

            # Combine diversified results with remaining results
            final_ranking = diversified_results + remaining_results

        return final_ranking


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features
        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing
        self.category_to_index = {category: i for i, category in enumerate(sorted(self.recognized_categories))}
        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring
        self.doc_tf_scorer = TF(document_index)
        self.doc_tf_idf_scorer = TF_IDF(document_index)
        self.title_tf_scorer = TF(title_index)
        self.title_tf_idf_scorer = TF_IDF(title_index)

        self.bm25_scorer = BM25(document_index)
        self.pivoted_normalization_scorer = PivotedNormalization(document_index)
        self.ce_scorer = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        if index == self.document_index:
            return self.doc_tf_scorer.score(docid, word_counts, Counter(query_parts))
        elif index == self.title_index:
            return self.title_tf_scorer.score(docid, word_counts, Counter(query_parts))
        else:
            raise ValueError("Invalid index provided")

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        if index == self.document_index:
            return self.doc_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))
        elif index == self.title_index:
            return self.title_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))
        else:
            raise ValueError("Invalid index provided")

    # TODO: BM25
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
        # TODO: Calculate the BM25 score and return it
        return self.bm25_scorer.score(docid, doc_word_counts, Counter(query_parts))

    # TODO: Pivoted Normalization
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
        # TODO: Calculate the pivoted normalization score and return it
        return self.pivoted_normalization_scorer.score(docid, doc_word_counts, Counter(query_parts))

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        document_categories = set(self.doc_category_info.get(docid, []))

        # Create a binary feature vector
        feature_vector = [0] * len(self.recognized_categories)

        for category in document_categories:
            if category in self.category_to_index:
                feature_vector[self.category_to_index[category]] = 1

        return feature_vector

    # TODO: PageRank
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        return self.docid_to_network_features.get(docid, {}).get('pagerank', 0.0)

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        return self.docid_to_network_features.get(docid, {}).get('hub_score', 0.0)

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        return self.docid_to_network_features.get(docid, {}).get('authority_score', 0.0)

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        try:
            return self.ce_scorer.score(docid, query)
        except:
            return 0.0

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO: Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # TODO: PageRank
        feature_vector.append(self.get_pagerank_score(docid))

        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO: Cross-Encoder Score

        # TODO: Document Categories
        #       This should be a list of binary values indicating which categories are present
        feature_vector.extend(self.get_document_categories(docid))

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
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like):
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generate the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)

