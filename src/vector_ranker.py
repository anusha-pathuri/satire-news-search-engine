import os
# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name)
        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not
        # submit that code to the autograder
        if len(encoded_docs) != len(row_to_docid):
            raise ValueError("Number of encoded documents must match number of document IDs")

        self.doc_embeddings = list(zip(row_to_docid, encoded_docs))

    def query(self, query: str, pseudofeedback_num_docs=0,
              pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: We don't use the user_id parameter in vector ranker. It is here just to align all the
                    Ranker interfaces.

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        pass
        # NOTE: Do not forget to handle edge cases on the input

        # TODO: Encode the query using the bi-encoder
        if not query or not isinstance(query, str):
            return []

        # TODO: Encode the query using the bi-encoder
        try:
            query_embedding = self.biencoder_model.encode(query)
        except Exception:
            return []

        # TODO (HW4): If the user has indicated we should use feedback
        if pseudofeedback_num_docs > 0:
            # TODO (HW4): Get the most-relevant document vectors for the initial query
            initial_results = []
            for docid, doc_embedding in self.doc_embeddings:
                # Use simple dot product
                similarity = float(np.dot(query_embedding, doc_embedding))
                initial_results.append((docid, similarity, doc_embedding))
            initial_results.sort(key=lambda x: x[1], reverse=True)

            # TODO (HW4): Compute the average vector of the specified number of most-relevant docs
            top_docs = initial_results[:pseudofeedback_num_docs]
            relevant_docs_avg = np.mean([doc[2] for doc in top_docs], axis=0)

            # TODO (HW4): Combine the original query doc with the feedback doc
            query_embedding = pseudofeedback_alpha * query_embedding + pseudofeedback_beta * relevant_docs_avg

        # TODO: Generate the ordered list of (document id, score) tuples
        results = []
        for docid, doc_embedding in self.doc_embeddings:
            # Use simple dot product
            similarity = float(np.dot(query_embedding, doc_embedding))
            results.append((docid, similarity))

        # TODO: Sort the list by relevance score in descending order
        return sorted(results, key=lambda x: x[1], reverse=True)

    # TODO (HW5): Find the dot product (unnormalized cosine similarity) for the list of documents (pairwise)
    # NOTE: You should return a matrix where element [i][j] would represent similarity between
    #   list_docs[i] and list_docs[j]
    def document_similarity(self, list_docs: list[int]) -> np.ndarray:
        """
        Calculates the pairwise similarities for a given list of documents

        Args:
            list_docs: A list of document IDs

        Returns:
            A matrix where element [i][j] is a similarity score between list_docs[i] and list_docs[j]
        """
        if not list_docs:
            return np.array([])

        # Create a dictionary for quick lookup of document embeddings
        doc_id_to_embedding = dict(self.doc_embeddings)

        # Get embeddings for the requested documents
        # Skip documents that don't exist in our collection
        embeddings = []
        for doc_id in list_docs:
            if doc_id in doc_id_to_embedding:
                embeddings.append(doc_id_to_embedding[doc_id])

        if not embeddings:
            return np.array([])

        # Convert list of embeddings to numpy array for efficient computation
        embeddings_matrix = np.array(embeddings)

        # Calculate pairwise similarities using dot product
        similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

        return similarity_matrix

