import os
import csv
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.document_preprocessor import SplitTokenizer, Tokenizer, RegexTokenizer
from src.utils import load_txt


class IndexType(Enum):
    BasicInvertedIndex = 'BasicInvertedIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """
    
    Statistics = ['unique_token_count', 'total_token_count', 'stored_total_token_count',
                  'number_of_documents', 'mean_document_length']

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.index = defaultdict(list)  # the index (a mapping of terms to their postings)
        self.statistics = dict.fromkeys(InvertedIndex.Statistics, 0)  # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {}  # metadata like length, number of unique tokens of the documents
        self.document_text = {}  # the first 500 words of each document
        
    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.split_tokenizer = SplitTokenizer(lowercase=False)

    def add_doc(self, docid: int, tokens: list[str], **metadata) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        if docid in self.document_metadata:  # if document already in index
            return  # nothing to add
        
        n_tokens_indexed = 0
        term_postings_idx = {}  # term -> index in postings list

        for token in tokens:            
            if token is None:  
                continue  # index valid tokens only
            
            n_tokens_indexed += 1   
            if token in term_postings_idx:
                doc_idx = term_postings_idx[token]
                self.index[token][doc_idx][1] += 1
            else:
                term_postings_idx[token] = len(self.index[token])
                self.index[token].append([docid, 1])
                self.vocabulary.add(token)
            
            self.statistics["vocab"][token] += 1  # term frequency in collection
            
        # add new metadata entry for document
        self.document_metadata[docid] = {
            "unique_tokens": len(term_postings_idx), 
            "length": len(tokens), 
            **metadata
        }

        # update index statistics
        self.statistics["total_token_count"] += len(tokens)  # total tokens processed (including filtered)
        self.statistics["stored_total_token_count"] += n_tokens_indexed  # total tokens indexed (excluding filtered)
        self.statistics["unique_token_count"] = len(self.vocabulary)  # unique terms in the index
        self.statistics["number_of_documents"] += 1  # collection size
        self.statistics["mean_document_length"] = self.statistics["total_token_count"] / \
            self.statistics["number_of_documents"]  # average document length (including filtered tokens)
            
    def store_doc_text(self, docid: int, text: str, num_words: int = 500) -> None:
        """
        Stores the first few words of the document 
        This should be called in the Indexer before filtering and tokenization.
        
        Args:
            docid: The id of the document
            text: Raw text of the document
            num_words: Number of words to store
        """
        if not text:
            self.document_text[docid] = ""
        else:
            self.document_text[docid] = " ".join(self.split_tokenizer.tokenize(text)[:num_words])

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid not in self.document_metadata:  # document not in index
            raise KeyError(f"Can't remove non-existent document {docid} from the index.")

        n_tokens_deleted = 0
        n_unique_tokens_deleted = 0

        for term in list(self.index):
            for idx, doc in enumerate(self.index[term]):
                if doc[0] == docid:
                    count = self.index[term].pop(idx)[1]  # delete posting and get term frequency in document
                    self.statistics["vocab"][term] -= count  # term frequency in collection
                    n_tokens_deleted += count
                    n_unique_tokens_deleted += 1

                    if not self.index[term]:  # if no more postings for the term
                        self.vocabulary.remove(term)
                        del self.index[term]
                        del self.statistics["vocab"][term]
                                
                    break  # document can't have multiple postings for the same term

            if n_unique_tokens_deleted == self.document_metadata[docid]["unique_tokens"]:
                break  # all terms in the document have been deleted        

        self.statistics["total_token_count"] -= self.document_metadata[docid]["length"]  # total tokens (including filtered)  
        self.statistics["stored_total_token_count"] -= n_tokens_deleted  # total tokens (excluding filtered)
        self.statistics["unique_token_count"] = len(self.vocabulary)  # unique terms in the index
        self.statistics["number_of_documents"] -= 1
        self.statistics["mean_document_length"] = 0 if not self.statistics["number_of_documents"] \
            else self.statistics["total_token_count"] / self.statistics["number_of_documents"]

        del self.document_metadata[docid]  # delete document metadata
        del self.document_text[docid]  # delete document text

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index.get(term, [])

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        term_metadata = {}
        if term in self.vocabulary:
            term_metadata["term_count"] = self.statistics["vocab"][term]
            term_metadata["doc_frequency"] = len(self.get_postings(term))
        
        return term_metadata

    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """  
        return self.document_metadata.get(docid, {})
    
    def get_doc_text(self, docid: int) -> str:
        """
        For the given document id, returns the stored text of the document.
        
        Args:
            docid: The id of the document

        Returns:
            Raw text of the document
        """
        return self.document_text.get(docid, "")
        
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        return self.statistics

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        os.makedirs(index_directory_name, exist_ok=True)

        with open(os.path.join(index_directory_name, "index.json"), "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False)
        
        with open(os.path.join(index_directory_name, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(self.statistics, f, ensure_ascii=False)
            
        with open(os.path.join(index_directory_name, "documents.json"), "w", encoding="utf-8") as f:
            document_data = {
                "metadata": self.document_metadata,
                "text": self.document_text
            }
            json.dump(document_data, f, ensure_ascii=False)

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        if not os.path.isdir(index_directory_name):
            raise FileNotFoundError(f"No such directory: {index_directory_name}")

        if not all(
            f in os.listdir(index_directory_name) 
            for f in ["index.json", "statistics.json", "documents.json"]
        ):
            raise FileNotFoundError("At least one file needed to load the index does not exist.")
        
        with open(os.path.join(index_directory_name, "index.json"), "r", encoding="utf-8") as f:
            # Convert serialized dictionary back to a collections.defaultdict object
            self.index = defaultdict(list, json.load(f))
            self.vocabulary = set(self.index.keys())
        
        with open(os.path.join(index_directory_name, "statistics.json"), "r", encoding="utf-8") as f:
            # Convert serialized dictionaries back to collections.Counter objects
            self.statistics = {
                key: (val if isinstance(val, (int, float)) else Counter(val))
                for key, val in json.load(f).items()
            }
        
        with open(os.path.join(index_directory_name, "documents.json"), "r", encoding="utf-8") as f:
            # JSON considers keys as strings, so convert them back to numbers
            document_data = json.load(f)
            self.document_metadata = {
                int(key): val for key, val in document_data["metadata"].items()
            }
            self.document_text = { 
                int(key): val for key, val in document_data["text"].items()
            }


class Indexer:
    '''
    Create the index used by the search/ranking algorithm.
    '''
    IndexClass = {
        IndexType.BasicInvertedIndex: BasicInvertedIndex,
    }

    @staticmethod
    def create_index(
            index_type: IndexType, 
            dataset_path: str,
            document_preprocessor: Tokenizer, 
            stopwords: set[str] = None,
            minimum_word_frequency: int = 1, 
            text_key: str = "text", 
            id_key: str = "docid", 
            source_key: str = "website",
            max_docs: int = -1, 
        ) -> InvertedIndex:
        '''
        Create an inverted index from a dataset.
        
        Args:
            index_type: The type of index to create; currently only supports BasicInvertedIndex.
            dataset_path: The file path to your dataset; supports .jsonl, .jsonl.gz, .csv, .csv.gz.
            document_preprocessor: Instance of a class with a 'tokenize' function that 
                takes as input some text and returns a list of valid tokens.
            stopwords: The set of stopwords to remove during preprocessing;
                set to 'None' if no stopword filtering is to be done.
            minimum_word_frequency: The minimum corpus frequency for a term to be indexed,
                defaults to 1 (i.e., all terms are indexed).
            text_key: The key in the JSON/CSV corresponding to the document text.
            id_key: The key in the JSON/CSV corresponding to the document ID.
            max_docs: The maximum number of documents to index; -1 to index all documents.
                Documents are processed in the order they are seen.
                
        Returns:
            An inverted index
        '''
        # Initialize the index     
        if index_type not in Indexer.IndexClass:  
            raise NotImplementedError(
                "Currently supported index types: BasicInvertedIndex, PositionalInvertedIndex.")
        
        index = Indexer.IndexClass[index_type]()

        # Process the documents
        if max_docs == 0:
            return index

        open_func = gzip.open if dataset_path.endswith(".gz") else open
        mode = 'rt' if dataset_path.endswith('.gz') or dataset_path.endswith('.csv') else 'r'
        with open_func(dataset_path, mode, encoding="utf-8") as f:
            is_csv = dataset_path.endswith('.csv') or dataset_path.endswith('.csv.gz')
            iterator = csv.DictReader(f) if is_csv else f
            
            # Load documents line-by-line
            for i, item in enumerate(tqdm(iterator), start=1):
                doc = item if is_csv else json.loads(item)
                docid = int(doc[id_key])
                text = doc[text_key]
                
                # Store the first 500 words in the document (raw text before filtering and tokenization)
                index.store_doc_text(docid, text, num_words=500)

                # Tokenize the document
                tokens = document_preprocessor.tokenize(text)

                # Store document metadata
                metadata = {
                    'source': doc[source_key] if source_key in doc else None,
                    'nsfw': doc['nsfw'] if 'nsfw' in doc else False,
                    'sarcasm_score': doc['sarcasm_score'] if 'sarcasm_score' in doc else None,
                }
                
                # Add the document to the index
                index.add_doc(docid, tokens, **metadata)

                if i == max_docs:
                    break

        # Optional stop-word filtering (remove common words)
        # Optional minimum word frequency filtering (remove rare words)
        if stopwords or minimum_word_frequency > 1:
            n_tokens_deleted = 0
            n_docs_deleted = 0
            
            for term, count in list(index.statistics["vocab"].items()):
                if term in stopwords or count < minimum_word_frequency:
                    n_tokens_deleted += count

                    for docid, *_ in index.get_postings(term):
                        index.document_metadata[docid]["unique_tokens"] -= 1

                        if index.document_metadata[docid]["unique_tokens"] == 0:  # all terms in the document are filtered
                            index.statistics["total_token_count"] -= index.document_metadata[docid]["length"]
                            del index.document_metadata[docid]
                            n_docs_deleted += 1
                            
                    index.vocabulary.remove(term)
                    del index.index[term]
                    del index.statistics["vocab"][term]
            
            if n_tokens_deleted > 0:
                index.statistics["stored_total_token_count"] -= n_tokens_deleted
                index.statistics["unique_token_count"] = len(index.vocabulary)
                index.statistics["number_of_documents"] -= n_docs_deleted
                index.statistics["mean_document_length"] = 0 if not index.statistics["number_of_documents"] \
                    else index.statistics["total_token_count"] / index.statistics["number_of_documents"]

        return index


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    dataset_path = os.path.join(data_dir, 'processed_articles_dedup_nsfwtags_sarcasm.csv')
    multiwords_path = os.path.join(data_dir, 'multiword_expressions.txt')
    stopwords_path = os.path.join(data_dir, 'stopwords_updated.txt')
    
    mwes = load_txt(multiwords_path) if os.path.exists(multiwords_path) else None
    stopwords = set(load_txt(stopwords_path)) if os.path.exists(stopwords_path) else set()
    
    preprocessor = RegexTokenizer("\w+(?:-\w+)*(?:'[^stmrvld]\w*)*", lowercase=True, multiword_expressions=mwes)
    index = Indexer.create_index(IndexType.BasicInvertedIndex, dataset_path, preprocessor, stopwords, 0, max_docs=1000, text_key='body')
    index.save(os.path.join(os.path.dirname(__file__), '..', '__cache__'))
    pass
