'''
Author: Zim Gong
Modified by: Anusha Pathuri
'''
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import BaseSearchEngine, SearchResponse
from src.document_preprocessor import RegexTokenizer
from src.indexing import Indexer, IndexType
from src.ranker import Ranker, BM25, TF_IDF
from src.utils import load_txt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', '__cache__')

STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')
DATASET_PATH = os.path.join(DATA_DIR, 'processed_articles_dedup_nsfwtags.csv')


def py2js_bool(value):
    return str(value).lower()


class SearchEngine(BaseSearchEngine):
    def __init__(self, max_docs: int = -1, ranker: str = 'BM25') -> None:
        print('Initializing Search Engine...')
        self.stopwords = set(load_txt(STOPWORDS_PATH))

        print('Loading indexes...')
        self.preprocessor = RegexTokenizer("\w+(?:-\w+)*(?:'[^stmrvld]\w*)*", lowercase=True)  
        self.document_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor,
            self.stopwords, 0, max_docs=max_docs, text_key='body'
        )
        self.title_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor,
            self.stopwords, 0, max_docs=max_docs, text_key='headline'
        )

        print('Loading ranker...')
        self.set_ranker(ranker)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'BM25') -> None:
        if ranker == 'BM25':
            self.scorer = BM25(self.document_index)
        elif ranker == 'TF_IDF':
            self.scorer = TF_IDF(self.document_index)
        else:
            raise ValueError("Invalid ranker type")
        
        self.ranker = Ranker(self.document_index, self.preprocessor, self.stopwords, 
                             self.scorer, score_top_k=100)
        self.pipeline = self.ranker

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [
            SearchResponse(id=idx+1, 
                           docid=result[0], 
                           score=result[1], 
                           title=self.title_index.get_doc_text(result[0]), 
                           text=self.document_index.get_doc_text(result[0]),
                           source=self.document_index.get_doc_metadata(result[0])['source'],
                           nsfw=py2js_bool(self.document_index.get_doc_metadata(result[0])['nsfw']))
            for idx, result in enumerate(results)
        ]


def initialize():
    search_obj = SearchEngine(max_docs=-1)  # set this to a smaller number for testing the app
    return search_obj


def main():
    search_obj = SearchEngine(max_docs=1000)
    search_obj.set_ranker('BM25')
    query = "american"
    results = search_obj.search(query)
    print(len(results))
    for result in results[:5]:
        print(result)


if __name__ == '__main__':
    main()
