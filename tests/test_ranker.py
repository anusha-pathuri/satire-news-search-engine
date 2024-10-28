import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.indexing import Indexer, IndexType
from src.ranker import Ranker, BM25, TF_IDF


DATASET_1_PATH = os.path.join(os.path.dirname(__file__), 'resources/dataset_1.jsonl')


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')


class MockTokenizer:
    def tokenize(self, text):
        return text.split()


class TestBM25(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_1_PATH, self.preprocessor, self.stopwords, 1)
        scorer = BM25(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, -0.31623109945742595), (3, -0.32042144088133173),
                    (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 1.5460888344441546), (3, 0.7257835477973098),
                    (1, -0.31623109945742595), (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


class TestTF_IDF(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_1_PATH, self.preprocessor, self.stopwords, 1)
        scorer = TF_IDF(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, 1.047224521431117),
                    (3, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.866760557116562), (3, 2.8559490532810434),
                    (1, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


if __name__ == '__main__':
    unittest.main()
