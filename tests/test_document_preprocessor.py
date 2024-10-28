import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.document_preprocessor import SplitTokenizer, RegexTokenizer


############ =======Test SplitTokenizer=========== ############
class TestSplitTokenizer(unittest.TestCase):
    """Test SplitTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = SplitTokenizer()
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["michigan"]
        tokenizer = SplitTokenizer()
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_doc_with_punc(self):
        """Test tokenizing a document with punctuations."""
        text = "This is a test sentence."
        expected_tokens = ['This', 'is', 'a', 'test', 'sentence.']
        tokenizer = SplitTokenizer(lowercase=False)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
    def test_unicode_text(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to the United States 🫥"
        expected_tokens = ['Welcome', 'to', 'the', 'United', 'States', '🫥']
        tokenizer = SplitTokenizer(lowercase=False)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)


############ =======Test RegexTokenizer=========== ############
class TestRegexTokenizer(unittest.TestCase):
    """Test RegexTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["michigan"]
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_doc_with_punc(self):
        """Test tokenizing a document with punctuations."""
        text = "RegexTokenizer can split on punctuation, like this: test!"
        expected_tokens = ['regextokenizer', 'can', 'split', 'on', 'punctuation', 'like', 'this', 'test']
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_unicode_text(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to the United States 🫥"
        expected_tokens = ['welcome', 'to', 'the', 'united', 'states']
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        

if __name__ == '__main__':
    unittest.main()

