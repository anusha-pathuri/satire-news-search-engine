from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.mwe import MWETokenizer


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        self.lowercase = lowercase
        
        self.multiword_expressions = multiword_expressions
        if self.multiword_expressions:
            if self.lowercase:
                self.multiword_expressions = list(set(map(str.lower, self.multiword_expressions)))
    
            self.mwe_tokenizer = MWETokenizer(
                [tuple(mwe.split()) for mwe in self.multiword_expressions], 
                separator=' '
            )
        else:
            self.mwe_tokenizer = None

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """            
        if self.lowercase:
            input_tokens = list(map(str.lower, input_tokens))
            
        if self.mwe_tokenizer:
            input_tokens = self.mwe_tokenizer.tokenize(input_tokens)

        return input_tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        if text is None or not text.strip():
            return []
        
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Tokenize a given string using NLTK's RegexpTokenizer with optional lower-casing and multi-word expression handling.

        Args:
            token_regex: The regular expression pattern to use for tokenization, '\w+' by default.
            lowercase: True if you want to lowercase all the tokens.
            multiword_expressions: A list of strings that should be recognized as single tokens.
                If set to 'None', no multi-word expression matching is performed.
        """
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize a given string.

        Args:
            text: The input string to be tokenized.

        Returns:
            A list of tokens.
        """
        if text is None or not text.strip():
            return []

        tokens = self.tokenizer.tokenize(text)
        return self.postprocess(tokens)


if __name__ == '__main__':
    pass
