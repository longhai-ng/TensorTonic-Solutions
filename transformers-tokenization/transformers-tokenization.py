import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens =[self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        current_id = 0
        #register special token first
        for token in special_tokens:
            self.word_to_id[token] = current_id
            self.id_to_word[current_id] = token
            current_id +=1
        
        #go through all texts
        unique_words = set()
        for text in texts:
            words = text.split()
            for word in words:
                unique_words.add(word)

        for word in sorted(list(unique_words)):
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] = word
            current_id+=1

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = []
        words = text.split() #get words to encode
        unk_id = self.word_to_id[self.unk_token]
        for word in words:
            token_id = self.word_to_id.get(word, unk_id)
            ids.append(token_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for id in ids:
            word = self.id_to_word.get(id,self.unk_token)
            words.append(word)
        return " ".join(words)
