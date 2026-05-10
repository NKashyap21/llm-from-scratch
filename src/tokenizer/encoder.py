import torch 
from torch.utils.data import Dataset,DataLoader
import tiktoken
import re 

class SimpleTokenizerV1:
    """Tokenizer that splits texts into words and punctuation. Will give error for new words"""
    def __init__(self,vocab:dict):
        self.str_to_int = vocab
        self.int_to_str = {s:i for i,s in vocab.items()}

    def encode(self,text):
        tokens = re.split(r'([.,?!:;"()\']|--|\s)',text)
        tokens = [item for item in tokens if item.strip()]
        ids = [self.str_to_int[s] for s in tokens]
        return ids 
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #Removes whitespace in front of the punctuation
        return text

class SimpleTokenizerV2:
    """Tokenizer that splits texts into words and punctuation. Will replace unkown words with <|unk|>"""
    def __init__(self,vocab:dict):
        self.str_to_int = vocab
        self.int_to_str = {s:i for i,s in vocab.items()}
    
    def encode(self,text):
        tokens = re.split(r'([.,?!:;"()\']|--|\s)',text)
        tokens = [item for item in tokens if item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in tokens]

        ids = [self.str_to_int[s] for s in tokens]
        return ids 

    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #Removes whitespace in front of the punctuation
        return text

class GPTEncoder:
    """Calls the tiktoken library for enoding and decoding. Uses gpt2 encoding."""
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2") 

    def encode(self,text,allowed_special={"<|endoftext|>"}):
        return self.tokenizer.encode(text,allowed_special=allowed_special) 
    
    def decode(self,ids):
        return self.tokenizer.decode(ids) 
    
