from typing import Literal
import openai
import tiktoken
import numpy as np
import json

from .batch_token_embedder import process_token_batch

type TEmbeddingModel = Literal[
  "text-embedding-ada-002", 
  "text-embedding-babbage-002", 
  "text-embedding-curie-002", 
  "text-embedding-davinci-002"
]

type TTokenEncoding = Literal[
  'cl100k_base',
  'p50k_base',
  'r_50k_base'
]

def get_max_token_value(encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.vocab_size - 1

def get_embeddings(token_batch, model):
    response = openai.Embedding.create(input=token_batch, model=model)
    return response['data']

def main(
  model:TEmbeddingModel="text-embedding-ada-002",
  encoding:TTokenEncoding="cl100k_base",
  request_limit:int=1000,
  start:int=0
):
    openai.api_key = "your-openai-api-key"
    max_token = get_max_token_value(encoding)
    first_token = start

    while first_token <= max_token:
        end_token = min(first_token + request_limit, max_token + 1)

        token_embeddings = process_token_batch(first_token, end_token, model)

        with open('token_embeddings.jsonl', 'a') as file:
            for item in token_embeddings:
                json.dump(item, file)
                file.write('\\n')

        first_token = end_token
