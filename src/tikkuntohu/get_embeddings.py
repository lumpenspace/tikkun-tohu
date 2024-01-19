import tikkuntohu.openai as openai
import tiktoken
import numpy as np
import json

from .tikkuntohu_types import TEmbeddingModel, TTokenEncoding
from .batch_token_embed import process_token_batch


def get_max_token_value(encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.vocab_size - 1

def get_embeddings(token_batch, model):
  """
  Retrieves embeddings for a batch of tokens using the specified model.

  Args:
    token_batch (list): A list of tokens.
    model (str): The name of the model to use for embedding.

  Returns:
    list: A list of embeddings for the input tokens.
  """
  response = openai.Embedding.create(input=token_batch, model=model)
  return response['data']

def main(
  model:TEmbeddingModel="text-embedding-ada-002",
  encoding:TTokenEncoding="cl100k_base",
  request_limit:int=1000,
  start:int=0,
  output_file:str="./embeddings.jsonl"
):
  """
  Main function to retrieve and save token embeddings.

  Args:
    model (TEmbeddingModel, optional): The embedding model to use. Defaults to "text-embedding-ada-002".
    encoding (TTokenEncoding, optional): The token encoding to use. Defaults to "cl100k_base".
    request_limit (int, optional): The maximum number of tokens to process in each request. Defaults to 1000.
    start (int, optional): The starting token index. Defaults to 0.
    output_file (str, optional): The path to the output file to save the embeddings. Defaults to "./embeddings.jsonl".
  """
  max_token = get_max_token_value(encoding)
  first_token = start

  while first_token <= max_token:
    end_token = min(first_token + request_limit, max_token + 1)

    token_embeddings = process_token_batch(first_token, end_token, model)

    with open(output_file, 'a') as file:
      for item in token_embeddings:
        json.dump(item, file)
        file.write('\\n')

    first_token = end_token
