from typing import Literal

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