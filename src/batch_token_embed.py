import src.openai as openai
import tiktoken

def get_embeddings(token_batch, model):
    response = openai.Embedding.create(input=token_batch, model=model)
    return response['data']

def process_token_batch(start_token, end_token, model, encoding="cl100k_base"):
    token_batch = [[i] for i in range(start_token, end_token)]
    embeddings_data = get_embeddings(token_batch, model)
    token_embeddings = []

    for idx, data in enumerate(embeddings_data):
        token_id = start_token + idx
        string_value = tiktoken.decode([token_id], encoding=encoding)
        embedding = data['embedding']
        token_embeddings.append([token_id, string_value, embedding])
    
    return token_embeddings
