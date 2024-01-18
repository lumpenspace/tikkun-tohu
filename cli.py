import click
import os
import sys
from src.tikkuntohu_types import TEmbeddingModel, TTokenEncoding
from src.get_embeddings import fetch_embeddings

def set_env_variables(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

@click.group()
@click.option('--api-key', type=str, help='OpenAI API key.')
@click.option('--model', '-m', default='text-embedding-ada-002', type=TEmbeddingModel, help='Model to use.')
@click.option('--output-dir', '-o', default='./embeddings', help='Directory to save embeddings to.')
@click.option('--batch-size', '-b', default=1000, type=int, help='Batch size for embedding requests.')
@click.option('--start', '-s', default=0, type=int, help='Start index for token embeddings.')
@click.option('--encoding', '-e', default='cl100k_base', type=TTokenEncoding, help='Encoding for tokenization.')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output.')
@click.pass_context
def cli(ctx, api_key, model, output_file, batch_size, start, encoding, verbose):
    """Retrieve embeddings for each token in GPT-4's vocabulary."""
    ctx.ensure_object(dict)
    if api_key:
        set_env_variables(api_key)
    elif 'OPENAI_API_KEY' not in os.environ:
        click.echo("API key is required. Set it via --api-key or OPENAI_API_KEY environment variable.", err=True)
        sys.exit(1)

    # pass api key to ENV as OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = api_key
    
    ctx.obj['MODEL'] = model
    ctx.obj['OUTPUT_FILE'] = output_file
    ctx.obj['BATCH_SIZE'] = batch_size
    ctx.obj['START_INDEX'] = start
    ctx.obj['ENCODING'] = encoding

@cli.command()
@click.pass_context
def fetch(ctx):
    """Fetch and save embeddings."""
    fetch_embeddings(
      model=ctx.obj['MODEL'],
      output_file=ctx.obj['OUTPUT_FILE'],
      request_limit=ctx.obj['BATCH_SIZE'],
      start=ctx.obj['START_INDEX'],
      encoding=ctx.obj['ENCODING'],
    )

if __name__ == '__main__':
    cli(obj={})
