# Tikkun/Tohu

Retrieve embeddings for each token in GPT-4's vocabulary using OpenAI's API.

## Installation

1. Install Poetry, a dependency and package management tool.
2. Clone this repository.
3. Run `poetry install` to install the necessary dependencies.

## Usage

Use the provided CLI to retrieve embeddings:

```bash
poetry run python -m tikkun.cli --help

Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  Retrieve embeddings for each token in GPT-4's vocabulary using OpenAI's
  API.

Options:
  --help  Show this message and exit.
  --api-key TEXT  OpenAI API key (alternatively, set OPENAI_API_KEY environment variable or add it to a .env file)
  --model -m TEXT  Model to use (see: TEmbeddingModel)
  --output-dir -o TEXT  Directory to save embeddings to.
  --batch-size -b INTEGER  Batch size to use when retrieving embeddings.
  --start -s INTEGER  Index to start at.
  --encoding -e TEXT  Tiktoken encoding to use.
  --verbose -v  Enable verbose logging.
```
