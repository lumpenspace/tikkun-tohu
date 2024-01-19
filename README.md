# Tikkun/Tohu

Retrieve embeddings for each token in GPT-4's vocabulary using OpenAI's API.

## Installation

1. Install Poetry, a dependency and package management tool.
2. Clone this repository.
3. Run `poetry install` to install the necessary dependencies.

## Usage

Use the provided CLI to retrieve embeddings:

```bash
poetry run tiktoh --help

  Usage: tiktoh [OPTIONS] COMMAND [ARGS]...

    Retrieve embeddings for each token in GPT-4's vocabulary.

  Options:
    --api-key TEXT                 OpenAI API key.
    -m, --model TEMBEDDINGMODEL    Model to use.
    -o, --output-dir TEXT          Directory to save embeddings to.
    -b, --batch-size INTEGER       Batch size for embedding requests.
    -s, --start INTEGER            Start index for token embeddings.
    -e, --encoding TTOKENENCODING  Encoding for tokenization.
    -v, --verbose                  Enable verbose output.
    --help                         Show this message and exit.

  Commands:
    fetch  Fetch and save embeddings.
```
