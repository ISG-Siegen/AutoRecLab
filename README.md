> [!CAUTION]
> This is the active **development branch** (`develop`) of AutoRecLab.
>
> - Features can change at any time.
> - Interfaces and prompts may be unstable.
> - Experimental behavior is expected.
>
> For the latest stable release, use the [`main`](../../tree/main) branch.

# AutoRecLab

AutoRecLab is an autonomous research agent for recommender-systems experimentation. It turns natural-language research tasks into executable code, evaluates the results, and iteratively improves solutions through tree search.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Usage](#setup-and-usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Contributing](#contributing)

## Introduction

AutoRecLab helps researchers move from a free-form experiment idea to runnable Python, intermediate artifacts, and a final report. It combines LLM-based planning with iterative execution, evaluation, and refinement loops tailored to recommender-systems workflows.

Core capabilities:

- Natural-language to executable experiment generation
- Iterative code improvement through tree search
- Built-in execution, scoring, debugging, and refinement loops
- Documentation-aware retrieval for OmniRec, LensKit, and RecBole via FAISS indices
- Configurable runtime behavior through `config.toml` and environment variables
- Optional type checking before execution for more reliable generated code

For a fuller walkthrough of setup, architecture, usage examples, and FAQ, see [docs/README.md](docs/README.md).

## Setup and Usage

### Requirements

- [Python](https://www.python.org/) >= 3.12
- [Graphviz (`dot`)](https://graphviz.org/) available on `PATH`
- [OpenAI API](https://openai.com/api/) key
- [`uv`](https://docs.astral.sh/uv/) for the recommended local workflow
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) for an isolated workflow with the runtime dependencies already installed

If you run AutoRecLab locally, you need to provide the Python environment and system requirements yourself, including Graphviz. The Docker workflow is often the easier starting point because the container already includes the project dependencies and tools such as `dot`.

If you need a minimal fallback, `pip install -e .` also works, but the repository is maintained primarily around `uv`.

### Environment

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your-openai-api-key
```

Docker Compose reads this file automatically. For local runs, `uv run` also picks it up from the project root.

### Documentation Embeddings

AutoRecLab uses FAISS-based documentation indices in `ragEmbeddings/` to ground code generation and framework lookups for OmniRec, LensKit, and RecBole.

Generate the embeddings locally with:

```bash
uv run python -m cli.embeddings.main generate --all
```

Generate only selected sources if needed:

```bash
uv run python -m cli.embeddings.main generate --omnirec --lenskit
```

Useful flags include `--omnirec`, `--lenskit`, `--recbole`, `--force`, and `--out`.

When you run the Docker workflow, the container entrypoint generates embeddings on startup.

If you do not want to generate the embeddings yourself, you can download pre-generated files from [this OneDrive folder](https://1drv.ms/f/c/04375478f480c0d7/IgCMRP8a-P6yTKIUflpDpq0zAQGu9UPOn0rt_0VydNMC0iY?e=MyC9Yz) and place the `lenskit`, `omnirec`, and `recbole` folders under `ragEmbeddings/`.

### Run with Docker

Use Docker when you want an isolated runtime that already has the project dependencies and required tools installed:

```bash
docker compose run --build sandbox
```

Compose reads `.env` automatically. By default, artifacts from Docker runs are written to `./sandbox` on the host and mounted to `/app/out` in the container. This workflow is useful when you want to avoid local setup friction, since the container already includes dependencies such as Graphviz `dot`.

Once the container shell is open, start AutoRecLab with:

```bash
uv run main.py
```

### Run Locally with uv

Install the project environment:

```bash
uv sync
```

Then start AutoRecLab:

```bash
uv run main.py
```

Example interactive prompt:

```text
Enter you request, write "!start" to start:
> Build a reproducible top-N recommendation experiment on MovieLens.
> Compare a popularity baseline and a matrix-factorization approach.
> Report Recall@10 and NDCG@10.
> !start
```

### Basic CLI Usage

Common commands include:

- Inline prompt: `uv run main.py --prompt "Analyze the signal and generate a report"`
- Prompt from file: `uv run main.py --prompt-file ./my-prompt.txt`
- Initialize the output workspace: `uv run main.py --init`
- List available datasets: `uv run main.py --list-datasets`
- List available models: `uv run main.py --list-models`
- Override the model for one run: `uv run main.py --model "gpt-4o"`
- Append a timestamp to the output directory: `uv run main.py --timestamp-out-dir`

For more examples and workflows, see [docs/usage-and-examples.md](docs/usage-and-examples.md).

## Configuration

`config.toml` is the main configuration file for AutoRecLab. The checked-in file defines the repository's current runtime defaults. If `config.toml` is missing, AutoRecLab writes a fresh one using its built-in defaults before starting.

The most commonly adjusted settings look like this:

```toml
out_dir = "./out"

[treesearch]
num_draft_nodes = 2
max_iterations = 5
refinement_iterations = 2

[exec]
timeout = 5400
enable_type_checking = true
keep_only_relevant_files = false

[agent.code]
model = "gpt-5.4-mini"
```

You can also override settings with environment variables using the `ARL_` prefix and `__` for nesting:

- `ARL_out_dir=./sandbox`
- `ARL_treesearch__max_iterations=8`
- `ARL_agent__code__model=gpt-5.4-mini`

Set the log level with `ISGSA_LOG=DEBUG|INFO|WARNING|ERROR`.

If disk space matters, `keep_only_relevant_files=true` reduces saved artifacts by keeping only the generated code and final results instead of every intermediate file.

## Outputs

By default, local runs write artifacts to `./out`. Docker runs write to `./sandbox` on the host because the container maps `/app/out` there. If you change `out_dir` or use `--timestamp-out-dir`, the same structure is created in the configured output folder.

Common artifacts include:

- `summary.md` for the final run summary
- `save.pkl` for the serialized tree state
- `tree_render/` for rendered search-tree visualizations
- `checkpoint/` for per-node code and execution artifacts
- `statistics/` for aggregated run statistics and plots
- `costs_log.csv` for model cost tracking
- `entered_prompt.txt` for the logged user prompt unless `--prompt-no-log` is used
- `workspace/` for execution-time working files

To render a saved tree again manually, run:

```bash
uv run viz.py -i ./out/save.pkl -o ./out/tree_render
```

## Contributing

Set up the development environment with:

```bash
uv sync
uv run pre-commit install
```

Contributors should keep `pre-commit` installed so the hooks run automatically before commits. If you activate the local virtual environment, the same install command is also available as `pre-commit install`.

The installed hooks currently cover Ruff linting and auto-fixes, Ruff formatting, and basic repository hygiene checks such as debug statements, trailing whitespace, YAML/TOML validation, and large added files.

For code changes, we also recommend running `uv run pytest` before opening a PR or after larger changes.
