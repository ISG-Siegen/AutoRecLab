# AI4Science - Recommender Systems Research Agent

An autonomous AI agent that uses tree search to iteratively develop, debug, and improve code implementations for recommender systems research tasks.

## Overview

This agent automates the research code development process by:
- Generating multiple initial implementations from your research task description
- Executing and scoring them based on automatically generated requirements
- Using tree search to explore improvements and debug failures
- Iteratively refining code until finding a satisfactory solution

## Quick Start

### Installation & Setup

**Option 1: Docker (Recommended)**

For complete isolation with all dependencies pre-configured:

```bash
# Create environment file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create workspace for datasets
mkdir -p sandbox/workspace
# cp your-dataset.csv sandbox/workspace/

# Build and run
docker compose run --build sandbox

# Inside container, run:
uv run main.py
```

**Option 2: UV**

UV provides the fastest and most reliable dependency management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Set API key and run
export OPENAI_API_KEY="your-key-here"
uv run main.py
```

**Option 3: pip**

```bash
pip install -e .
export OPENAI_API_KEY="your-key-here"
python main.py
```

### Usage

Run the agent and enter your research task description. The prompt supports multi-line input - type `!start` when ready:

```
Enter you request, write "!start" to start:
> Implement a collaborative filtering recommender using matrix factorization
> on the MovieLens dataset. Evaluate with RMSE and MAE metrics.
> !start
```

**What happens next:**
1. The agent generates 3 initial implementations (3 is default, configurable)
2. Each implementation is executed and scored
3. The agent uses tree search to iteratively improve the best solutions
4. Process continues until a satisfactory solution is found or max iterations reached
5. Final results and summary are presented

## Configuration

### Basic Settings

Edit `config.toml` to customize agent behavior:

```toml
[treesearch]
num_draft_nodes = 3      # Number of initial implementations to generate
max_iterations = 10      # Maximum improvement iterations
debug_prob = 0.3         # Probability of debugging vs improving (0.0-1.0)
epsilon = 0.3            # Exploration vs exploitation rate

[exec]
timeout = 3600           # Execution timeout in seconds
workspace = "./workspace"

[agent]
k_fold_validation = 1    # Preferred number of folds for validation (1 = no CV)
data_preview = false     # Tells the LLM to give an overview of the dataset structure

[agent.code]
model = "o5-mini"        # LLM model to use
model_temp = 1.0         # Temperature for code generation
```

### Using Datasets

Place your datasets in the appropriate directory before running the agent:

**Local execution (UV/pip):**
- Directory: `./workspace/`
- Example: `cp movielens.csv ./workspace/`
- Reference in prompt: `"Load data from movielens.csv in the working directory"`

**Docker execution:**
- Directory: `./sandbox/workspace/`
- Example: `cp movielens.csv ./sandbox/workspace/`
- Reference in prompt: `"Load data from movielens.csv in the working directory"`

### Available Libraries

The agent can use these packages (defined in `pyproject.toml`):
- `numpy==1.26.4`, `numba==0.58.1`, `pandas==2.3.2`
- `scipy==1.16.2`, `scikit-learn==1.7.1`, `lenskit==0.14.4`

## Customization

### Adding Custom Libraries

To add new Python packages that the agent can use:

1. **Add dependency:**
   ```bash
   uv add package-name  # with UV
   # or edit pyproject.toml manually
   ```

2. **Update agent awareness** in `treesearch/minimal_agent.py`:
   ```python
   pkgs = [
       "numpy==1.26.4",
       "your-package==2.0.0",  # Add your package here
       # ...
   ]
   ```

3. **Resync environment:**
   ```bash
   uv sync  # for local
   docker compose run --build  # for Docker
   ```

### Modifying Agent Behavior

- **Prompts & code generation:** `treesearch/minimal_agent.py`
- **Tree search strategy:** `treesearch/search.py`
- **Execution settings:** `config.toml`

## How It Works

The agent uses a tree search approach to iteratively improve code:

1. **Task Analysis**: Automatically generates specific code requirements from your research task description
2. **Draft Generation**: Creates multiple initial implementations using LLM (default: 3)
3. **Execution & Scoring**: Runs each implementation and scores based on:
   - Execution success (whether code runs without bugs)
   - Requirement fulfillment (0-100% based on auto-generated requirements)
   - Code quality and conceptual correctness
4. **Tree Search**: Uses epsilon-greedy strategy to select nodes for expansion:
   - Prioritizes debugging buggy implementations
   - Improves working implementations to increase scores
   - Balances exploration of new solutions vs exploitation of best solutions
5. **Termination**: Stops when a satisfactory solution is found (high score, all requirements met) or maximum iterations reached

## Output Files

Results are automatically saved to:
- `./workspace/` (local) or `./sandbox/` (Docker) - Execution workspace with generated code and results
- `./out/save.pkl` - Pickled tree search state (all nodes and their scores)
- `./out/code_requirements.json` - Auto-generated code requirements for the task

## Requirements

- Python ≥3.11
- OpenAI API key (**Note:** Anthropic/Claude API not yet implemented)
- UV (recommended), pip, or Docker
- Packages listed in `pyproject.toml`

## Project Structure

```
├── main.py              # Entry point
├── config.toml          # Configuration file
├── pyproject.toml       # Dependencies
├── workspace/           # Local execution workspace
├── sandbox/workspace/   # Docker execution workspace
├── treesearch/
│   ├── minimal_agent.py # Agent logic and prompts
│   ├── search.py        # Tree search algorithm
│   ├── node.py          # Node representation
│   └── backend/         # LLM interfaces
└── utils/               # Logging utilities
```
