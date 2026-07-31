# Agent Tools Repository

This is a code repository for building and managing intelligent agent tools, containing multiple projects and modules related to AI agents.

## Top-Level Directory Structure

| Directory | Description |
|-----------|-------------|
| `packages/` | Python package directory, containing all installable sub-packages |
| `packages/openjiuwentools_core/` | Core package, providing foundational functionality |
| `packages/openjiuwentools_infer_router/` | Inference routing package, providing a KV Cache-aware routing system |
| `openJiuwen-vllm-affinity/` | Integration project between OpenJiuwen and vLLM, supporting affinity scheduling |
| `Agent Innovation Challenge/` | Code and resources related to the Agent Innovation Challenge |

## Package Management Mechanism

This project adopts a modular package management mechanism, where each sub-package is independently built as a wheel package:

| Source directory | Wheel package name |
|------------------|---------------------|
| `packages/openjiuwentools_core` | `openjiuwentools-core-{version}-py3-none-any.whl` |
| `packages/openjiuwentools_infer_router` | `openjiuwentools-infer-router-{version}-py3-none-any.whl` |

### Installation

#### Option 1: Install the meta-package (recommended)

```bash
# Install the core package (minimal dependencies)
pip install openjiuwentools

# Install a sub-package for a specific feature
pip install "openjiuwentools[infer-router]"

# Install all commonly used sub-packages
pip install "openjiuwentools[top]"
```

#### Option 2: Install sub-packages directly

```bash
# Install a specific sub-package individually
pip install openjiuwentools-infer-router
pip install openjiuwentools-core
```

#### Option 3: Development mode install (from source)

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools

# Install with uv (recommended)
uv sync --extra top

# Or install in editable mode with pip
pip install -e ".[top]"
```

## Development Environment Setup

### 1. System Requirements

- Python 3.10 or higher
- Git
- Linux or macOS recommended

### 2. Clone the Repository

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools
```

### 3. Install Dependency Management Tools

It is recommended to manage dependencies with pip or uv:

```bash
# Install uv with pip (optional)
pip install uv
```

### 4. Virtual Environment Setup

Create a unified virtual environment for the entire repository:

```bash
# Create a virtual environment in the repository root
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install project dependencies
pip install -e ".[top]"
# or with uv
uv sync --extra top
```

## pre-commit Configuration

### 1. Install pre-commit

```bash
# Global install (recommended)
pip install pre-commit

# Or install within a virtual environment
pip install -e ".[dev]"
```

### 2. Install Hooks

Run in the repository root:

```bash
pre-commit install
```

### 3. pre-commit Configuration Notes

The repository uses the following pre-commit hooks:

- **ruff**: fast Python code linting and formatting tool
- **codespell**: source code spell checker
- **typos**: another source code spell checker
- **markdownlint-cli**: Markdown file format checker

Hook configuration is in the `.pre-commit-config.yaml` file at the repository root.

### 4. Run pre-commit Manually

```bash
# Check all files
pre-commit run --all-files

# Check specific files
pre-commit run --files file1.py file2.md
```

## Development Notes

### 1. Code Style

- Follow the PEP 8 code style guide
- Use ruff for code formatting and linting
- Keep code concise and clear, with appropriate comments

### 2. Commit Conventions

- Ensure code passes all pre-commit checks before committing
- Commit messages should be clear and accurate, describing the specific changes
- Avoid including unrelated changes in the same commit

### 3. Branch Management

- Use `main` as the stable branch
- Use `feature/feature-name` branches for feature development
- Use `fix/bug-name` branches for bug fixes

### 4. Testing

- Write unit tests for new features
- Ensure all tests pass before committing code
- Run tests with pytest:

  ```bash
  pytest
  ```

### 5. Documentation

- Update relevant documentation to reflect code changes
- Keep API documentation accurate and complete
- Write documentation in Markdown format

## Project-Specific Notes

### openjiuwentools-infer-router package

- Provides LLM inference routing and scheduling
- Supports multiple load balancing algorithms and scheduling strategies
- Includes a complete API service layer and monitoring system
- KV Cache-aware routing, supporting vLLM and SGLang

### openJiuwen-vllm-affinity project

- Integrates OpenJiuwen with vLLM
- Supports affinity-based scheduling strategies
- Optimizes large model inference performance

## Contact

If you have questions or suggestions, please reach out via:

- Project homepage: <https://gitcode.com/openJiuwen/agent-tools>
- Issue tracker: <https://gitcode.com/openJiuwen/agent-tools/issues>
- Discussions: <https://gitcode.com/openJiuwen/agent-tools/discussions>

This product serves solely as a workflow orchestration tool and does not embed any AI model capabilities. When users integrate AI models for specific business scenarios, they shall bear full responsibility for compliance obligations under the EU AI Act and other relevant regulatory frameworks.
