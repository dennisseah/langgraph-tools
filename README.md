# langgraph-tools
trying out tools in langgraph

## Setup

1. Clone the repository
2. `cd langgraph-tools` (root directory of this git repository)
3. `python -m venv .venv`
4. `poetry install` (install the dependencies)
5. code . (open the project in vscode)
6. install the recommended extensions (cmd + shift + p -> `Extensions: Show Recommended Extensions`)
7. `pre-commit install` (install the pre-commit hooks)
8. copy the `.env.sample` file to `.env` and fill in the values

## Samples

`-s --summarize` perform summarization

`-e --extract-entities` extract entities from the text

`-c --count-words` count the number of words in the text

### Using LangGraph

```sh
python -m langgraph_tools.executors.graph_executor -f test1.txt -s -e -c
```

```sh
python -m langgraph_tools.executors.graph_executor -f test2.txt -s -e -c
```

### Using LangChain Agent Executor

```sh
python -m langgraph_tools.executors.chain_executor -f test1.txt -s -e -c
```

```sh
python -m langgraph_tools.executors.chain_executor -f test2.txt -s -e -c
```


## Unit Test Coverage

```sh
python -m pytest -p no:warnings --cov-report term-missing --cov=langgraph_tools tests
```
