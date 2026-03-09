# Example RAG Implementation using the OpenAI API

## Getting Started

### Software Environment Prep
1. Install Homebrew
2. Install conda and use pyenv
3. Install python dependencies per `requirements.txt`

### Folder Structure
```
example-rag/
  README.md
  .env --note you have to create this--
  requirements.txt
  data/ --note you have to create this folder and subdirectories--
    evals/              # eval set goes here
    raw/                # put PDFs here
    processed/          # chunked jsonl
    index/              # vector db index + metadata
  src/
    ingest.py
    build_index.py
    query.py
    rag/
      __init__.py
      io.py
      chunking.py
      embeddings.py
      vectordb.py
      prompts.py
      evals.py
      guardrails.py
```

### Data Loading, Chunking, and Indexing
1. Add raw data to `data/raw` folder
2. run `python src/ingest.py` to chunk the raw data
3. run `python src/build_index.py` to embed the chunks into a vector database

### Using the RAG System
1. run `python src/query.py` to start the interpreter to ask questions of your data

## Running Evals

### Create the initial eval template
1. Create an evaluation template using in the bash terminal:
```
python - <<'PY'
from pathlib import Path
from src.rag.evals import make_template_eval_set
make_template_eval_set(Path("data/evals/eval_set.jsonl"))
PY
```

### Run Evals
1. Run evals and generate metrics using:
```
python - <<'PY'
from pathlib import Path
from src.rag.evals import run_evals
summary = run_evals(
    eval_path=Path("data/evals/eval_set.jsonl"),
    out_path=Path("data/evals/results.jsonl"),
    k=6,
    do_groundedness_judge=True,
)
print(summary)
PY
```

### Review Evals
Metrics returned in the results object from running evals:

#### Test Set Metrics
- **n**: Number of evaluation questions in the eval set.

#### Retrieval Metrics
The following evaluates retrieval quality
- **recall@1**: Value between 0 and 1. Measures how often a gold chunk appears as the top retrieved chunk.
- **recall@3**: Value between 0 and 1. Measures how often a gold chunk appears within the top 3 retrieved chunks.
- **recall@5**: Value between 0 and 1. Measures how often a gold chunk appears within the top 5 retrieved chunks.
- **recall@10**: Value between 0 and 1. Measures how often a gold chunk appears within the top 10 retrieved chunks.
- **MRR (Mean Reciprocal Rank)**: Value between 0 and 1 that measures how early the first correct chunk appears in the retrieval ranking. Higher values indicate that relevant chunks are retrieved closer to the top.

#### Generation Metrics
The following evaluates whether the system produces faithful answers
- **grounded_rate**: Value between 0 and 1 measuring the proportion of answers that are fully supported by the retrieved sources, based on an automated groundedness judge.

