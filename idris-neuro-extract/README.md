# Idris Neuro Extract

MVP neuroscience claim extractor for recursive self-healing detection research.

Extracts scientific claims from neuroscience abstracts using a single-prompt LLM architecture. Prompt layer lives in `.md` data files, enabling path-isolated evolution by the self-healing harness.

## Setup

```bash
pip install -r requirements.txt
export LLAMA_CPP_BASE_URL=http://localhost:8080/v1
export LLAMA_CPP_MODEL_ID=qwen3-27b-mtp-6bit
```

## Usage

```python
from extractor.extractor import Extractor

ext = Extractor()
result = ext.extract("12345678", "abstract text here...")
print(result.claims)
```

## Evaluation

```bash
python -m evaluation.runner
```

## License

MIT
