# nlplib

Internal library of reusable components for NLP projects: data preparation, LLM inference and training, quality metrics, adaptation, and a set of helper utilities (text processing, morphology, NER).

## Preparing
Clone repo 
```bash
git clone https://github.com/ilyagvozdarev/nlplib.git
cd nlplib
```

Install package
```bash
pip install -e .
```

## Package Structure
```
|- adapt
   |- lep
|- dataset
|- infer
|- metrics
|- training
   |- trainers
|- utils
   |- asr
   |- ner
   |- processing
```