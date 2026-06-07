# Ground Truth Review Notes

## Header

- **Date**: 2026-05-28
- **Reviewer**: LLM (self-healing study annotator)
- **Sample size**: 30 abstracts from ground_truth.jsonl (200 total)
- **Sample selection**: First 30 entries in file order (lines 1-30)

## Overall Quality Assessment

The non-empty annotations in ground_truth.jsonl are high quality: claims are granular, distinct, and faithfully derived from source text with no systematic errors. The 20 empty abstracts (10% of corpus) represent a recall gap at the extraction stage that should be addressed before the corpus is used as ground truth for the self-healing study.
