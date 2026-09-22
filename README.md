# linguistics-fact-checking
Repository for the paper "The Linguist’s Lie Detector: Benchmarking Linguistic Veracity in Large Language Models" (https://aclanthology.org/2026.nslp-1.23/)

The goal of this project is to build an AI-assisted pipeline for Linguistic Fact-Checking. We aim to automatically extract linguistic claims (theoretical, typological, specific language facts) from academic articles, verify their accuracy, and evaluate the robustness of Large Language Models (LLMs) in acting as "Linguist" agents.

## Pipeline

The project is organised into three main analytical stages:

1. Data Ingestion & Extraction: src/run_segmentatio.py and src/run_extraction.py
   1.1. Segmentation: Splits raw article sections into clear, atomic sentences.
   1.2. Extraction: Analyzes each sentence to extract claims.
2. Classification: src/run_classification.py
3. Negation: src/run_negation.py
4. Knowledge Verification ("The LLM Linguist"): src/run_verification.py
   - This script tests the domain knowledge of the LLM.
   
## Repository Structure

1. **annotation_work**: guidelines and annotations for the statement extraction and statement classification tasks.
2. **src**: scripts to run the pipeline.
3. **glossaDatasetDownload.py**: script to download the XML files from the Glossa journal.
4. **pipeline_notebook.ipynb**: notebook with the pipeline as steps.
5. **input_data**:
   - *gold_statements.csv*: 342 human-curated gold statements extracted from the 63 paragraphs.
6. **output_data**: results for the complete subset of 11 articles selected in the paper
   - statements.csv
   - clasified_statements.csv
9. **paper_results**: results from the experiments.
   - *linguistic_statements.csv*: 235 human-curated gold linguistic statements + two LLM-generated false variants using GPT-5.2 and Gemini 2.5 Flash.
10. **evaluation_paper.ipynb**: evaluation of the results for the paper.
