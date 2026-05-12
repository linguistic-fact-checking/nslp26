# linguistics-fact-checking
Repository for the paper "The Linguist’s Lie Detector: Benchmarking Linguistic Veracity in Large Language Models"


The goal of this project is to build an AI-assisted pipeline for Linguistic Fact-Checking. We aim to automatically extract linguistic claims (theoretical, typological, specific language facts) from academic articles, verify their accuracy, and evaluate the robustness of Large Language Models (LLMs) in acting as "Linguist" agents.

## Repository Structure
The project is organised into three main analytical stages:

1. Data Ingestion & Extraction: src/run_segmentatio.py and src/run_extraction.py
   1.1. Segmentation: Splits raw article sections into clear, atomic sentences.
   1.2. Extraction: Analyzes each sentence to extract claims.
2. Classification: src/run_classification.py
3. Negation: src/run_negation.py
4. Knowledge Verification ("The LLM Linguist"): src/run_verification.py
   This script tests the domain knowledge of the LLM.
