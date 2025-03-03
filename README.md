# AI Transparency in Healthcare: Literature Screening with OpenAlex and ASReview

## Project Overview

This project is part of my Master’s thesis at Utrecht University in collaboration with the **Dutch Ministry of Health, Welfare, and Sport. The goal is to build a literature overview on AI transparency in healthcare using AI-driven screening techniques. 

To achieve this, I used OpenAlex metadata from 2019 to 2025 which was vectorised with Jina Embedding v3 model. With these embeddings, I computed the cosine similarity between two target texts and all OpenAlex records. The top 10.000 cosine similarities of both target texts are screened with ASReview and the relevant records are used in the literature overview.

## Repository Structure

- **`cosine_similarity_calculation.py`** - Computes cosine similarity between the target texts and OpenAlex dataset.
- **`find_target_embedding.py`** - Extracts embeddings of the target texts.
- **`take_top.py`** - Filters the most relevant papers based on similarity scores.
- **`cos_sim_target1.parquet` & `cos_sim_target2.parquet`** - Parquet files containing cosine similarity scores for each target text.
- **`target_embedding.parquet`** - Stores vector representations of the target texts.
- **`work_ids.csv`** - CSV file containing IDs of the most relevant OpenAlex records.

## Methodology

1. **Vectorization**  
   - The texts were converted into numerical vector representations using Jina Embeddings v3, a multilingual feature extractor.

2. **Cosine Similarity Calculation**  
   - Cosine similarity was computed between the two target texts and the OpenAlex dataset.  
   - The results were ranked from highest to lowest similarity.

3. **Screening and Selection**  
   - The top 10,000 most similar papers were selected.  
   - These papers were screened with ASReview to refine the literature overview.

4. **Literature Overview Construction**  
   - The final selection of papers forms the basis of the literature review.  
   - This review supports the development of AI transparency frameworks in healthcare.

## Dependencies

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- pyarrow (for handling Parquet files)  
- Jina Embeddings v3 (for text embedding)  

## Installation

Clone the repository and install dependencies.

## Usage

Run the following scripts in order:  

1. Extract embeddings for target texts  
2. Compute cosine similarity  
3. Filter the most relevant papers  

## Research Context

This project supports my thesis on AI transparency in healthcare. By structuring and prioritizing transparency-related literature, it aims to assist policymakers in developing AI governance frameworks.

## License

This project is licensed under the CC-BY license

## Contact

**Author:** Renske Bos  
**Affiliation:** Utrecht University, Ministry of Health, Welfare, and Sport  
**Supervisors:** Prof. dr. A.G.J. van de Schoot, Annette Brons  
**Email:** [renskebos2000@icloud.com]

