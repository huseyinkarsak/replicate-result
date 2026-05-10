# Replication of SAX-VSM Algorithm on OSULeaf Dataset

This project is an independent replication and critical analysis of the **SAX-VSM** (Symbolic Aggregate approximation - Vector Space Model) algorithm, as proposed by **Senin and Malinchik (2013)**. The study focuses on the classification and interpretability of the **OSULeaf** time series dataset.

## Project Information
* **Author:** Hüseyin Karsak
* **Student ID:** 2019402078
* **Course:** IE 48B: Special Topics in Time Series Analytics
* **Institution:** Boğaziçi University, Department of Industrial Engineering

## Project Overview
This study replicates the SAX-VSM classifier, which combines Symbolic Aggregate approximation (SAX) with the Vector Space Model (tf-idf) to enable interpretable time series classification. While the original paper reported an error rate of **0.107** for the OSULeaf dataset, this replication achieved an error rate of **0.161** using Grid Search optimization. The project specifically demonstrates how the algorithm captures distinctive botanical features (such as leaf serrations and tip structures) as class-defining patterns.

## Repository Structure
* `replication_report.qmd`: The primary Quarto document containing the technical analysis, implementation code, and results.
* `requirements.txt`: List of Python dependencies required to run the analysis.
* `data/`: Directory containing the dataset files (`OSULeaf_TRAIN.ts` and `OSULeaf_TEST.ts`).

## Installation and Usage

### 1. Install Dependencies
Open your terminal in the project directory and run the following command:
```bash
   pip install -r requirements.txt
```
### 2. Render the Report
To generate the final HTML analysis report, ensure you have Quarto installed and run:
```bash
   quarto render replication_report.qmd
```

### Key Findings

**interpretability:** The algorithm effectively identifies and highlights characteristic botanical features for the Acer Circinatum, Acer Glabrum, and Quercus Garryana classes.Accuracy vs. 

**Interpretability Trade-off:** While a window size of W=80 yielded the highest statistical accuracy, a smaller window size of W=60 provided superior visual alignment with physical botanical traits.

**Performance Analysis:** The discrepancy between the replicated and original error rates is attributed to the high sensitivity of SAX parameters and the use of a discrete Grid Search instead of the original's DIRECT global optimization algorithm.

### References
Senin, P., & Malinchik, S. (2013). SAX-VSM: Interpretable Time Series Classification Using SAX and Vector Space Model. 2013 IEEE 13th International Conference on Data Mining, 1175-1180.