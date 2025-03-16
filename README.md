# ML-Based RNA-Seq Classification for Autoimmune Disease Biomarker Discovery

## Executive Summary
We developed a machine learning pipeline for classifying RNA-Seq gene expression data into three categories: Healthy, Rheumatoid Arthritis (RA), and Systemic Lupus Erythematosus (SLE). Our best-performing model, SGDClassifier with Elastic Net regularization, achieved 93.30% accuracy, outperforming more complex non-linear approaches. Key genes identified as potential biomarkers include ANP32D, IFI27, and MIR378F, offering insights into molecular mechanisms underlying these autoimmune conditions.

## Introduction
This project aims to:
1. Classify individuals as healthy or having RA or SLE based on RNA-Seq gene expression profiles
2. Identify key genes that could serve as potential biomarkers for these conditions

RNA-Seq quantifies mRNA molecules in biological samples, reflecting gene expression levels and offering insights into disease mechanisms.

## Methodology

### Data Preprocessing
- **Feature Alignment**: Identified common genes across all datasets to create a consistent feature space
- **Transformation**: Applied log2(x+1) transformation to stabilize variance and reduce skewness
- **Class Imbalance**: Addressed imbalanced class distribution using weighted learning approaches

### Model Development

#### Linear Approach
- **SGDClassifier**: Tested multiple regularization strategies:
  - No penalty (baseline)
  - L1 penalty (feature selection)
  - L2 penalty (coefficient shrinkage)
  - Elastic Net penalty (combined L1 and L2)
- **Evaluation**: 5-fold stratified cross-validation with class weighting

#### Non-Linear Approach
- **XGBoost**: Implemented to capture potential non-linear relationships
- **Feature Selection**: Tested model with all features vs. top 500 genes
- **Interpretability**: Used SHAP values to identify gene contributions

## Results

### Model Performance

| **Model**                     | **Accuracy** | **Precision (Macro)** | **Recall (Macro)** | **F1-Score (Macro)** |
|-------------------------------|--------------|-----------------------|--------------------|----------------------|
| **SGDClassifier (Elastic Net)**| **93.30%**  | **0.89**              | **0.96**           | **0.92**             |
| **XGBoost (All Features)**    | 74.75%       | 0.86                  | 0.84               | 0.81                 |
| **XGBoost (Top 500 Features)**| 75.42%       | 0.87                  | 0.84               | 0.81                 |

The Elastic Net model demonstrated superior performance across all metrics, with particularly strong class-specific results:

| **Class**   | **Precision** | **Recall** | **F1-Score** |
|-------------|---------------|------------|--------------|
| **Healthy** | 0.76          | 1.00       | 0.87         |
| **RA**      | 0.92          | 1.00       | 0.96         |
| **SLE**     | 1.00          | 0.87       | 0.93         |

### Biomarker Discovery

#### Key Genes by Condition

**Healthy Biomarkers**:
- **Upregulated**: TRIM69, GTF2IP12, DES, SMPD5, EME2
- **Downregulated**: PINK1-AS, SNX32, GTF2IP1, HPR, CHRNB2

**RA Biomarkers**:
- **Upregulated**: TPI1P3, MIR378F, SLC23A3, CXCL9, AGMO
- **Downregulated**: ANP32D, TCP11X2, TCP11X1, CPEB1, ACADL

**SLE Biomarkers**:
- **Upregulated**: ANP32D, IFI27, SLC25A18, MIR6724-1, SLCO5A1
- **Downregulated**: AGMO, IGLV3-30, MIR378F, CLRN1, SLC26A3

**Notable Findings**:
- **ANP32D**: Strongly downregulated in RA, upregulated in SLE
- **MIR378F**: Upregulated in RA, downregulated in SLE
- **IFI27**: Significantly upregulated in SLE, potential key biomarker

## Analysis & Insights

### Model Performance Analysis
1. **Linear vs. Non-Linear**: Despite XGBoost's theoretical advantage in capturing complex relationships, the linear Elastic Net model performed substantially better, suggesting:
   - Linear relationships may sufficiently capture gene expression patterns for classification
   - Regularization is more critical than model complexity for high-dimensional, sparse RNA-Seq data

2. **Regularization Benefits**: Elastic Net's combined L1/L2 approach effectively:
   - Managed the high feature-to-sample ratio
   - Reduced overfitting through sparse feature selection
   - Handled multicollinearity among gene expression levels

### Challenges & Limitations
- **Dimensionality**: High gene count relative to sample size presents statistical challenges
- **Data Sparsity**: Zero-inflated gene expression data requires specialized handling
- **Disease Similarity**: RA and SLE share autoimmune mechanisms, complicating differentiation
- **Validation Need**: Computational biomarkers require experimental validation

## Conclusions & Future Directions

### Key Takeaways
- **Model Selection**: For RNA-Seq classification, regularized linear models can outperform complex non-linear alternatives
- **Potential Biomarkers**: ANP32D, IFI27, and MIR378F warrant further investigation as disease biomarkers
- **Disease Differentiation**: Distinct gene expression patterns can effectively differentiate between RA and SLE

### Future Work
- **External Validation**: Validate identified biomarkers in independent cohorts
- **Biological Pathway Analysis**: Explore biological significance of key genes
- **Feature Engineering**: Incorporate prior biological knowledge into model development
- **Advanced Models**: Test deep learning approaches with regulatory prior information
- **Multi-omics Integration**: Combine RNA-Seq with other data types (DNA methylation, proteomics) for improved classification
