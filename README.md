# CoDMC
**CONFIGURATION**

Backbone: CNN-BiGRU
Treatments: SMOTETomek + Focal Loss, No Resampling + Cross-Entropy (2 treatments)
Meta-learner: Logistic Regression
Datasets: SOR, OTDR, HardFailure, SoftFailure

**HOW TO RUN**

Execute all FIVE notebooks sequentially, from 01 to 05. For each notebook, select Run All and execute all cells from top to bottom without skipping any cells.

All outputs are automatically saved to a new directory, results_clean/, including tables, figures, models, and predictions. The original results/ directory will remain unchanged and will not be overwritten.

01_eda_4_dataset.ipynb

Exploratory data analysis: class distribution, imbalance ratio, correlation analysis, t-SNE visualization, and outlier detection.

02_eksperimen_hdmc.ipynb

Train the CNN-BiGRU backbone using the final 10-fold cross-validation protocol with two imbalance treatments.

03_baseline_dan_eksternal.ipynb

Train 10 comparison models (5 external models and 5 baseline models) using the same experimental protocol as HDMC.

04_visualisasi.ipynb

Generate all visualizations, including confusion matrices, learning curves, calibration plots, per-class F1-scores, computational efficiency analysis, heatmaps, and architecture diagrams.

05_hasil_akhir_dan_konsolidasi.ipynb

Consolidate the experimental results into HASIL_KONSOLIDASI_HDMC_vs_PEMBANDING.xlsx, including performance summaries, statistical significance tests against 10 comparison models, meta-learner ablation studies, and backbone comparisons.
# Dataset Availability and Reproducibility

This directory documents the four datasets used to evaluate the **HDMC (Hybrid Deep Learning Meta-Classification)** framework for optical fiber fault detection. The framework combines a CNN–BiGRU backbone trained using two complementary treatments—SMOTETomek with Focal Loss and no resampling with Cross-Entropy—and a Logistic Regression meta-learner.

## Dataset overview

| Dataset | Origin | Availability in this repository | Official source |
|---|---|---|---|
| **SOR** | Collected by the study authors using an optical fiber laboratory testbed | **Author-owned data will be provided here**, subject to applicable institutional permissions | This repository (`dataset/`) |
| **OTDR** | Abdelli et al. | **Not redistributed**; access to the original files has not been confirmed | [IEEE DataPort dataset record](https://doi.org/10.21227/PDPN-1B78) |
| **HardFailure** | ARNO testbed, InRete Lab, Scuola Superiore Sant'Anna | **Not redistributed**; obtain from the original repository | [HardFailure_dataset.csv](https://github.com/Network-And-Services/optical-failure-dataset/blob/main/HardFailure_dataset.csv) |
| **SoftFailure** | ARNO testbed, InRete Lab, Scuola Superiore Sant'Anna | **Not redistributed**; obtain from the original repository | [SoftFailure_dataset.csv](https://github.com/Network-And-Services/optical-failure-dataset/blob/main/SoftFailure_dataset.csv) |

## 1. SOR dataset (collected by the authors)

The SOR dataset was collected using a laboratory optical fiber testbed and Optical Time-Domain Reflectometer (OTDR) measurements. It includes normal observations and multiple types of optical fiber fault events. The data are intended to support reproducibility of the HDMC experiments.

**Availability:** The authors plan to share their own SOR data in this repository. Please consult the files in this directory for the actual dataset, its format, and any applicable terms of use.

**Important:** The final data filename, directory layout, schema, and data-preparation instructions must be documented once the SOR files are uploaded. Do not assume that the notebooks can ingest the raw files without preprocessing.

## 2. OTDR dataset (Abdelli et al.)

This is a third-party optical fiber fault dataset associated with the work of Abdelli and collaborators.

- **Original dataset record:** [*Dataset for Optical fiber faults*, IEEE DataPort (2022)](https://doi.org/10.21227/PDPN-1B78)
- **Related publication by Abdelli et al.:** [*Optical Fiber Fault Detection and Localization in a Noisy OTDR Trace Based on Denoising Convolutional Autoencoder and Bidirectional Long Short-Term Memory*](https://arxiv.org/abs/2203.12604)
- **Bibliographic record:** [DBLP — Dataset for Optical fiber faults](https://dblp.org/rec/data/10/AbdelliATGPC22)

**Availability notice:** The original data are **not** mirrored or redistributed here. Access to the original IEEE DataPort files could not be confirmed when this README was prepared. The linked paper provides scientific context, **not a replacement for the underlying dataset**. Researchers who need the exact data should consult the original data record or contact the dataset owners.

**Dataset identification:** Abdelli and colleagues published more than one OTDR-related dataset. This repository points to *Dataset for Optical fiber faults* (DOI: `10.21227/PDPN-1B78`); users should check that this matches the dataset version used in the experiment before reproducing reported results.

## 3. HardFailure dataset (ARNO testbed)

The HardFailure dataset was collected using the ARNO optical networking testbed at the InRete Lab, TeCIP Institute, Scuola Superiore Sant'Anna. The original dataset creators emulated hard failures and recorded optical network monitoring metrics.

- **Official dataset repository:** https://github.com/Network-And-Services/optical-failure-dataset
- **HardFailure CSV file:** https://github.com/Network-And-Services/optical-failure-dataset/blob/main/HardFailure_dataset.csv
- **Related publication:** [*Learning Long-and Short-Term Temporal Patterns for ML-driven Fault Management in Optical Communication Networks*](https://doi.org/10.1109/TNSM.2022.3146869)

Download the file from the original repository and follow the repository's documentation and applicable usage conditions. The dataset is not redistributed here.

## 4. SoftFailure dataset (ARNO testbed)

The SoftFailure dataset comes from the same ARNO optical networking testbed. The source repository describes simulated soft-failure events and monitoring data acquired under changing optical attenuation.

- **Official dataset repository:** https://github.com/Network-And-Services/optical-failure-dataset
- **SoftFailure CSV file:** https://github.com/Network-And-Services/optical-failure-dataset/blob/main/SoftFailure_dataset.csv

Download the file from the original repository and follow the original provider's documentation and applicable usage conditions. The dataset is not redistributed here.

