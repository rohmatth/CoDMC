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
