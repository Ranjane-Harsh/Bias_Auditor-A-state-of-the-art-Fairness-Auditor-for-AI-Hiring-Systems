# Bias Auditor: A State-of-the-Art Fairness Auditor for AI Hiring Systems

## 📌 Overview
**Bias Auditor** is a comprehensive, end-to-end framework designed to detect, analyze, and mitigate algorithmic bias in AI-driven hiring systems. By systematically evaluating machine learning models across sensitive attributes (such as gender, race, education level, and college tier), this tool ensures that automated recruitment processes remain fair, transparent, and equitable.

## ✨ Key Features
* **Automated Data Processing**: Robust pipelines for data cleaning, loading, and preprocessing.
* **Flexible Model Training**: Built-in support for training various ML models (Logistic Regression, Random Forest, XGBoost) using YAML-based configurations.
* **Comprehensive Bias Detection**: Evaluates model fairness using industry-standard metrics, including:
  * Disparate Impact Ratio (DIR)
  * Selection Rate
  * False Positive Rate (FPR) Parity
  * True Positive Rate (TPR) Parity
* **State-of-the-Art Bias Mitigation**:
  * *Pre-processing*: Reweighting techniques to balance representation in the training data.
  * *Post-processing*: Equalized odds adjustments to ensure fair decision boundaries.
* **Model Interpretability**: Integrates SHAP (SHapley Additive exPlanations) to provide transparent insights into feature importance and model decision-making.
* **Automated Reporting**: Generates detailed visualizations (`.png`) and statistical summaries (`.csv`) for bias metrics across all protected groups.

## 📂 Repository Structure

The project is modularized into distinct functional components:

```text
├── Bias_Detection/                        # Core logic for calculating fairness metrics
│   ├── bias_metrices.py
│   └── bias_reporter.py
├── Bias_Mitigation/                       # Algorithms to reduce detected bias
│   ├── postprocessor_equalized.py
│   └── preprocessor_reweighting.py
├── Configs/                               # YAML configuration files for models and mitigation
│   ├── logistic_regression.yaml
│   ├── mitigation_config.yaml
│   ├── random_forest.yaml
│   └── xgboost.yaml
├── Data_Acquisation_and_preprocessing/    # Data pipelines
│   ├── data_cleaner.py
│   ├── data_loader.py
│   └── data_preprocessor.py
├── Dataset/                               # Synthetic and raw hiring datasets
├── Interpretability/                      # SHAP integration for explainable AI
│   └── shap_interpretability.py
├── Model_training_and_Validation/         # Model training, evaluation, and logging
│   ├── evaluator.py
│   ├── model_config.py
│   └── model_trainer.py
├── Models/                                # Saved serialized models (e.g., .joblib)
├── Reports/                               # Generated bias reports, charts, and CSV summaries
├── Utils/                                 # Project documentation and context
└── main.py                                # Primary execution entry point
