import numpy as np
from Data_Acquisation_and_preprocessing.data_preprocessor import get_data,load_and_preprocess_data
from Data_Acquisation_and_preprocessing.data_loader import log_status,extract_sensitive_columns
from Model_training_and_Validation.model_config import load_model_config,initialize_model 
from Model_training_and_Validation.model_trainer import train_model,generate_predictions,save_model
from Model_training_and_Validation.evaluator import evaluate_performance,evaluate_fairness
from Bias_Detection.bias_metrices import compute_all_metrices
from Bias_Detection.bias_reporter import run_bias_report
from Bias_Mitigation.preprocessor_reweighting import load_mitigation_config, reweighing
from Bias_Mitigation.postprocessor_equalized import extract_positive_scores, compute_global_tpr_fpr, run_equalized_odds_postprocessing
from Interpretability.shap_interpretability import run_interpretability_pipeline


def run_pipeline():
    training_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\generated_bias_auditor_dataset.csv"
    testing_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\test_synthetic_ai_hiring_dataset_v2.csv"
    output_dir = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Reports"
    #Loading, preprocessing and spliting training dataset
    log_status("INFO","Processing Training Dataset : ")
    X_train,y_train = get_data(training_dataset)
    print("\n")

    #Loading, preprocessing and spliting testing dataset
    log_status("INFO","Processing Testing Dataset")
    X_test, y_test = get_data(testing_dataset)
       
    #Loading configuration for training models
    config_dict = load_model_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\random_forest.yaml")
    model_instance = initialize_model(config_dict)

    #Training and Generating Predictions
    trained_model = train_model(model_instance,X_train,y_train, None)
    y_pred,y_proba = generate_predictions(trained_model,X_test)
    #save_model(trained_model,"Random_forest",r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Models")

    #Evaluating Bias Metrices
    metrices = evaluate_performance(y_test,y_pred)
    print(f"These are the metrices after training : {metrices}")
    standardized_df = load_and_preprocess_data(training_dataset)
    
    sensitive_columns = ["gender","race","college_tier","education_level","hired"]
    sensitive_df = extract_sensitive_columns(standardized_df,sensitive_columns)
    #fairness_results = evaluate_fairness(y_test,y_pred,sensitive_df)
    
    print("\n")
    #summary_dict = compute_all_metrices(y_test,y_pred,sensitive_df)
    #print("This is the summary dictonary")
    #run_bias_report(summary_dict,output_dir)
    
    #Bias mitigation using various methods
    mitigation_config = load_mitigation_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\mitigation_config.yaml")
    sample_weights = reweighing(sensitive_df, sensitive_columns)

    #Retraining and evaluating the model after Bias Mitigation using preprocessing reweighing
    retrained_model = train_model(model_instance, X_train, y_train, sample_weights)
    y_pred_r, y_prob_r = generate_predictions(retrained_model, X_test)
    retrained_metrices = evaluate_performance(y_test, y_pred_r)
    print(f"These are the metrices after retraining: {retrained_metrices}")

    test_standardized_df = load_and_preprocess_data(testing_dataset)
    test_sensitive_df = extract_sensitive_columns(test_standardized_df, sensitive_columns)

    #Bias Mitigation using post-processor Equalized Odds Mitigation
    eo_results = run_equalized_odds_postprocessing(y_test , y_proba , mitigation_config, 0.5, test_sensitive_df)

    #SHAP interpretability
    run_interpretability_pipeline(trained_model, X_test, y_pred, y_proba, test_sensitive_df, top_k=3, threshold=0.5)

    run_interpretability_pipeline(retrained_model, X_test, y_pred_r , y_prob_r, test_sensitive_df, top_k=3, threshold=0.5)

    


if __name__ == "__main__":
    run_pipeline()

    