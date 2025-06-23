from Data_Acquisation_and_preprocessing.data_preprocessor import get_data,load_and_preprocess_data
from Data_Acquisation_and_preprocessing.data_loader import log_status,extract_sensitive_columns
from Model_training_and_Validation.model_config import load_model_config,initialize_model 
from Model_training_and_Validation.model_trainer import train_model,generate_predictions,save_model
from Model_training_and_Validation.evaluator import evaluate_performance,evaluate_fairness
from Bias_Detection.bias_metrices import compute_all_metrices


def run_pipeline():
    training_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\synthetic_ai_hiring_dataset_v2.csv"
    testing_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\test_synthetic_ai_hiring_dataset_v2.csv"
    
    #Loading, preprocessing and spliting training dataset
    log_status("INFO","Processing Training Dataset : ")
    X_train,y_train = get_data(training_dataset)
    print("\n")

    #Loading, preprocessing and spliting testing dataset
    log_status("INFO","Processing Testing Dataset")
    X_test, y_test = get_data(testing_dataset)   

    config_dict = load_model_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\logistic_regression.yaml")
    model_instance = initialize_model(config_dict)

    trained_model = train_model(model_instance,X_train,y_train)
    y_pred,y_proba = generate_predictions(trained_model,X_test)
    #save_model(trained_model,"Random_forest",r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Models")
    metrices = evaluate_performance(y_test,y_pred)
    print(metrices)
    standardized_df = load_and_preprocess_data(testing_dataset)
    sensitive_columns = ["gender","race","college_tier","education_level"]

    sensitive_df = extract_sensitive_columns(standardized_df,sensitive_columns)
    fairness_results = evaluate_fairness(y_test,y_pred,sensitive_df)
    print("\n")
    summary_dict = compute_all_metrices(y_test,y_pred,sensitive_df)
    print("This is the summary dictonary")
    print(summary_dict)


if __name__ == "__main__":
    run_pipeline()