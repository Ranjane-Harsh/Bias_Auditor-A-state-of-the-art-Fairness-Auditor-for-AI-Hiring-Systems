from Data_Acquisation_and_preprocessing.data_preprocessor import get_data
from Data_Acquisation_and_preprocessing.data_loader import log_status
from Model_training_and_Validation.model_config import load_model_config,initialize_model 
from Model_training_and_Validation.model_trainer import train_model,generate_predictions


def run_pipeline():
    training_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\synthetic_ai_hiring_dataset_v2.csv"
    testing_dataset = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\test_synthetic_ai_hiring_dataset_v2.csv"
    log_status("INFO","Processing Training Dataset : ")
    X_train,y_train = get_data(training_dataset)
    print("\n")
    log_status("INFO","Processing Testing Dataset")
    X_test, y_test = get_data(testing_dataset)

    config_dict = load_model_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\random_forest.yaml")
    model_instance = initialize_model(config_dict)

    trained_model = train_model(model_instance,X_train,y_train)
    y_pred,y_proba = generate_predictions(trained_model,X_test)
    print(y_pred[:5])
    print(y_proba[:5])


if __name__ == "__main__":
    run_pipeline()