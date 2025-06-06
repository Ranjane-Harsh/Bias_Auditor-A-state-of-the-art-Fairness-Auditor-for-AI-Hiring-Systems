from Data_Acquisation_and_preprocessing.data_preprocessor import get_data
from Model_training_and_Validation.model_config import load_model_config,initialize_model 
from Model_training_and_Validation.model_trainer import train_model


def run_pipeline():
    file_path = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\synthetic_ai_hiring_dataset_v2.csv"
    X_train,y_train = get_data(file_path)
    config_dict = load_model_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\random_forest.yaml")
    model_instance = initialize_model(config_dict)
    trained_model = train_model(model_instance,X_train,y_train)


if __name__ == "__main__":
    run_pipeline()