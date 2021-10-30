from src.utils.all_utils import read_yaml, save_reports, create_directory
from src.utils.data_management import train_valid_generator
import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def evaluate_model(config_path, params_path):
    """
    Evaluates the model
    """
    # Read the config fileh
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    # Model Directory
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    train_model_dir_path = os.path.join(artifacts_dir, artifacts['TRAINED_MODEL_DIR'])
    model_file_name = [i for i in os.listdir(train_model_dir_path) if i.endswith('.h5')][0]
    model_file_path = os.path.join(train_model_dir_path, model_file_name)

    # Load the model
    model = tf.keras.models.load_model(model_file_path)
    # evaluate the model
    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"])
    
    train_loss, train_accuracy = model.evaluate_generator(train_generator, steps=len(train_generator))
    valid_loss, valid_accuracy = model.evaluate_generator(valid_generator, steps=len(valid_generator))

    # Scores
    score_dir = config["artifacts"]["REPORT_DIR"]
    scores_dir_path = os.path.join(artifacts_dir, score_dir)
    create_directory([scores_dir_path]) 
    scores = {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy
                }

    # Save the results
    score_path = os.path.join(artifacts_dir, score_dir, config['artifacts']['MODEL_SCORE'])
    save_reports(scores, score_path)
    logging.info("Model evaluation done")
    logging.info("Model evaluation results saved at: {}".format(score_path))

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> Stage Five started !!!")
        evaluate_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("Stage Five completed! Training is Done >>>>> /n")
    except Exception as e:
        logging.exception(e)
        raise e