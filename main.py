# main.py

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import logging

import config
from data.base_loader import get_data_loader
from model.base_mllm import get_mllm

from trainer.trainer import setup_pruner, train_pruner # 假设 trainer.trainer.py 中的 setup_pruner 现在设置 RandomPruner
from evaluator.evaluator import evaluate_performance

def setup_logger():
    """Sets up the logger to write to a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(config.LOG_FILE)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main():
    """
    Main function to run the Random-based token pruning pipeline.
    """
    # 0. Setup Logger
    logger = setup_logger()
    logger.info("--- 0. Logger Initialized ---")

    # 1. Load Data
    logger.info("--- 1. Initializing Data Loader ---")
    data_loader = get_data_loader(config)
    logger.info(f"Data loader for '{config.DATASET_NAME}' initialized.")

    # 2. Load MLLM
    logger.info("--- 2. Initializing MLLM ---")
    mllm = get_mllm(config)
    logger.info(f"MLLM '{config.MODEL_ID}' initialized.")

    # 3. Setup Pruner
    logger.info("--- 3. Setting up Random Pruner ---")
   
    pruner = setup_pruner(config, mllm) # This now calls the function that sets up RandomPruner
    # Optional: Train the pruner if a training mechanism is implemented for other types
    # pruner = train_pruner(config, pruner, data_loader, mllm)
    logger.info("Random Pruner is ready.")

    # 4. Evaluate the Pruner (This is the main execution step now)
    logger.info("--- 4. Starting Pruner Evaluation ---")

    # --- 首先评估无剪枝情况 ---
    logger.info("--- Evaluating WITHOUT Pruning (EVAL_MODE='none') ---")
    config.EVAL_MODE = "none"
    evaluate_performance(pruner, config, mllm, data_loader, logger)
    
    # Evaluate in different modes
    config.EVAL_MODE = "none"
    logger.info("Skipping evaluation for EVAL_MODE='none'.")
    # evaluate_performance(pruner, config, mllm, data_loader, logger)

    config.EVAL_MODE = "full"
    evaluate_performance(pruner, config, mllm, data_loader, logger)

    #config.EVAL_MODE = "budget"
    #evaluate_performance(pruner, config, mllm, data_loader, logger)

    logger.info("Random Pruner evaluation finished.")

if __name__ == "__main__":
    main()
