from config import Config as BaseConfig
import os

class Config(BaseConfig):
    # Use 5% subsets
    SPLIT1_TRAIN = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/5percent_split1_train.jsonl"
    SPLIT1_VAL = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/5percent_split1_val.jsonl"
    SPLIT1_TEST = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/5percent_split1_test.jsonl"
    
    # Smaller batch for testing
    BATCH_SIZE = 48
    EPOCHS = 2
    NUM_WORKERS = 16