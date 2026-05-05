from config import Config as BaseConfig
import os

class Config(BaseConfig):
    # Use 50% subsets
    SPLIT1_TRAIN = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/50percent_split1_train.jsonl"
    SPLIT1_VAL = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/50percent_split1_val.jsonl"
    SPLIT1_TEST = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets/50percent_split1_test.jsonl"
    
    # Optimized for A6000
    BATCH_SIZE = 48
    EPOCHS = 20
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 4
    USE_AMP = True