# ============================================
# config.py - Updated with Split 2 paths
# ============================================

import torch
import os

class Config:
    # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
    DATA_DIR = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset"
    
    # Split 1 (Identification)
    SPLIT1_TRAIN_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_train.jsonl")
    SPLIT1_VAL_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_val.jsonl")
    SPLIT1_TEST_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_test.jsonl")
    
    SPLIT1_TRAIN_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_train.jsonl")
    SPLIT1_VAL_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_val.jsonl")
    SPLIT1_TEST_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_test.jsonl")
    
    # Split 2 (Open-set) - ADD THESE
    SPLIT2_TRAIN_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_train.jsonl")
    SPLIT2_VAL_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_val.jsonl")
    SPLIT2_TEST_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_test.jsonl")
    
    SPLIT2_TRAIN_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_train.jsonl")
    SPLIT2_VAL_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_val.jsonl")
    SPLIT2_TEST_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_test.jsonl")
    
    # Audio parameters
    SAMPLE_RATE = 16000
    MAX_DURATION = 7.0
    MAX_LENGTH = int(MAX_DURATION * SAMPLE_RATE)
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU Optimizations
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    USE_AMP = True
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Model parameters
    NUM_SPEAKERS = 11315
    EMBEDDING_DIM = 256
    
    # HD-SIN Specific parameters
    SPEAKER_EMBED_DIM = 64
    LANG_EMBED_DIM = 32
    PHONE_EMBED_DIM = 128
    HYPERNET_HIDDEN = 512
    ADAPTER_DIM = 128

config = Config()