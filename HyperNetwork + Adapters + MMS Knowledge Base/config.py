# ============================================
# config.py - Shared configuration for all architectures
# ============================================

import torch
import os

class Config:
    # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
    DATA_DIR = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset"
    
    # For 5% test version
    SPLIT1_TRAIN_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_train.jsonl")
    SPLIT1_VAL_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_val.jsonl")
    SPLIT1_TEST_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split1_test.jsonl")
    
    # For 50% version
    SPLIT1_TRAIN_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_train.jsonl")
    SPLIT1_VAL_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_val.jsonl")
    SPLIT1_TEST_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split1_test.jsonl")
    
    # For Split 2 (open-set)
    SPLIT2_TRAIN_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_train.jsonl")
    SPLIT2_VAL_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_val.jsonl")
    SPLIT2_TEST_5 = os.path.join(DATA_DIR, "manifests_subsets", "5percent_split2_test.jsonl")
    
    SPLIT2_TRAIN_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_train.jsonl")
    SPLIT2_VAL_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_val.jsonl")
    SPLIT2_TEST_50 = os.path.join(DATA_DIR, "manifests_subsets", "50percent_split2_test.jsonl")
    
    # Audio parameters
    SAMPLE_RATE = 16000
    MAX_DURATION = 7.0  # seconds
    MAX_LENGTH = int(MAX_DURATION * SAMPLE_RATE)
    
    # Training parameters
    BATCH_SIZE = 32  # Slightly smaller for complex model
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU Optimizations
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    USE_AMP = True  # Mixed precision training
    
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
    NUM_WAVLM_LAYERS = 24
    ADAPTER_DIM = 128

    # Add these to your config.py if not present

config = Config()