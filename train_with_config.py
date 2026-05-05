#!/usr/bin/env python3
# ============================================
# train_with_config.py - Train with custom config file
# Usage: python train_with_config.py --arch a --config config_test_5percent.py --epochs 2
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import WavLMModel, WavLMConfig
import json
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import argparse
import importlib.util
from collections import defaultdict
import random
import sys

# ============================================
# DATASET CLASSES
# ============================================

class SpeakerDataset(Dataset):
    """Dataset for speaker identification (Architecture A)"""
    
    def __init__(self, manifest_path, config, is_training=True):
        self.config = config
        self.is_training = is_training
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Create speaker to index mapping
        self.speaker_to_idx = {}
        self.idx_to_speaker = []
        for item in self.data:
            if item['speaker_id'] not in self.speaker_to_idx:
                idx = len(self.speaker_to_idx)
                self.speaker_to_idx[item['speaker_id']] = idx
                self.idx_to_speaker.append(item['speaker_id'])
        
        print(f"📊 Dataset loaded: {len(self.data)} files, {len(self.speaker_to_idx)} speakers")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio_path = item['audio_path']
        audio, sr = sf.read(audio_path)
        
        # Resample if needed
        if sr != self.config.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        
        # Pad or truncate
        if len(audio) > self.config.MAX_LENGTH:
            audio = audio[:self.config.MAX_LENGTH]
        else:
            padding = self.config.MAX_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        # Convert to tensor
        audio = torch.FloatTensor(audio)
        
        # Get speaker label
        speaker_id = item['speaker_id']
        label = self.speaker_to_idx[speaker_id]
        
        return {
            'audio': audio,
            'label': label,
            'speaker_id': speaker_id,
            'path': audio_path
        }


class TripletDataset(Dataset):
    """Dataset for triplet loss training (Architectures B and C)"""
    
    def __init__(self, manifest_path, config):
        self.config = config
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Group by speaker
        self.speaker_to_files = defaultdict(list)
        for item in self.data:
            self.speaker_to_files[item['speaker_id']].append(item)
        
        self.speakers = list(self.speaker_to_files.keys())
        print(f"📊 Triplet dataset: {len(self.data)} files, {len(self.speakers)} speakers")
    
    def __len__(self):
        return len(self.data) // 3
    
    def __getitem__(self, idx):
        # Select anchor speaker
        anchor_speaker = random.choice(self.speakers)
        anchor_files = self.speaker_to_files[anchor_speaker]
        
        # Select anchor and positive
        anchor_item = random.choice(anchor_files)
        positive_item = random.choice([f for f in anchor_files if f['audio_path'] != anchor_item['audio_path']])
        
        # Select negative
        negative_speaker = random.choice([s for s in self.speakers if s != anchor_speaker])
        negative_item = random.choice(self.speaker_to_files[negative_speaker])
        
        # Load audio helper
        def load_audio(item):
            audio, sr = sf.read(item['audio_path'])
            if sr != self.config.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
            if len(audio) > self.config.MAX_LENGTH:
                audio = audio[:self.config.MAX_LENGTH]
            else:
                padding = self.config.MAX_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')
            return torch.FloatTensor(audio)
        
        return {
            'anchor': load_audio(anchor_item),
            'positive': load_audio(positive_item),
            'negative': load_audio(negative_item),
            'anchor_speaker': anchor_speaker,
            'negative_speaker': negative_speaker
        }


# ============================================
# ARCHITECTURE CLASSES
# ============================================

class ArchitectureA(nn.Module):
    """Architecture A: Classifier Fine-tuning"""
    
    def __init__(self, config):
        super().__init__()
        
        print("🔄 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        
        # Freeze all WavLM parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable components
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.NUM_SPEAKERS)
        )
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Architecture A: {self.trainable_params:,} trainable params")
    
    def forward(self, audio):
        outputs = self.wavlm(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        
        # Weighted sum of layers
        weighted_sum = torch.zeros_like(hidden_states[-1])
        for i, hidden in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * hidden
        
        # Mean pooling
        pooled = weighted_sum.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits, pooled


class ArchitectureB(nn.Module):
    """Architecture B: PEFT Metric Learning"""
    
    def __init__(self, config):
        super().__init__()
        
        print("🔄 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        
        # Freeze all WavLM parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable components
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        self.embedding_proj = nn.Linear(hidden_size, config.EMBEDDING_DIM)
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Architecture B: {self.trainable_params:,} trainable params")
    
    def forward(self, audio):
        outputs = self.wavlm(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        
        # Weighted sum
        weighted_sum = torch.zeros_like(hidden_states[-1])
        for i, hidden in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * hidden
        
        # Mean pooling
        pooled = weighted_sum.mean(dim=1)
        
        # Project and normalize
        embedding = self.embedding_proj(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class ArchitectureC(nn.Module):
    """Architecture C: Full Model Fine-tuning"""
    
    def __init__(self, config, unfrozen_layers=3):
        super().__init__()
        
        print("🔄 Loading WavLM-large (partially frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        
        # Freeze all first
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers
        if hasattr(self.wavlm, 'encoder') and hasattr(self.wavlm.encoder, 'layers'):
            total_layers = len(self.wavlm.encoder.layers)
            for i in range(total_layers - unfrozen_layers, total_layers):
                for param in self.wavlm.encoder.layers[i].parameters():
                    param.requires_grad = True
            print(f"   Unfroze top {unfrozen_layers} of {total_layers} layers")
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable projection
        self.embedding_proj = nn.Linear(hidden_size, config.EMBEDDING_DIM)
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Architecture C: {self.trainable_params:,} / {self.total_params:,} trainable ({100.*self.trainable_params/self.total_params:.2f}%)")
    
    def forward(self, audio):
        outputs = self.wavlm(audio)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        pooled = last_hidden.mean(dim=1)
        
        # Project and normalize
        embedding = self.embedding_proj(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


# ============================================
# CHECKPOINT MANAGER
# ============================================

class CheckpointManager:
    """Manages saving and loading checkpoints"""
    
    def __init__(self, arch_name, checkpoint_dir="checkpoints"):
        self.arch_name = arch_name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, f"{arch_name}_checkpoint.pt")
        self.best_model_path = os.path.join(checkpoint_dir, f"{arch_name}_best.pt")
        
    def save(self, epoch, model, optimizer, scaler, best_score, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'config': {
                'batch_size': getattr(model, 'batch_size', 32),
                'arch': self.arch_name
            }
        }
        
        # Add scaler for mixed precision
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best model separately
        if is_best:
            torch.save(model.state_dict(), self.best_model_path)
            print(f"✅ Saved best model (score: {best_score:.4f})")
    
    def load(self, model, optimizer=None, scaler=None):
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            print(f"📂 Found checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cuda')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"✅ Resumed from epoch {checkpoint['epoch']}")
            return checkpoint['epoch'], checkpoint['best_score']
        
        return 0, float('inf')


# ============================================
# TRAINING FUNCTIONS
# ============================================

def create_dataloaders(config):
    """Create all dataloaders"""
    
    # For Architecture A
    train_dataset_a = SpeakerDataset(config.SPLIT1_TRAIN, config)
    val_dataset_a = SpeakerDataset(config.SPLIT1_VAL, config)
    
    train_loader_a = DataLoader(
        train_dataset_a, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    val_loader_a = DataLoader(
        val_dataset_a,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    # For Architectures B and C
    train_dataset_triplet = TripletDataset(config.SPLIT1_TRAIN, config)
    
    train_loader_triplet = DataLoader(
        train_dataset_triplet,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    return {
        'a': (train_loader_a, val_loader_a),
        'triplet': train_loader_triplet
    }


def train_architecture_a(train_loader, val_loader, config, resume=False):
    """Train Architecture A with checkpointing and AMP"""
    
    model = ArchitectureA(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler - CORRECT SYNTAX
    scaler = GradScaler(enabled=config.USE_AMP)  # Remove 'cuda' argument
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager("architecture_a")
    
    start_epoch = 0
    best_acc = 0
    
    # Resume from checkpoint if requested
    if resume:
        start_epoch, best_acc = checkpoint_manager.load(model, optimizer, scaler)
    
    print("\n🚀 Training Architecture A (Classifier)")
    print("="*60)
    print(f"Starting from epoch {start_epoch+1}/{config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}, Mixed Precision: {config.USE_AMP}")
    
    for epoch in range(start_epoch, config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in pbar:
            audio = batch['audio'].to(config.DEVICE, non_blocking=True)
            labels = batch['label'].to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward - CORRECT SYNTAX
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(audio)
                    loss = criterion(logits, labels)
            else:
                logits, _ = model(audio)
                loss = criterion(logits, labels)
            
            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
                audio = batch['audio'].to(config.DEVICE, non_blocking=True)
                labels = batch['label'].to(config.DEVICE, non_blocking=True)
                
                if config.USE_AMP:
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(audio)
                else:
                    logits, _ = model(audio)
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        checkpoint_manager.save(epoch + 1, model, optimizer, scaler, best_acc, is_best)
    
    return model


def train_architecture_b(train_loader, config, resume=False):
    """Train Architecture B with checkpointing and AMP"""
    
    model = ArchitectureB(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed precision scaler - FIXED
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager("architecture_b")
    
    start_epoch = 0
    best_loss = float('inf')
    
    if resume:
        start_epoch, best_loss = checkpoint_manager.load(model, optimizer, scaler)
    
    print("\n🚀 Training Architecture B (PEFT + Triplet)")
    print("="*60)
    print(f"Starting from epoch {start_epoch+1}/{config.EPOCHS}")
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in pbar:
            anchor = batch['anchor'].to(config.DEVICE, non_blocking=True)
            positive = batch['positive'].to(config.DEVICE, non_blocking=True)
            negative = batch['negative'].to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward - FIXED
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                    loss = F.relu(pos_dist - neg_dist + 1.0).mean()
            else:
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                loss = F.relu(pos_dist - neg_dist + 1.0).mean()
            
            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        checkpoint_manager.save(epoch + 1, model, optimizer, scaler, best_loss, is_best)
    
    return model


def train_architecture_c(train_loader, config, resume=False):
    """Train Architecture C with checkpointing and AMP"""
    
    model = ArchitectureC(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1)
    
    # Mixed precision scaler - FIXED
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager("architecture_c")
    
    start_epoch = 0
    best_loss = float('inf')
    
    if resume:
        start_epoch, best_loss = checkpoint_manager.load(model, optimizer, scaler)
    
    print("\n🚀 Training Architecture C (Full Fine-tuning)")
    print("="*60)
    print(f"Starting from epoch {start_epoch+1}/{config.EPOCHS}")
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in pbar:
            anchor = batch['anchor'].to(config.DEVICE, non_blocking=True)
            positive = batch['positive'].to(config.DEVICE, non_blocking=True)
            negative = batch['negative'].to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward - FIXED
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                    loss = F.relu(pos_dist - neg_dist + 1.0).mean()
            else:
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                loss = F.relu(pos_dist - neg_dist + 1.0).mean()
            
            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        checkpoint_manager.save(epoch + 1, model, optimizer, scaler, best_loss, is_best)
    
    return model


# ============================================
# CONFIG LOADER
# ============================================

def load_config(config_path):
    """Load config file dynamically"""
    try:
        spec = importlib.util.spec_from_file_location("custom_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.Config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train speaker identification architectures')
    parser.add_argument('--arch', type=str, default='all', 
                        choices=['a', 'b', 'c', 'all'],
                        help='Architecture to train (a, b, c, or all)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file (e.g., config_test_5percent.py)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')
    
    args = parser.parse_args()
    
    # Load custom config
    print(f"\n📂 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config.EPOCHS = args.epochs
        print(f"   Overriding epochs to: {config.EPOCHS}")
    
    # Print system info
    print("\n" + "="*60)
    print("🎯 BASELINE ARCHITECTURES TRAINING")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Mixed precision: {config.USE_AMP}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Resume: {args.resume}")
    print(f"Architecture: {args.arch}")
    
    # Verify manifest files exist
    if not os.path.exists(config.SPLIT1_TRAIN):
        print(f"\n❌ Error: Training manifest not found: {config.SPLIT1_TRAIN}")
        print("Please check your config file paths.")
        sys.exit(1)
    
    # Create dataloaders
    print("\n📂 Creating dataloaders...")
    loaders = create_dataloaders(config)
    
    # Train selected architectures
    if args.arch == 'a' or args.arch == 'all':
        train_architecture_a(loaders['a'][0], loaders['a'][1], config, args.resume)
    
    if args.arch == 'b' or args.arch == 'all':
        train_architecture_b(loaders['triplet'], config, args.resume)
    
    if args.arch == 'c' or args.arch == 'all':
        train_architecture_c(loaders['triplet'], config, args.resume)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("\nModels and checkpoints saved in 'checkpoints' directory:")
    if args.arch in ['a', 'all']:
        print("  - architecture_a_best.pt")
        print("  - architecture_a_checkpoint.pt")
    if args.arch in ['b', 'all']:
        print("  - architecture_b_best.pt")
        print("  - architecture_b_checkpoint.pt")
    if args.arch in ['c', 'all']:
        print("  - architecture_c_best.pt")
        print("  - architecture_c_checkpoint.pt")


if __name__ == "__main__":
    main()