#!/usr/bin/env python3
# ============================================
# architecture_c.py - Full Model Fine-tuning
# Usage: python architecture_c.py --config 5percent --split 1 --epochs 10 --batch_size 64 --auto_resume
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from transformers import WavLMModel, WavLMConfig
from tqdm import tqdm
import argparse
import os
import sys
import signal
from collections import defaultdict
import random
import json
import soundfile as sf
import numpy as np

from config import config

# Global variables for graceful shutdown
training_active = False
current_model = None
current_optimizer = None
current_scaler = None
current_checkpoint = None
current_epoch = 0
current_loss = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving checkpoint before exit...")
    if training_active and current_checkpoint:
        current_checkpoint.save(
            current_epoch + 1,
            current_model,
            current_optimizer,
            current_scaler,
            current_loss,
            is_best=False
        )
        print("✅ Checkpoint saved! You can resume with --resume flag.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================
# DATASET CLASS FOR TRIPLET LOSS
# ============================================

class TripletDataset(Dataset):
    """Dataset for triplet loss training"""
    
    def __init__(self, manifest_path, config):
        self.config = config
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Group by speaker - only keep speakers with >=2 files
        self.speaker_to_files = defaultdict(list)
        for item in self.data:
            self.speaker_to_files[item['speaker_id']].append(item)
        
        self.valid_speakers = []
        for speaker, files in self.speaker_to_files.items():
            if len(files) >= 2:
                self.valid_speakers.append(speaker)
        
        print(f"📊 Triplet dataset: {len(self.data)} files, {len(self.valid_speakers)} valid speakers")
        self.num_triplets = len(self.valid_speakers) * 20
    
    def __len__(self):
        return self.num_triplets
    
    def __getitem__(self, idx):
        # Select anchor speaker
        anchor_speaker = random.choice(self.valid_speakers)
        anchor_files = self.speaker_to_files[anchor_speaker]
        
        # Select anchor and positive (different files)
        anchor_item, positive_item = random.sample(anchor_files, 2)
        
        # Select negative from different speaker
        negative_speaker = random.choice([s for s in self.valid_speakers if s != anchor_speaker])
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
                audio = np.pad(audio, (0, self.config.MAX_LENGTH - len(audio)), 'constant')
            return torch.FloatTensor(audio)
        
        return {
            'anchor': load_audio(anchor_item),
            'positive': load_audio(positive_item),
            'negative': load_audio(negative_item)
        }


# ============================================
# ARCHITECTURE C MODEL
# ============================================

class ArchitectureC(nn.Module):
    """Architecture C: Full Model Fine-tuning"""
    
    def __init__(self, unfrozen_layers=3):
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
        print(f"✅ Architecture C: {self.trainable_params:,}/{self.total_params:,} trainable ({100.*self.trainable_params/self.total_params:.2f}%)")
    
    def forward(self, audio):
        outputs = self.wavlm(audio)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        pooled = last_hidden.mean(dim=1)
        
        # Project and normalize
        embedding = self.embedding_proj(pooled)
        return F.normalize(embedding, p=2, dim=1)


class CheckpointManager:
    def __init__(self, arch_name, split):
        self.arch_name = arch_name
        self.split = split
        self.checkpoint_dir = config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_checkpoint.pt")
        self.best_path = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_best.pt")
        self.resume_file = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_resume.txt")
    
    def save(self, epoch, model, optimizer, scaler, best_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_path)
        with open(self.resume_file, 'w') as f:
            f.write(str(epoch))
        
        if is_best:
            torch.save(model.state_dict(), self.best_path)
            print(f"✅ Saved best model (loss: {best_loss:.4f})")
    
    def load(self, model, optimizer, scaler):
        if os.path.exists(self.checkpoint_path):
            print(f"📂 Found checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"✅ Resumed from epoch {checkpoint['epoch']}")
            return checkpoint['epoch'], checkpoint['best_loss']
        return 0, float('inf')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--auto_resume', action='store_true')
    args = parser.parse_args()
    
    # Override config
    config.BATCH_SIZE = args.batch_size
    if args.auto_resume:
        args.resume = True
    
    # Set paths
    if args.split == 1:
        if args.config == '5percent':
            train_manifest = config.SPLIT1_TRAIN_5
        else:
            train_manifest = config.SPLIT1_TRAIN_50
    else:
        if args.config == '5percent':
            train_manifest = config.SPLIT2_TRAIN_5
        else:
            train_manifest = config.SPLIT2_TRAIN_50
    
    print("\n" + "="*60)
    print("🎯 ARCHITECTURE C: Full Model Fine-tuning")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {args.epochs}")
    print(f"Auto-resume: {args.auto_resume or args.resume}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    train_dataset = TripletDataset(train_manifest, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Initialize model
    model = ArchitectureC().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1)
    scaler = GradScaler('cuda', enabled=config.USE_AMP)
    checkpoint = CheckpointManager("architecture_c", args.split)
    
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = checkpoint.load(model, optimizer, scaler)
    
    # Make variables global for signal handler
    global current_model, current_optimizer, current_scaler, current_checkpoint, training_active
    current_model = model
    current_optimizer = optimizer
    current_scaler = scaler
    current_checkpoint = checkpoint
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        training_active = True
        model.train()
        train_loss = 0
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            anchor = batch['anchor'].to(config.DEVICE)
            positive = batch['positive'].to(config.DEVICE)
            negative = batch['negative'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                    loss = F.relu(pos_dist - neg_dist + 1.0).mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                loss = F.relu(pos_dist - neg_dist + 1.0).mean()
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            current_loss = train_loss / valid_batches
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = train_loss / valid_batches
        print(f"\nEpoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        checkpoint.save(epoch + 1, model, optimizer, scaler, best_loss, is_best)
        training_active = False
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()