#!/usr/bin/env python3
# ============================================
# architecture_a.py - Classifier Fine-tuning
# Usage: python architecture_a.py --config 5percent --split 1 --epochs 10 --batch_size 64 --auto_resume
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import WavLMModel, WavLMConfig
from tqdm import tqdm
import argparse
import os
import sys
import signal

from config import config
from dataset import SpeakerDataset

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


class ArchitectureA(nn.Module):
    """Architecture A: Classifier Fine-tuning"""
    
    def __init__(self, num_speakers):
        super().__init__()
        
        print("🔄 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_speakers)
        )
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Architecture A: {self.trainable_params:,} trainable params")
    
    def forward(self, audio):
        outputs = self.wavlm(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        
        weighted_sum = torch.zeros_like(hidden_states[-1])
        for i, hidden in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * hidden
        
        pooled = weighted_sum.mean(dim=1)
        return self.classifier(pooled)


class CheckpointManager:
    def __init__(self, arch_name, split):
        self.arch_name = arch_name
        self.split = split
        self.checkpoint_dir = config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_checkpoint.pt")
        self.best_path = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_best.pt")
        self.resume_file = os.path.join(self.checkpoint_dir, f"{arch_name}_split{split}_resume.txt")
    
    def save(self, epoch, model, optimizer, scaler, best_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_path)
        with open(self.resume_file, 'w') as f:
            f.write(str(epoch))
        
        if is_best:
            torch.save(model.state_dict(), self.best_path)
            print(f"✅ Saved best model (acc: {best_acc:.2f}%)")
    
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
            return checkpoint['epoch'], checkpoint['best_acc']
        return 0, 0


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
            val_manifest = config.SPLIT1_VAL_5
        else:
            train_manifest = config.SPLIT1_TRAIN_50
            val_manifest = config.SPLIT1_VAL_50
    else:
        if args.config == '5percent':
            train_manifest = config.SPLIT2_TRAIN_5
            val_manifest = config.SPLIT2_VAL_5
        else:
            train_manifest = config.SPLIT2_TRAIN_50
            val_manifest = config.SPLIT2_VAL_50
    
    print("\n" + "="*60)
    print("🎯 ARCHITECTURE A: Classifier Fine-tuning")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {args.epochs}")
    print(f"Auto-resume: {args.auto_resume or args.resume}")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = SpeakerDataset(train_manifest, config)
    val_dataset = SpeakerDataset(val_manifest, config)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    # Initialize model
    model = ArchitectureA(len(train_dataset.speaker_to_idx)).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=config.USE_AMP)
    checkpoint = CheckpointManager("architecture_a", args.split)
    
    start_epoch = 0
    best_acc = 0
    if args.resume:
        start_epoch, best_acc = checkpoint.load(model, optimizer, scaler)
    
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
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            audio = batch['audio'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    logits = model(audio)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(audio)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                audio = batch['audio'].to(config.DEVICE)
                labels = batch['label'].to(config.DEVICE)
                
                if config.USE_AMP:
                    with torch.amp.autocast('cuda'):
                        logits = model(audio)
                else:
                    logits = model(audio)
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        checkpoint.save(epoch + 1, model, optimizer, scaler, best_acc, is_best)
        training_active = False
    
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()