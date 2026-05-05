#!/usr/bin/env python3
# ============================================
# train.py - Enhanced with checkpointing & GPU optimizations
# Run: python train.py --arch [a|b|c|all] --resume
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import os
import sys
import json
from datetime import datetime

from config import config
from dataset import create_dataloaders
from architectures import ArchitectureA, ArchitectureB, ArchitectureC

class CheckpointManager:
    """Manages saving and loading checkpoints"""
    
    def __init__(self, arch_name, checkpoint_dir=config.CHECKPOINT_DIR):
        self.arch_name = arch_name
        self.checkpoint_dir = checkpoint_dir
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
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
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
            checkpoint = torch.load(self.checkpoint_path, map_location=config.DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"✅ Resumed from epoch {checkpoint['epoch']}")
            return checkpoint['epoch'], checkpoint['best_score']
        
        return 0, float('inf')


def train_architecture_a(train_loader, val_loader, resume=False):
    """Train Architecture A with checkpointing and AMP"""
    
    model = ArchitectureA(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
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
            
            # Mixed precision forward
            with autocast(enabled=config.USE_AMP):
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
                
                with autocast(enabled=config.USE_AMP):
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


def train_architecture_b(train_loader, resume=False):
    """Train Architecture B with checkpointing and AMP"""
    
    model = ArchitectureB(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed precision scaler
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
            
            # Mixed precision forward
            with autocast(enabled=config.USE_AMP):
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


def train_architecture_c(train_loader, resume=False):
    """Train Architecture C with checkpointing and AMP"""
    
    model = ArchitectureC(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1)
    
    # Mixed precision scaler
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
            
            # Mixed precision forward
            with autocast(enabled=config.USE_AMP):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='all', 
                        choices=['a', 'b', 'c', 'all'],
                        help='Architecture to train (a, b, c, or all)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs to train')
    args = parser.parse_args()
    
    # Update epochs
    config.EPOCHS = args.epochs
    
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
    
    # Create dataloaders
    print("\n📂 Creating dataloaders...")
    loaders = create_dataloaders(config)
    
    # Train selected architectures
    if args.arch == 'a' or args.arch == 'all':
        train_architecture_a(loaders['a'][0], loaders['a'][1], args.resume)
    
    if args.arch == 'b' or args.arch == 'all':
        train_architecture_b(loaders['triplet'], args.resume)
    
    if args.arch == 'c' or args.arch == 'all':
        train_architecture_c(loaders['triplet'], args.resume)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("\nModels and checkpoints saved in 'checkpoints' directory:")
    print("  - architecture_*_best.pt (best model only)")
    print("  - architecture_*_checkpoint.pt (full checkpoint for resuming)")


if __name__ == "__main__":
    main()