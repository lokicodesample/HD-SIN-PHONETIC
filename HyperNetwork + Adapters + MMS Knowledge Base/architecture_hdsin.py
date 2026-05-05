#!/usr/bin/env python3
# ============================================
# architecture_hdsin.py - HD-SIN-Phonetic with AUTO-RESUME
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import WavLMModel, WavLMConfig, Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import argparse
import os
import sys
import signal
import atexit

from config import config
from dataset_hdsin import HDSINDataset, TripletHDSINDataset

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
            current_epoch + 1,  # Save next epoch to resume from here
            current_model,
            current_optimizer,
            current_scaler,
            current_loss,
            is_best=False
        )
        print("✅ Checkpoint saved! You can resume with --resume flag.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class HyperNetwork(nn.Module):
    """Generates adapter weights for each speaker-language pair"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class HDSINPhonetic(nn.Module):
    """
    HD-SIN-Phonetic: Your Novel Architecture
    """
    
    def __init__(self, num_speakers, num_languages=3):
        super().__init__()
        
        print("\n" + "="*60)
        print("🚀 INITIALIZING HD-SIN-PHONETIC")
        print("="*60)
        
        # ============================================
        # 1. FROZEN WAVLM
        # ============================================
        print("\n📌 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.wavlm.eval()
        
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        self.hidden_size = wavlm_config.hidden_size  # 1024
        
        # ============================================
        # 2. FROZEN MMS - Phonetic Knowledge Base
        # ============================================
        print("📌 Loading MMS model for phonetic knowledge...")
        self.mms_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/mms-1b-all",
            attn_implementation="sdpa",
            torch_dtype=torch.float16
        )
        self.mms_processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
        
        for param in self.mms_model.parameters():
            param.requires_grad = False
        self.mms_model.eval()
        
        self.mms_vocab_size = self.mms_model.config.vocab_size
        
        # ============================================
        # 3. TRAINABLE EMBEDDINGS
        # ============================================
        print("📌 Creating trainable embeddings...")
        self.speaker_embeddings = nn.Embedding(num_speakers, config.SPEAKER_EMBED_DIM)
        self.language_embeddings = nn.Embedding(num_languages, config.LANG_EMBED_DIM)
        self.phone_embeddings = nn.Embedding(self.mms_vocab_size, config.PHONE_EMBED_DIM)
        
        # ============================================
        # 4. HYPERNETWORK
        # ============================================
        print("📌 Building HyperNetwork...")
        hyper_input_dim = config.SPEAKER_EMBED_DIM + config.LANG_EMBED_DIM
        hyper_output_dim = 24 * config.ADAPTER_DIM * 2
        
        self.hypernetwork = HyperNetwork(
            input_dim=hyper_input_dim,
            hidden_dim=config.HYPERNET_HIDDEN,
            output_dim=hyper_output_dim
        )
        
        # ============================================
        # 5. CROSS-ATTENTION FUSION
        # ============================================
        print("📌 Building cross-attention fusion...")
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            kdim=config.PHONE_EMBED_DIM,
            vdim=config.PHONE_EMBED_DIM,
            num_heads=8,
            batch_first=True
        )
        
        # ============================================
        # 6. FINAL PROJECTION
        # ============================================
        self.speaker_projection = nn.Linear(self.hidden_size, config.EMBEDDING_DIM)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # ============================================
        # 7. COUNT PARAMETERS
        # ============================================
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n✅ HD-SIN-Phonetic initialized:")
        print(f"   Trainable parameters: {self.trainable_params:,}")
        print(f"   Speaker embeddings: {num_speakers} × {config.SPEAKER_EMBED_DIM}")
    
    def forward(self, audio, speaker_ids=None, language_ids=None):
        batch_size = audio.shape[0]
        
        # STEP 1: Extract WavLM Features
        with torch.no_grad():
            wavlm_outputs = self.wavlm(audio, output_hidden_states=True)
            wavlm_features = wavlm_outputs.last_hidden_state
        
        # STEP 2: Extract Phonetic Knowledge from MMS
        with torch.no_grad():
            inputs = self.mms_processor(audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(audio.device)
            
            mms_outputs = self.mms_model(input_values)
            mms_logits = mms_outputs.logits
            phone_ids = mms_logits.argmax(dim=-1)
            phone_embs = self.phone_embeddings(phone_ids)
        
        # Align time steps
        if phone_embs.shape[1] != wavlm_features.shape[1]:
            phone_embs = phone_embs.transpose(1, 2)
            phone_embs = F.interpolate(
                phone_embs, 
                size=wavlm_features.shape[1],
                mode='linear',
                align_corners=False
            )
            phone_embs = phone_embs.transpose(1, 2)
        
        # STEP 3: Cross-Attention Fusion
        enhanced_features, attention_weights = self.cross_attention(
            wavlm_features, phone_embs, phone_embs
        )
        enhanced_features = self.layer_norm(enhanced_features + wavlm_features)
        
        # STEP 4: Pool to Final Speaker Embedding
        pooled = enhanced_features.mean(dim=1)
        speaker_embedding = self.speaker_projection(pooled)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=1)
        
        return speaker_embedding, attention_weights


class CheckpointManager:
    def __init__(self, arch_name):
        self.arch_name = arch_name
        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{arch_name}_checkpoint.pt")
        self.best_path = os.path.join(self.checkpoint_dir, f"{arch_name}_best.pt")
        
        # Auto-resume file - stores last epoch
        self.resume_file = os.path.join(self.checkpoint_dir, f"{arch_name}_resume.txt")
        
    def save(self, epoch, model, optimizer, scaler, best_loss, is_best=False):
        """Save checkpoint with all training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'arch': self.arch_name
            }
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save full checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save resume info
        with open(self.resume_file, 'w') as f:
            f.write(str(epoch))
        
        if is_best:
            torch.save(model.state_dict(), self.best_path)
            print(f"✅ Saved best model (loss: {best_loss:.4f})")
    
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
            
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            
            print(f"✅ Resumed from epoch {epoch} (best loss: {best_loss:.4f})")
            return epoch, best_loss
        
        return 0, float('inf')
    
    def get_last_epoch(self):
        """Get last completed epoch from resume file"""
        if os.path.exists(self.resume_file):
            with open(self.resume_file, 'r') as f:
                return int(f.read().strip())
        return 0


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--auto_resume', action='store_true', help='Auto-resume if checkpoint exists')
    args = parser.parse_args()
    
    # Override batch size
    config.BATCH_SIZE = args.batch_size
    
    # Set paths
    if args.split == 1:
        train_manifest = config.SPLIT1_TRAIN_5 if args.config == '5percent' else config.SPLIT1_TRAIN_50
    else:
        train_manifest = config.SPLIT2_TRAIN_5 if args.config == '5percent' else config.SPLIT2_TRAIN_50
    
    print("\n" + "="*60)
    print("🎯 HD-SIN-PHONETIC")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {args.epochs}")
    print(f"Auto-resume: {args.auto_resume or args.resume}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    train_dataset = TripletHDSINDataset(train_manifest, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = HDSINPhonetic(num_speakers=config.NUM_SPEAKERS).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler('cuda', enabled=config.USE_AMP)
    checkpoint = CheckpointManager(f"hdsin_split{args.split}")
    
    # Handle resume logic
    start_epoch = 0
    best_loss = float('inf')
    
    if args.auto_resume:
        # Auto-resume if checkpoint exists
        args.resume = True
        print("🔄 Auto-resume enabled - will continue from last checkpoint")
    
    if args.resume:
        start_epoch, best_loss = checkpoint.load(model, optimizer, scaler)
        print(f"📊 Resuming from epoch {start_epoch}/{args.epochs}")
    else:
        print("🆕 Starting fresh training")
    
    # Make variables global for signal handler
    global training_active, current_model, current_optimizer, current_scaler
    global current_checkpoint, current_epoch, current_loss
    
    current_model = model
    current_optimizer = optimizer
    current_scaler = scaler
    current_checkpoint = checkpoint
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        training_active = True
        current_epoch = epoch
        
        model.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            current_batch = batch_idx
            
            anchor = batch['anchor'].to(config.DEVICE)
            positive = batch['positive'].to(config.DEVICE)
            negative = batch['negative'].to(config.DEVICE)
            anchor_lang = batch['anchor_language'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.USE_AMP:
                with autocast('cuda'):
                    anchor_emb, _ = model(anchor, None, anchor_lang)
                    positive_emb, _ = model(positive, None, anchor_lang)
                    negative_emb, _ = model(negative, None, anchor_lang)
                    
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                    loss = F.relu(pos_dist - neg_dist + 1.0).mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                anchor_emb, _ = model(anchor, None, anchor_lang)
                positive_emb, _ = model(positive, None, anchor_lang)
                negative_emb, _ = model(negative, None, anchor_lang)
                
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                loss = F.relu(pos_dist - neg_dist + 1.0).mean()
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            current_loss = total_loss / valid_batches
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{current_loss:.4f}'})
            
            # Auto-save every 100 batches (optional)
            if batch_idx % 100 == 0 and batch_idx > 0:
                checkpoint.save(epoch + 1, model, optimizer, scaler, best_loss, is_best=False)
        
        avg_loss = total_loss / valid_batches
        print(f"\nEpoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        checkpoint.save(epoch + 1, model, optimizer, scaler, best_loss, is_best)
        
        training_active = False
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"📁 Checkpoints saved in: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()