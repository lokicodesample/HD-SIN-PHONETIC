#!/usr/bin/env python3
# ============================================
# evaluate_b.py - Evaluate Architecture B (PEFT Triplet)
# Usage: python evaluate_b.py --config 5percent --split 1 --checkpoint checkpoints/architecture_b_split1_best.pt
# ============================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import sys
from collections import defaultdict

from config import config
from architecture_b import ArchitectureB


class EvalDataset:
    """Simple dataset for evaluation"""
    def __init__(self, manifest_path, config):
        import json
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import soundfile as sf
        import numpy as np
        item = self.data[idx]
        audio, sr = sf.read(item['audio_path'])
        if sr != self.config.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        if len(audio) > self.config.MAX_LENGTH:
            audio = audio[:self.config.MAX_LENGTH]
        else:
            padding = self.config.MAX_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return {
            'audio': torch.FloatTensor(audio),
            'speaker_id': item['speaker_id']
        }


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    config.BATCH_SIZE = args.batch_size
    
    # Set paths
    if args.split == 1:
        if args.config == '5percent':
            test_manifest = config.SPLIT1_TEST_5
        else:
            test_manifest = config.SPLIT1_TEST_50
    else:
        if args.config == '5percent':
            test_manifest = config.SPLIT2_TEST_5
        else:
            test_manifest = config.SPLIT2_TEST_50
    
    print("\n" + "="*60)
    print(f"📊 EVALUATING ARCHITECTURE B")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load dataset
    print("\n📂 Loading test dataset...")
    test_dataset = EvalDataset(test_manifest, config)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    print("\n🔧 Loading model...")
    model = ArchitectureB().to(config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.float()
    model.eval()
    
    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    embeddings = []
    speaker_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            audio = batch['audio'].to(config.DEVICE)
            emb = model(audio)
            embeddings.append(emb.cpu())
            speaker_ids.extend(batch['speaker_id'])
    
    embeddings = torch.cat(embeddings, dim=0)
    print(f"✅ Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    
    # Group by speaker
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(i)
    
    # Create trials
    print("\n📊 Creating evaluation trials...")
    genuine_scores = []
    impostor_scores = []
    
    # Genuine trials (same speaker)
    for spk, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            for i in range(min(len(indices), 50)):
                for j in range(i+1, min(len(indices), 50)):
                    score = F.cosine_similarity(embeddings[indices[i]].unsqueeze(0), 
                                                embeddings[indices[j]].unsqueeze(0)).item()
                    genuine_scores.append(score)
    
    # Impostor trials (different speakers)
    speakers_list = list(speaker_to_indices.keys())
    for _ in range(min(50000, len(speakers_list) * 50)):
        spk1 = np.random.choice(speakers_list)
        spk2 = np.random.choice([s for s in speakers_list if s != spk1])
        idx1 = np.random.choice(speaker_to_indices[spk1])
        idx2 = np.random.choice(speaker_to_indices[spk2])
        score = F.cosine_similarity(embeddings[idx1].unsqueeze(0), 
                                    embeddings[idx2].unsqueeze(0)).item()
        impostor_scores.append(score)
    
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    eer, _ = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n📈 RESULTS:")
    print(f"   =========================")
    print(f"   EER: {eer*100:.2f}%")
    print(f"   AUC: {roc_auc:.4f}")
    print(f"   Genuine scores mean: {np.mean(genuine_scores):.4f}")
    print(f"   Impostor scores mean: {np.mean(impostor_scores):.4f}")
    print(f"   =========================")
    
    # Save results
    output_file = f"results_b_split{args.split}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Architecture B Results\n")
        f.write(f"=====================\n")
        f.write(f"Dataset: {args.config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"EER: {eer*100:.2f}%\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Genuine mean: {np.mean(genuine_scores):.4f}\n")
        f.write(f"Impostor mean: {np.mean(impostor_scores):.4f}\n")
    print(f"\n✅ Results saved to: {output_file}")
    
    return {'eer': eer*100, 'auc': roc_auc}


if __name__ == "__main__":
    evaluate()