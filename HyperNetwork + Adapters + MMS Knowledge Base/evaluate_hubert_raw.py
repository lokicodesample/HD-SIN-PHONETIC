#!/usr/bin/env python3
# ============================================
# evaluate_hubert_raw.py - Evaluate Raw HuBERT (No Fine-tuning)
# Usage: 
#   python evaluate_hubert_raw.py --config 5percent --split 1
#   python evaluate_hubert_raw.py --config 5percent --split 2
# ============================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import json
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from transformers import HubertModel, HubertConfig
from config import config
from dataset_hdsin import HDSINDataset


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


def compute_per_language_eer(embeddings, speaker_ids, languages):
    """Compute EER per language"""
    results = {}
    language_map = {'tamil': 0, 'telugu': 1, 'english': 2}
    
    for lang_name, lang_idx in language_map.items():
        lang_indices = [i for i, lang in enumerate(languages) if lang == lang_name]
        
        if len(lang_indices) < 10:
            print(f"⚠️ Not enough samples for {lang_name} ({len(lang_indices)} files)")
            continue
        
        lang_embeddings = embeddings[lang_indices]
        lang_speakers = [speaker_ids[i] for i in lang_indices]
        
        genuine_scores = []
        impostor_scores = []
        
        speaker_to_idx = defaultdict(list)
        for i, spk in enumerate(lang_speakers):
            speaker_to_idx[spk].append(i)
        
        # Genuine trials
        for spk, indices in speaker_to_idx.items():
            if len(indices) >= 2:
                n_trials = min(len(indices), 50)
                for i in range(n_trials):
                    for j in range(i+1, n_trials):
                        score = F.cosine_similarity(
                            lang_embeddings[indices[i]].unsqueeze(0),
                            lang_embeddings[indices[j]].unsqueeze(0)
                        ).item()
                        genuine_scores.append(score)
        
        # Impostor trials
        speakers_list = list(speaker_to_idx.keys())
        for _ in range(min(10000, len(speakers_list) * 20)):
            spk1 = np.random.choice(speakers_list)
            spk2 = np.random.choice([s for s in speakers_list if s != spk1])
            idx1 = np.random.choice(speaker_to_idx[spk1])
            idx2 = np.random.choice(speaker_to_idx[spk2])
            score = F.cosine_similarity(
                lang_embeddings[idx1].unsqueeze(0),
                lang_embeddings[idx2].unsqueeze(0)
            ).item()
            impostor_scores.append(score)
        
        if genuine_scores and impostor_scores:
            all_scores = genuine_scores + impostor_scores
            all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
            eer, _ = compute_eer(all_labels, all_scores)
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            
            results[lang_name] = {
                'eer': eer * 100,
                'auc': roc_auc,
                'genuine_count': len(genuine_scores),
                'impostor_count': len(impostor_scores),
                'genuine_mean': np.mean(genuine_scores),
                'impostor_mean': np.mean(impostor_scores)
            }
    
    return results


class RawHuBERTExtractor:
    """Extract embeddings from frozen HuBERT without any fine-tuning"""
    
    def __init__(self, device='cuda'):
        print("🔄 Loading raw HuBERT-large (no fine-tuning, frozen)...")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert.to(device)
        self.hubert.eval()
        
        # Freeze all parameters
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        self.device = device
        self.hidden_size = 1024
        print(f"✅ Raw HuBERT loaded, hidden size: {self.hidden_size}")
        
    def encode_batch(self, audio):
        """
        Extract embeddings from raw HuBERT using mean pooling
        audio: [batch, time] tensor
        Returns: [batch, 1, hidden_size]
        """
        with torch.no_grad():
            outputs = self.hubert(audio, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.unsqueeze(1)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_trials', type=int, default=50000)
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()
    
    config.BATCH_SIZE = args.batch_size
    
    # Set paths
    if args.split == 1:
        test_manifest = config.SPLIT1_TEST_5 if args.config == '5percent' else config.SPLIT1_TEST_50
    else:
        test_manifest = config.SPLIT2_TEST_5 if args.config == '5percent' else config.SPLIT2_TEST_50
    
    print("\n" + "="*70)
    print("📊 EVALUATING RAW HUBERT (NO FINE-TUNING)")
    print("="*70)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Model: HuBERT-Large (Frozen, No Fine-tuning)")
    print(f"Batch size: {config.BATCH_SIZE}")
    
    if not os.path.exists(test_manifest):
        print(f"\n❌ Error: Test manifest not found: {test_manifest}")
        sys.exit(1)
    
    # Load dataset
    print("\n📂 Loading test dataset...")
    test_dataset = HDSINDataset(test_manifest, config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load HuBERT
    print(f"\n🔧 Loading Raw HuBERT...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    extractor = RawHuBERTExtractor(device=device)
    model_name = "HuBERT-Large (Raw, No Fine-tuning)"
    
    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    embeddings = []
    speaker_ids = []
    languages = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            audio = batch['audio'].to(config.DEVICE)
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            
            batch_emb = extractor.encode_batch(audio)
            batch_emb = batch_emb.squeeze(1)
            
            embeddings.append(batch_emb.cpu())
            speaker_ids.extend(batch['speaker_id'])
            languages.extend(batch['language_name'])
    
    embeddings = torch.cat(embeddings, dim=0)
    n_files = len(embeddings)
    print(f"\n✅ Extracted {n_files} embeddings, dim={embeddings.shape[1]}")
    
    # Group by speaker
    print("\n📊 Grouping by speaker...")
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(i)
    
    print(f"   Unique speakers: {len(speaker_to_indices)}")
    
    # Create trials
    print("\n📊 Creating evaluation trials...")
    
    # Genuine trials
    print("   Creating genuine trials...")
    genuine_scores = []
    for spk, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            n_trials = min(len(indices), 100)
            for i in range(n_trials):
                for j in range(i+1, n_trials):
                    score = F.cosine_similarity(
                        embeddings[indices[i]].unsqueeze(0),
                        embeddings[indices[j]].unsqueeze(0)
                    ).item()
                    genuine_scores.append(score)
    
    print(f"   Genuine trials: {len(genuine_scores)}")
    
    # Impostor trials
    print("   Creating impostor trials...")
    impostor_scores = []
    speakers_list = list(speaker_to_indices.keys())
    max_trials = min(args.max_trials, len(speakers_list) * 50)
    
    speakers_array = np.array(speakers_list)
    
    for _ in tqdm(range(max_trials), desc="   Creating impostor trials"):
        spk1, spk2 = np.random.choice(speakers_array, size=2, replace=False)
        idx1 = np.random.choice(speaker_to_indices[spk1])
        idx2 = np.random.choice(speaker_to_indices[spk2])
        score = F.cosine_similarity(
            embeddings[idx1].unsqueeze(0),
            embeddings[idx2].unsqueeze(0)
        ).item()
        impostor_scores.append(score)
    
    print(f"   Impostor trials: {len(impostor_scores)}")
    
    # Compute results
    print("\n📊 Computing results...")
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Per-language results
    per_lang_results = compute_per_language_eer(embeddings, speaker_ids, languages)
    
    # Print results
    print("\n" + "="*70)
    print(f"📈 {model_name} RESULTS")
    print("="*70)
    print(f"\n{'Overall Results':=^50}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Threshold at EER: {threshold:.4f}")
    print(f"\nGenuine scores: mean={np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"Impostor scores: mean={np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    
    if per_lang_results:
        print(f"\n{'Per-Language Results':=^50}")
        for lang, res in per_lang_results.items():
            print(f"\n{lang.capitalize()}:")
            print(f"  EER: {res['eer']:.2f}%")
            print(f"  AUC: {res['auc']:.4f}")
            print(f"  Genuine mean: {res['genuine_mean']:.4f}")
            print(f"  Impostor mean: {res['impostor_mean']:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, f"hubert_raw_split{args.split}_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"Raw HuBERT (No Fine-tuning) Results\n")
        f.write(f"====================================\n\n")
        f.write(f"Dataset: {args.config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Overall EER: {eer*100:.2f}%\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Genuine mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}\n")
        f.write(f"Impostor mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}\n\n")
        
        if per_lang_results:
            f.write("Per-Language Results:\n")
            f.write("--------------------\n")
            for lang, res in per_lang_results.items():
                f.write(f"\n{lang.capitalize()}:\n")
                f.write(f"  EER: {res['eer']:.2f}%\n")
                f.write(f"  AUC: {res['auc']:.4f}\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate plot
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Score Distribution
    axes[0, 0].hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', density=True, color='green')
    axes[0, 0].hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', density=True, color='red')
    axes[0, 0].axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'Raw HuBERT Score Distribution (EER: {eer*100:.2f}%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ROC Curve
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].scatter(eer, 1-eer, color='red', s=100, label=f'EER = {eer*100:.2f}%')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # DET Curve
    fnr = 1 - tpr
    axes[1, 0].plot(fpr, fnr, linewidth=2, color='purple')
    axes[1, 0].scatter(eer, eer, color='red', s=100, label=f'EER = {eer*100:.2f}%')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('False Negative Rate')
    axes[1, 0].set_title('DET Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Per-Language EER
    if per_lang_results:
        langs = list(per_lang_results.keys())
        eers = [per_lang_results[lang]['eer'] for lang in langs]
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[1, 1].bar(langs, eers, color=colors_bar[:len(langs)])
        axes[1, 1].set_xlabel('Language')
        axes[1, 1].set_ylabel('EER (%)')
        axes[1, 1].set_title('Per-Language EER (Raw HuBERT)')
        for bar, val in zip(bars, eers):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.2f}%', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_plot = os.path.join(args.output_dir, f"hubert_raw_split{args.split}_results.png")
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_plot}")
    plt.close()
    
    return {
        'model': 'hubert_raw',
        'split': args.split,
        'eer': eer * 100,
        'auc': roc_auc,
        'per_language': per_lang_results
    }


if __name__ == "__main__":
    results = evaluate()
    
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - RAW HUBERT")
    print("="*70)
    print(f"Split: {results['split']}")
    print(f"EER: {results['eer']:.2f}%")
    print(f"AUC: {results['auc']:.4f}")