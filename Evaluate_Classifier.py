#!/usr/bin/env python3
# ============================================
# evaluate_wavlm_classifier.py - Evaluate WavLM + Classifier (Architecture A)
# with Per-Language Results
# Usage: 
#   python evaluate_wavlm_classifier.py --config 5percent --split 1 --checkpoint checkpoints/architecture_a_split1_best.pt
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

from config import config
from architecture_a import ArchitectureA


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


class EvalDataset:
    """Dataset for evaluation with language info"""
    def __init__(self, manifest_path, config):
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.config = config
        print(f"   Loaded {len(self.data)} files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import soundfile as sf
        import numpy as np
        import librosa
        
        item = self.data[idx]
        audio, sr = sf.read(item['audio_path'])
        if sr != self.config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        if len(audio) > self.config.MAX_LENGTH:
            audio = audio[:self.config.MAX_LENGTH]
        else:
            padding = self.config.MAX_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return {
            'audio': torch.FloatTensor(audio),
            'speaker_id': item['speaker_id'],
            'language': item.get('language', 'unknown')
        }


def compute_per_language_eer(embeddings, speaker_ids, languages):
    """Compute EER per language"""
    results = {}
    language_map = {'tamil': 0, 'telugu': 1, 'english': 2}
    
    for lang_name, lang_idx in language_map.items():
        # Get indices for this language
        lang_indices = [i for i, lang in enumerate(languages) if lang == lang_name]
        
        if len(lang_indices) < 10:
            print(f"⚠️ Not enough samples for {lang_name} ({len(lang_indices)} files)")
            continue
        
        # Get embeddings and speaker IDs for this language
        lang_embeddings = embeddings[lang_indices]
        lang_speakers = [speaker_ids[i] for i in lang_indices]
        
        # Create trials for this language
        genuine_scores = []
        impostor_scores = []
        
        # Group indices by speaker
        speaker_to_idx = defaultdict(list)
        for i, spk in enumerate(lang_speakers):
            speaker_to_idx[spk].append(i)
        
        # Genuine trials (same speaker)
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
        
        # Impostor trials (different speakers)
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


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_trials', type=int, default=50000)
    parser.add_argument('--num_speakers', type=int, default=None)
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
    
    print("\n" + "="*70)
    print("📊 EVALUATING WAVLM + CLASSIFIER (Architecture A)")
    print("="*70)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {config.BATCH_SIZE}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load dataset
    print("\n📂 Loading test dataset...")
    test_dataset = EvalDataset(test_manifest, config)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Get number of speakers from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    
    if args.num_speakers:
        num_speakers = args.num_speakers
    elif 'model_state_dict' in checkpoint:
        classifier_weight = checkpoint['model_state_dict']['classifier.3.weight']
        num_speakers = classifier_weight.shape[0]
    else:
        classifier_weight = checkpoint['classifier.3.weight']
        num_speakers = classifier_weight.shape[0]
    
    print(f"   Model expects {num_speakers} speakers")
    
    # Load model
    print("\n🔧 Loading model...")
    model = ArchitectureA(num_speakers).to(config.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded model weights")
    
    model = model.float()
    model.eval()
    
    # Extract embeddings using class probabilities
    print("\n🔍 Extracting embeddings...")
    embeddings = []
    speaker_ids = []
    languages = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            audio = batch['audio'].to(config.DEVICE)
            logits = model(audio)
            probs = F.softmax(logits, dim=1)
            embeddings.append(probs.cpu())
            speaker_ids.extend(batch['speaker_id'])
            languages.extend(batch['language'])
    
    embeddings = torch.cat(embeddings, dim=0)
    print(f"✅ Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    
    # Group by speaker
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(i)
    
    print(f"   Unique speakers: {len(speaker_to_indices)}")
    print(f"   Languages: {set(languages)}")
    
    # Overall trials
    print("\n📊 Creating evaluation trials...")
    
    # Genuine trials
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
    
    # Overall results
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   =========================")
    print(f"   EER: {eer*100:.2f}%")
    print(f"   AUC: {roc_auc:.4f}")
    print(f"   Genuine mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"   Impostor mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    print(f"   =========================")
    
    # Per-language results
    print(f"\n📈 PER-LANGUAGE RESULTS:")
    per_lang_results = compute_per_language_eer(embeddings, speaker_ids, languages)
    
    for lang, res in per_lang_results.items():
        print(f"\n{lang.capitalize()}:")
        print(f"   EER: {res['eer']:.2f}%")
        print(f"   AUC: {res['auc']:.4f}")
        print(f"   Genuine mean: {res['genuine_mean']:.4f}")
        print(f"   Impostor mean: {res['impostor_mean']:.4f}")
        print(f"   Trials: genuine={res['genuine_count']}, impostor={res['impostor_count']}")
    
    # Save results
    output_file = f"wavlm_classifier_split{args.split}_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"WavLM + Classifier (Architecture A) Results\n")
        f.write(f"==========================================\n\n")
        f.write(f"Dataset: {args.config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Overall EER: {eer*100:.2f}%\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Genuine mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}\n")
        f.write(f"Impostor mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}\n\n")
        
        f.write("Per-Language Results:\n")
        f.write("--------------------\n")
        for lang, res in per_lang_results.items():
            f.write(f"\n{lang.capitalize()}:\n")
            f.write(f"  EER: {res['eer']:.2f}%\n")
            f.write(f"  AUC: {res['auc']:.4f}\n")
            f.write(f"  Genuine mean: {res['genuine_mean']:.4f}\n")
            f.write(f"  Impostor mean: {res['impostor_mean']:.4f}\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Create visualization
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Score Distribution
    axes[0, 0].hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', density=True, color='green')
    axes[0, 0].hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', density=True, color='red')
    axes[0, 0].axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'WavLM+Classifier Score Distribution (EER: {eer*100:.2f}%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].scatter(eer, 1-eer, color='red', s=100, label=f'EER = {eer*100:.2f}%')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. DET Curve
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
    
    # 4. Per-Language EER Bar Chart
    if per_lang_results:
        langs = list(per_lang_results.keys())
        eers = [per_lang_results[lang]['eer'] for lang in langs]
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[1, 1].bar(langs, eers, color=colors_bar[:len(langs)])
        axes[1, 1].set_xlabel('Language')
        axes[1, 1].set_ylabel('EER (%)')
        axes[1, 1].set_title('Per-Language EER (WavLM+Classifier)')
        for bar, val in zip(bars, eers):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.2f}%', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_plot = f"wavlm_classifier_split{args.split}_results.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_plot}")
    plt.close()
    
    return {
        'model': 'wavlm_classifier',
        'split': args.split,
        'eer': eer * 100,
        'auc': roc_auc,
        'per_language': per_lang_results
    }


if __name__ == "__main__":
    results = evaluate()
    
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - WAVLM + CLASSIFIER")
    print("="*70)
    print(f"Split: {results['split']}")
    print(f"EER: {results['eer']:.2f}%")
    print(f"AUC: {results['auc']:.4f}")