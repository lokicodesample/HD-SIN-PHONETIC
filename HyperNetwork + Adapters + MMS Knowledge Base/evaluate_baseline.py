#!/usr/bin/env python3
# ============================================
# evaluate_baseline.py - Evaluate ECAPA-TDNN & X-Vector Baselines with AUC
# Usage: 
#   python evaluate_baseline.py --config 5percent --split 1 --model ecapa
#   python evaluate_baseline.py --config 5percent --split 2 --model xvector
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
from dataset_hdsin import HDSINDataset

# Try importing speechbrain
try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("⚠️ SpeechBrain not installed. Install with: pip install speechbrain")


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


def compute_per_language_eer(embeddings, speaker_ids, languages, language_map):
    """Compute EER per language"""
    results = {}
    
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
        
        # Genuine trials
        for spk, indices in speaker_to_idx.items():
            if len(indices) >= 2:
                for i in range(min(len(indices), 50)):
                    for j in range(i+1, min(len(indices), 50)):
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
            
            # Also compute AUC for this language
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


class RawWavLMExtractor:
    """Extract embeddings from frozen WavLM without any fine-tuning"""
    
    def __init__(self, device='cuda'):
        from transformers import WavLMModel
        
        print("🔄 Loading raw WavLM-large (no fine-tuning, frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.to(device)
        self.wavlm.eval()
        
        # Freeze all parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        self.device = device
        self.hidden_size = 1024
        print(f"✅ Raw WavLM loaded, hidden size: {self.hidden_size}")
        
    def encode_batch(self, audio):
        """
        Extract embeddings from raw WavLM using mean pooling
        audio: [batch, time] tensor
        Returns: [batch, 1, hidden_size] to match SpeechBrain format
        """
        with torch.no_grad():
            outputs = self.wavlm(audio, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.unsqueeze(1)


def evaluate_baseline():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'],
                        help='Use 5percent or 50percent dataset')
    parser.add_argument('--split', type=int, required=True, choices=[1, 2],
                        help='Split 1 (identification) or Split 2 (open-set)')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['ecapa', 'xvector', 'wavlm_raw'],
                        help='Model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_trials', type=int, default=50000,
                        help='Maximum number of impostor trials')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save results')
    args = parser.parse_args()
    
    if not SPEECHBRAIN_AVAILABLE and args.model in ['ecapa', 'xvector']:
        print("❌ SpeechBrain not installed. Please install: pip install speechbrain")
        sys.exit(1)
    
    # Override config
    config.BATCH_SIZE = args.batch_size
    
    # Set paths based on split
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
    if args.model == 'wavlm_raw':
        print("📊 EVALUATING RAW WAVLM (NO FINE-TUNING)")
    else:
        print(f"📊 EVALUATING BASELINE: {args.model.upper()}")
    print("="*70)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Model: {args.model.upper() if args.model != 'wavlm_raw' else 'Raw WavLM'}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max trials: {args.max_trials}")
    
    # Check if test manifest exists
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
    
    # Load model
    print(f"\n🔧 Loading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    if args.model == 'ecapa':
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
        model_name = "ECAPA-TDNN"
        emb_dim = 192
    elif args.model == 'xvector':
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            run_opts={"device": device}
        )
        model_name = "X-Vector"
        emb_dim = 512
    else:  # wavlm_raw
        classifier = RawWavLMExtractor(device=device)
        model_name = "WavLM-Large (Raw, No Fine-tuning)"
        emb_dim = 1024
    
    print(f"   Model loaded, embedding dimension: {emb_dim}")
    
    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    embeddings = []
    speaker_ids = []
    languages = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            audio = batch['audio'].to(config.DEVICE)
            
            # Ensure correct shape [batch, time]
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            
            # Extract embeddings
            batch_emb = classifier.encode_batch(audio)
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
    
    # Compute overall results
    print("\n📊 Computing results...")
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Per-language results
    language_map = {'tamil': 0, 'telugu': 1, 'english': 2}
    per_lang_results = compute_per_language_eer(embeddings, speaker_ids, languages, language_map)
    
    # Print final results
    print("\n" + "="*70)
    print(f"📈 {model_name} RESULTS")
    print("="*70)
    print(f"\n{'Overall Results':=^50}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Threshold at EER: {threshold:.4f}")
    print(f"\nGenuine scores: mean={np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"Impostor scores: mean={np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    print(f"Genuine trials: {len(genuine_scores)}")
    print(f"Impostor trials: {len(impostor_scores)}")
    
    if per_lang_results:
        print(f"\n{'Per-Language Results':=^50}")
        for lang, res in per_lang_results.items():
            print(f"\n{lang.capitalize()}:")
            print(f"  EER: {res['eer']:.2f}%")
            print(f"  AUC: {res['auc']:.4f}")
            print(f"  Genuine mean: {res['genuine_mean']:.4f}")
            print(f"  Impostor mean: {res['impostor_mean']:.4f}")
            print(f"  Trials: genuine={res['genuine_count']}, impostor={res['impostor_count']}")
    
    # Save results to file
    output_file = os.path.join(args.output_dir, f"{args.model}_split{args.split}_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"{model_name} Results\n")
        f.write(f"=====================\n\n")
        f.write(f"Dataset: {args.config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Overall EER: {eer*100:.2f}%\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Genuine mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}\n")
        f.write(f"Impostor mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}\n")
        f.write(f"Genuine trials: {len(genuine_scores)}\n")
        f.write(f"Impostor trials: {len(impostor_scores)}\n\n")
        
        if per_lang_results:
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
    axes[0, 0].set_title(f'Score Distribution (EER: {eer*100:.2f}%)')
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
        axes[1, 1].set_title(f'Per-Language EER ({model_name})')
        for bar, val in zip(bars, eers):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.2f}%', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_plot = os.path.join(args.output_dir, f"{args.model}_split{args.split}_results.png")
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_plot}")
    plt.close()
    
    # Return results summary
    return {
        'model': args.model,
        'split': args.split,
        'eer': eer * 100,
        'auc': roc_auc,
        'per_language': per_lang_results
    }


if __name__ == "__main__":
    results = evaluate_baseline()
    
    # Print final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Model: {results['model'].upper()}")
    print(f"Split: {results['split']}")
    print(f"EER: {results['eer']:.2f}%")
    print(f"AUC: {results['auc']:.4f}")