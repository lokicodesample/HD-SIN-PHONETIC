#!/usr/bin/env python3
# ============================================
# evaluate_hdsin.py - Evaluate HD-SIN-Phonetic
# Usage: python evaluate_hdsin.py --config 5percent --split 1 --checkpoint checkpoints/hdsin_split1_best.pt
# ============================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import sys

from config import config
from dataset_hdsin import HDSINDataset
from architecture_hdsin import HDSINPhonetic


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
        speaker_to_idx = {}
        for i, spk in enumerate(lang_speakers):
            if spk not in speaker_to_idx:
                speaker_to_idx[spk] = []
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
            
            results[lang_name] = {
                'eer': eer * 100,
                'genuine_count': len(genuine_scores),
                'impostor_count': len(impostor_scores),
                'genuine_mean': np.mean(genuine_scores),
                'impostor_mean': np.mean(impostor_scores)
            }
    
    return results


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'],
                        help='Use 5percent or 50percent dataset')
    parser.add_argument('--split', type=int, required=True, choices=[1, 2],
                        help='Split 1 (identification) or Split 2 (open-set)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max_trials', type=int, default=50000,
                        help='Maximum number of trials for evaluation')
    args = parser.parse_args()
    
    # Override batch size
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
    
    print("\n" + "="*60)
    print("📊 EVALUATING HD-SIN-PHONETIC")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max trials: {args.max_trials}")
    
    # Check if manifest exists
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
    print("\n🔧 Loading model...")
    model = HDSINPhonetic(num_speakers=config.NUM_SPEAKERS).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')}, loss: {checkpoint.get('best_loss', 'unknown'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    # IMPORTANT: Convert model to float32 for evaluation
    model = model.float()
    model.eval()
    
    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    embeddings = []
    speaker_ids = []
    languages = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            audio = batch['audio'].to(config.DEVICE)
            language = batch['language'].to(config.DEVICE)
            
            # Forward pass
            emb, _ = model(audio, None, language)
            embeddings.append(emb.cpu())
            speaker_ids.extend(batch['speaker_id'])
            languages.extend(batch['language_name'])
    
    embeddings = torch.cat(embeddings, dim=0)
    n_files = len(embeddings)
    
    print(f"\n📊 Extracted {n_files} embeddings, dim={embeddings.shape[1]}")
    
    # Create trials
    print("\n📊 Creating evaluation trials...")
    
    # Group indices by speaker
    speaker_to_indices = {}
    for i, spk in enumerate(speaker_ids):
        if spk not in speaker_to_indices:
            speaker_to_indices[spk] = []
        speaker_to_indices[spk].append(i)
    
    print(f"   Unique speakers: {len(speaker_to_indices)}")
    
    # Genuine trials (same speaker)
    genuine_scores = []
    for spk, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            for i in range(min(len(indices), 100)):
                for j in range(i+1, min(len(indices), 100)):
                    score = F.cosine_similarity(
                        embeddings[indices[i]].unsqueeze(0),
                        embeddings[indices[j]].unsqueeze(0)
                    ).item()
                    genuine_scores.append(score)
    
    print(f"   Genuine trials: {len(genuine_scores)}")
    
    # Impostor trials (different speakers)
    impostor_scores = []
    speakers_list = list(speaker_to_indices.keys())
    max_trials = min(args.max_trials, len(speakers_list) * 20)
    
    for _ in tqdm(range(max_trials), desc="Creating impostor trials"):
        spk1 = np.random.choice(speakers_list)
        spk2 = np.random.choice([s for s in speakers_list if s != spk1])
        idx1 = np.random.choice(speaker_to_indices[spk1])
        idx2 = np.random.choice(speaker_to_indices[spk2])
        score = F.cosine_similarity(
            embeddings[idx1].unsqueeze(0),
            embeddings[idx2].unsqueeze(0)
        ).item()
        impostor_scores.append(score)
    
    print(f"   Impostor trials: {len(impostor_scores)}")
    
    # Compute overall EER
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute per-language EER
    language_map = {'tamil': 0, 'telugu': 1, 'english': 2}
    per_lang_results = compute_per_language_eer(embeddings, speaker_ids, languages, language_map)
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Overall Results':=^50}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Threshold at EER: {threshold:.4f}")
    print(f"\nGenuine scores: mean={np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"Impostor scores: mean={np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    
    if per_lang_results:
        print(f"\n{'Per-Language Results':=^50}")
        for lang, results in per_lang_results.items():
            print(f"\n{lang.capitalize()}:")
            print(f"  EER: {results['eer']:.2f}%")
            print(f"  Genuine mean: {results['genuine_mean']:.4f}")
            print(f"  Impostor mean: {results['impostor_mean']:.4f}")
            print(f"  Trials: genuine={results['genuine_count']}, impostor={results['impostor_count']}")
    
    # Create visualizations
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Score Distribution Histogram
    axes[0, 0].hist(genuine_scores, bins=50, alpha=0.5, 
                    label=f'Genuine (n={len(genuine_scores)})', density=True, color='green')
    axes[0, 0].hist(impostor_scores, bins=50, alpha=0.5, 
                    label=f'Impostor (n={len(impostor_scores)})', density=True, color='red')
    axes[0, 0].axvline(threshold, color='blue', linestyle='--', 
                       label=f'EER threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'Score Distribution (EER: {eer*100:.2f}%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].scatter(eer, 1-eer, color='red', s=100, 
                       label=f'EER = {eer*100:.2f}%', zorder=5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. DET Curve (Detection Error Tradeoff)
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
        axes[1, 1].set_title('Per-Language Equal Error Rate')
        for bar, val in zip(bars, eers):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{val:.2f}%', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f'hdsin_split{args.split}_results.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_filename}")
    
    plt.show()
    
    # Save results to file
    results_txt = f"""============================================
HD-SIN-PHONETIC EVALUATION RESULTS
============================================

Configuration:
  Dataset: {args.config}
  Split: {args.split}
  Checkpoint: {args.checkpoint}
  Batch size: {config.BATCH_SIZE}

Overall Results:
  EER: {eer*100:.2f}%
  AUC: {roc_auc:.4f}
  Threshold: {threshold:.4f}
  
  Genuine Scores: mean={np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}
  Impostor Scores: mean={np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}
  
  Genuine Trials: {len(genuine_scores)}
  Impostor Trials: {len(impostor_scores)}

"""

    if per_lang_results:
        results_txt += "\nPer-Language Results:\n"
        for lang, res in per_lang_results.items():
            results_txt += f"""
{lang.capitalize()}:
  EER: {res['eer']:.2f}%
  Genuine mean: {res['genuine_mean']:.4f}
  Impostor mean: {res['impostor_mean']:.4f}
  Genuine trials: {res['genuine_count']}
  Impostor trials: {res['impostor_count']}
"""

    results_txt += "\n============================================\n"
    
    results_file = f'hdsin_split{args.split}_results.txt'
    with open(results_file, 'w') as f:
        f.write(results_txt)
    print(f"✅ Results saved to: {results_file}")
    
    print("\n✅ Evaluation complete!")
    
    return {
        'eer': eer,
        'auc': roc_auc,
        'threshold': threshold,
        'per_language': per_lang_results,
        'genuine_mean': np.mean(genuine_scores),
        'impostor_mean': np.mean(impostor_scores)
    }


if __name__ == "__main__":
    results = evaluate()
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Overall EER: {results['eer']*100:.2f}%")
    print(f"Overall AUC: {results['auc']:.4f}")
    if results.get('per_language'):
        for lang, res in results['per_language'].items():
            print(f"{lang.capitalize()} EER: {res['eer']:.2f}%")