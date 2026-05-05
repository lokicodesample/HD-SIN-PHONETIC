#!/usr/bin/env python3
# ============================================
# evaluate_indicwav2vec.py - Evaluate IndicWav2Vec for speaker verification
# Usage:
#   python evaluate_indicwav2vec.py --config 5percent --split 1 --language tamil
#   python evaluate_indicwav2vec.py --config 5percent --split 2 --language telugu
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

from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
from config import config
from dataset_hdsin import HDSINDataset


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


def compute_per_language_eer(embeddings, speaker_ids, languages, target_language):
    """Compute EER for specific language"""
    # Filter for target language
    lang_indices = [i for i, lang in enumerate(languages) if lang == target_language]
    
    if len(lang_indices) < 10:
        print(f"⚠️ Not enough samples for {target_language} ({len(lang_indices)} files)")
        return None
    
    lang_embeddings = embeddings[lang_indices]
    lang_speakers = [speaker_ids[i] for i in lang_indices]
    
    genuine_scores = []
    impostor_scores = []
    
    # Group by speaker
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
        
        return {
            'eer': eer * 100,
            'auc': roc_auc,
            'genuine_count': len(genuine_scores),
            'impostor_count': len(impostor_scores),
            'genuine_mean': np.mean(genuine_scores),
            'impostor_mean': np.mean(impostor_scores)
        }
    
    return None


class IndicWav2VecExtractor:
    """Extract embeddings from IndicWav2Vec for speaker verification"""
    
    def __init__(self, language, device='cuda'):
        self.device = device
        self.language = language
        
        # Map language to model ID
        model_map = {
            'tamil': 'ai4bharat/indicwav2vec-tamil',
            'telugu': 'ai4bharat/indicwav2vec-telugu',
            'english': 'facebook/wav2vec2-large-960h'
        }
        
        model_id = model_map.get(language, 'ai4bharat/indicwav2vec-tamil')
        print(f"🔄 Loading {model_id}...")
        
        try:
            self.model = AutoModelForCTC.from_pretrained(model_id).to(device)
            self.processor = AutoProcessor.from_pretrained(model_id)
        except:
            # Fallback for older models
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"✅ Model loaded, hidden size: {self.hidden_size}")
    
    def encode_batch(self, audio):
        """
        Extract embeddings using mean pooling
        audio: [batch, time] tensor
        Returns: [batch, 1, hidden_size]
        """
        with torch.no_grad():
            # Get hidden states
            outputs = self.model(audio, output_hidden_states=True)
            # Use last hidden state
            last_hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            # Mean pooling over time
            embedding = last_hidden.mean(dim=1)
            # L2 normalize
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.unsqueeze(1)


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate IndicWav2Vec for speaker verification')
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'],
                        help='Use 5percent or 50percent dataset')
    parser.add_argument('--split', type=int, required=True, choices=[1, 2],
                        help='Split 1 (identification) or Split 2 (open-set)')
    parser.add_argument('--language', type=str, required=True, choices=['tamil', 'telugu', 'english'],
                        help='Language to evaluate (uses language-specific model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_trials', type=int, default=50000,
                        help='Maximum number of impostor trials')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save results')
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
    print(f"📊 EVALUATING INDICWAV2VEC ({args.language.upper()})")
    print("="*70)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Language: {args.language}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max trials: {args.max_trials}")
    
    # Check test manifest
    if not os.path.exists(test_manifest):
        print(f"\n❌ Error: Test manifest not found: {test_manifest}")
        sys.exit(1)
    
    # Load dataset
    print("\n📂 Loading test dataset...")
    test_dataset = HDSINDataset(test_manifest, config)
    
    # Filter for target language if needed (for language-specific evaluation)
    # Note: This creates a filtered dataset for the target language
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"\n🔧 Loading IndicWav2Vec model for {args.language}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    extractor = IndicWav2VecExtractor(language=args.language, device=device)
    model_name = f"IndicWav2Vec-{args.language.capitalize()}"
    
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
    
    # Get language distribution
    lang_counts = defaultdict(int)
    for lang in languages:
        lang_counts[lang] += 1
    print(f"   Language distribution: {dict(lang_counts)}")
    
    # Create trials (overall)
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
    print("\n📊 Computing overall results...")
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute per-language results for the target language
    print(f"\n📊 Computing results for {args.language}...")
    per_lang_result = compute_per_language_eer(embeddings, speaker_ids, languages, args.language)
    
    # Print overall results
    print("\n" + "="*70)
    print(f"📈 {model_name} RESULTS")
    print("="*70)
    print(f"\n{'Overall Results':=^50}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Threshold at EER: {threshold:.4f}")
    print(f"\nGenuine scores: mean={np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"Impostor scores: mean={np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    
    if per_lang_result:
        print(f"\n{'Language-Specific Results':=^50}")
        print(f"\n{args.language.capitalize()}:")
        print(f"  EER: {per_lang_result['eer']:.2f}%")
        print(f"  AUC: {per_lang_result['auc']:.4f}")
        print(f"  Genuine mean: {per_lang_result['genuine_mean']:.4f}")
        print(f"  Impostor mean: {per_lang_result['impostor_mean']:.4f}")
        print(f"  Trials: genuine={per_lang_result['genuine_count']}, impostor={per_lang_result['impostor_count']}")
    
    # Save results to file
    output_file = os.path.join(args.output_dir, f"indicwav2vec_{args.language}_split{args.split}_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"IndicWav2Vec-{args.language.capitalize()} Results\n")
        f.write(f"==============================================\n\n")
        f.write(f"Dataset: {args.config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"Overall EER: {eer*100:.2f}%\n")
        f.write(f"Overall AUC: {roc_auc:.4f}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Genuine mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}\n")
        f.write(f"Impostor mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}\n")
        
        if per_lang_result:
            f.write(f"\n{args.language.capitalize()}-Specific Results:\n")
            f.write(f"  EER: {per_lang_result['eer']:.2f}%\n")
            f.write(f"  AUC: {per_lang_result['auc']:.4f}\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate plots
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Score Distribution
    axes[0, 0].hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', density=True, color='green')
    axes[0, 0].hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', density=True, color='red')
    axes[0, 0].axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'{model_name} Score Distribution (EER: {eer*100:.2f}%)')
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
    
    # 4. Per-Language EER Bar Chart (if we have per-language results)
    if per_lang_result:
        langs = [args.language.capitalize()]
        eers = [per_lang_result['eer']]
        colors_bar = ['#45B7D1']
        bars = axes[1, 1].bar(langs, eers, color=colors_bar)
        axes[1, 1].set_xlabel('Language')
        axes[1, 1].set_ylabel('EER (%)')
        axes[1, 1].set_title(f'{model_name} - Per-Language EER')
        for bar, val in zip(bars, eers):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.2f}%', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_plot = os.path.join(args.output_dir, f"indicwav2vec_{args.language}_split{args.split}_results.png")
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_plot}")
    plt.close()
    
    return {
        'model': f'indicwav2vec_{args.language}',
        'split': args.split,
        'eer': eer * 100,
        'auc': roc_auc,
        'language_eer': per_lang_result['eer'] if per_lang_result else None
    }


if __name__ == "__main__":
    results = evaluate()
    
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - INDICWAV2VEC")
    print("="*70)
    print(f"Language: {results['model']}")
    print(f"Split: {results['split']}")
    print(f"Overall EER: {results['eer']:.2f}%")
    print(f"Overall AUC: {results['auc']:.4f}")
    if results['language_eer']:
        print(f"Language-specific EER: {results['language_eer']:.2f}%")