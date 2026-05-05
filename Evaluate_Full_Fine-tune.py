#!/usr/bin/env python3
# ============================================
# Evaluate_Full_Fine-tune.py - Evaluate Architecture C
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
import json
import time
from collections import defaultdict

from config import config
from architecture_c import ArchitectureC


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


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_trials', type=int, default=50000)
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
    print(f"📊 EVALUATING ARCHITECTURE C")
    print("="*60)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load dataset
    print("\n📂 Loading test dataset...")
    test_dataset = EvalDataset(test_manifest, config)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Load model
    print("\n🔧 Loading model...")
    model = ArchitectureC().to(config.DEVICE)
    
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
    languages = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            audio = batch['audio'].to(config.DEVICE)
            emb = model(audio)
            embeddings.append(emb.cpu())
            speaker_ids.extend(batch['speaker_id'])
            languages.extend(batch['language'])
    
    embeddings = torch.cat(embeddings, dim=0)
    print(f"✅ Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    
    # Group by speaker
    speaker_to_indices = defaultdict(list)
    lang_to_indices = defaultdict(list)
    
    for i, (spk, lang) in enumerate(zip(speaker_ids, languages)):
        speaker_to_indices[spk].append(i)
        lang_to_indices[lang].append(i)
    
    print(f"   Unique speakers: {len(speaker_to_indices)}")
    print(f"   Languages: {list(lang_to_indices.keys())}")
    
    # Overall trials
    print("\n📊 Creating evaluation trials...")
    
    # Genuine trials
    genuine_scores = []
    for spk, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            n_trials = min(len(indices), 50)
            for i in range(n_trials):
                for j in range(i+1, n_trials):
                    score = F.cosine_similarity(
                        embeddings[indices[i]].unsqueeze(0), 
                        embeddings[indices[j]].unsqueeze(0)
                    ).item()
                    genuine_scores.append(score)
    
    # Impostor trials
    impostor_scores = []
    speakers_list = list(speaker_to_indices.keys())
    max_trials = min(args.max_trials, len(speakers_list) * 50)
    
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
    
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    eer, _ = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   =========================")
    print(f"   EER: {eer*100:.2f}%")
    print(f"   AUC: {roc_auc:.4f}")
    print(f"   Genuine scores: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"   Impostor scores: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
    print(f"   =========================")
    
    # Per-language results
    print(f"\n📈 PER-LANGUAGE RESULTS:")
    for lang in ['tamil', 'telugu', 'english']:
        if lang not in lang_to_indices:
            continue
        
        lang_indices = set(lang_to_indices[lang])
        
        # Filter genuine and impostor for this language
        lang_genuine = []
        for spk, indices in speaker_to_indices.items():
            # Get speaker's language (use first file)
            spk_lang = None
            for idx in indices[:1]:
                if idx in lang_indices:
                    spk_lang = lang
                    break
            
            if spk_lang == lang and len(indices) >= 2:
                n_trials = min(len(indices), 30)
                for i in range(n_trials):
                    for j in range(i+1, n_trials):
                        if indices[i] in lang_indices and indices[j] in lang_indices:
                            score = F.cosine_similarity(
                                embeddings[indices[i]].unsqueeze(0), 
                                embeddings[indices[j]].unsqueeze(0)
                            ).item()
                            lang_genuine.append(score)
        
        # Impostor for this language
        lang_impostor = []
        other_speakers = [s for s in speakers_list if s not in speaker_to_indices]
        for _ in range(min(10000, len(speakers_list) * 10)):
            spk1 = np.random.choice(speakers_list)
            spk2 = np.random.choice([s for s in speakers_list if s != spk1])
            
            # Check if either speaker is from this language
            idx1 = np.random.choice(speaker_to_indices[spk1])
            idx2 = np.random.choice(speaker_to_indices[spk2])
            
            if (idx1 in lang_indices) != (idx2 in lang_indices):
                score = F.cosine_similarity(
                    embeddings[idx1].unsqueeze(0), 
                    embeddings[idx2].unsqueeze(0)
                ).item()
                lang_impostor.append(score)
        
        if lang_genuine and lang_impostor:
            all_scores_lang = lang_genuine + lang_impostor
            all_labels_lang = [1] * len(lang_genuine) + [0] * len(lang_impostor)
            lang_eer, _ = compute_eer(all_labels_lang, all_scores_lang)
            
            print(f"\n   {lang.capitalize()}:")
            print(f"      EER: {lang_eer*100:.2f}%")
            print(f"      Genuine mean: {np.mean(lang_genuine):.4f}")
            print(f"      Impostor mean: {np.mean(lang_impostor):.4f}")
            print(f"      Trials: genuine={len(lang_genuine)}, impostor={len(lang_impostor)}")
    
    # Save results
    output_file = f"results_c_split{args.split}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Architecture C Results\n")
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