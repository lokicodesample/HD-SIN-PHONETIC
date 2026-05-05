#!/usr/bin/env python3
# ============================================
# evaluate_indicwav2vec.py - Updated with correct model IDs
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


class IndicWav2VecExtractor:
    """Extract embeddings from IndicWav2Vec for speaker verification"""
    
    def __init__(self, language, device='cuda'):
        self.device = device
        self.language = language
        
        # Updated model IDs - Use multilingual model for all languages
        # Option 1: Use the multilingual IndicWav2Vec model
        # Option 2: Use IndicConformer (newer)
        
        # Try different model IDs in order
        model_options = {
            'multilingual': 'ai4bharat/indicwav2vec',
            'tamil': 'ai4bharat/indicwav2vec-tamil',
            'telugu': 'ai4bharat/indicwav2vec-telugu',
            'indicconformer': 'ai4bharat/IndicConformer'
        }
        
        # Use multilingual model as fallback
        model_id = 'ai4bharat/indicwav2vec'  # Multilingual model
        
        print(f"🔄 Loading {model_id} for {language}...")
        
        try:
            self.model = AutoModelForCTC.from_pretrained(model_id).to(device)
            self.processor = AutoProcessor.from_pretrained(model_id)
        except:
            print(f"⚠️ Falling back to Wav2Vec2-large...")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"✅ Model loaded, hidden size: {self.hidden_size}")
    
    def encode_batch(self, audio):
        """Extract embeddings using mean pooling"""
        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.unsqueeze(1)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['5percent', '50percent'])
    parser.add_argument('--split', type=int, required=True, choices=[1, 2])
    parser.add_argument('--language', type=str, required=True, choices=['tamil', 'telugu', 'english'])
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
    print(f"📊 EVALUATING INDICWAV2VEC ({args.language.upper()})")
    print("="*70)
    print(f"Dataset: {args.config}")
    print(f"Split: {args.split} ({'Identification' if args.split==1 else 'Open-set'})")
    print(f"Language: {args.language}")
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
    
    # Language distribution
    lang_counts = defaultdict(int)
    for lang in languages:
        lang_counts[lang] += 1
    print(f"   Language distribution: {dict(lang_counts)}")
    
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
    print("\n📊 Computing overall results...")
    all_scores = genuine_scores + impostor_scores
    all_labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Filter for target language only
    print(f"\n📊 Computing results for {args.language}...")
    lang_indices = [i for i, lang in enumerate(languages) if lang == args.language]
    
    if len(lang_indices) > 10:
        lang_embeddings = embeddings[lang_indices]
        lang_speakers = [speaker_ids[i] for i in lang_indices]
        
        lang_speaker_to_idx = defaultdict(list)
        for i, spk in enumerate(lang_speakers):
            lang_speaker_to_idx[spk].append(i)
        
        lang_genuine = []
        for spk, indices in lang_speaker_to_idx.items():
            if len(indices) >= 2:
                for i in range(min(len(indices), 50)):
                    for j in range(i+1, min(len(indices), 50)):
                        score = F.cosine_similarity(
                            lang_embeddings[indices[i]].unsqueeze(0),
                            lang_embeddings[indices[j]].unsqueeze(0)
                        ).item()
                        lang_genuine.append(score)
        
        lang_impostor = []
        speakers_list = list(lang_speaker_to_idx.keys())
        for _ in range(min(10000, len(speakers_list) * 20)):
            spk1 = np.random.choice(speakers_list)
            spk2 = np.random.choice([s for s in speakers_list if s != spk1])
            idx1 = np.random.choice(lang_speaker_to_idx[spk1])
            idx2 = np.random.choice(lang_speaker_to_idx[spk2])
            score = F.cosine_similarity(
                lang_embeddings[idx1].unsqueeze(0),
                lang_embeddings[idx2].unsqueeze(0)
            ).item()
            lang_impostor.append(score)
        
        if lang_genuine and lang_impostor:
            lang_scores = lang_genuine + lang_impostor
            lang_labels = [1] * len(lang_genuine) + [0] * len(lang_impostor)
            lang_eer, _ = compute_eer(lang_labels, lang_scores)
            lang_fpr, lang_tpr, _ = roc_curve(lang_labels, lang_scores)
            lang_auc = auc(lang_fpr, lang_tpr)
        else:
            lang_eer = None
            lang_auc = None
    else:
        print(f"⚠️ Not enough samples for {args.language}")
        lang_eer = None
        lang_auc = None
    
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
    
    if lang_eer is not None:
        print(f"\n{args.language.capitalize()}-Specific Results:")
        print(f"  EER: {lang_eer:.2f}%")
        print(f"  AUC: {lang_auc:.4f}")
    
    # Save results
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
        
        if lang_eer is not None:
            f.write(f"\n{args.language.capitalize()}-Specific Results:\n")
            f.write(f"  EER: {lang_eer:.2f}%\n")
            f.write(f"  AUC: {lang_auc:.4f}\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return {
        'model': f'indicwav2vec_{args.language}',
        'split': args.split,
        'eer': eer * 100,
        'auc': roc_auc,
        'language_eer': lang_eer if lang_eer else None
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