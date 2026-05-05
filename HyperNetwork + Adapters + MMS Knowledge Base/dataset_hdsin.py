# ============================================
# dataset_hdsin.py - Dataset for HD-SIN-Phonetic
# Includes language information for each file
# ============================================

import torch
from torch.utils.data import Dataset
import json
import soundfile as sf
import numpy as np
import random
from collections import defaultdict

class HDSINDataset(Dataset):
    """Dataset for HD-SIN-Phonetic with language information"""
    
    def __init__(self, manifest_path, config):
        self.config = config
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        # Create mappings
        self.speaker_to_idx = {}
        self.language_to_idx = {'tamil': 0, 'telugu': 1, 'english': 2}
        
        for item in self.data:
            if item['speaker_id'] not in self.speaker_to_idx:
                self.speaker_to_idx[item['speaker_id']] = len(self.speaker_to_idx)
        
        print(f"📊 HD-SIN Dataset: {len(self.data)} files")
        print(f"   Speakers: {len(self.speaker_to_idx)}")
        print(f"   Languages: {set(item['language'] for item in self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio, sr = sf.read(item['audio_path'])
        
        # Resample if needed
        if sr != self.config.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        
        # Pad or truncate
        if len(audio) > self.config.MAX_LENGTH:
            audio = audio[:self.config.MAX_LENGTH]
        else:
            padding = self.config.MAX_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        return {
            'audio': torch.FloatTensor(audio),
            'speaker_id': item['speaker_id'],
            'speaker_idx': self.speaker_to_idx[item['speaker_id']],
            'language': self.language_to_idx[item['language']],
            'language_name': item['language']
        }


class TripletHDSINDataset(Dataset):
    """Triplet dataset for HD-SIN with language info"""
    
    def __init__(self, manifest_path, config):
        self.config = config
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Group by speaker - only keep speakers with >=2 files
        self.speaker_to_files = defaultdict(list)
        for item in self.data:
            self.speaker_to_files[item['speaker_id']].append(item)
        
        self.valid_speakers = []
        for speaker, files in self.speaker_to_files.items():
            if len(files) >= 2:
                self.valid_speakers.append(speaker)
        
        # Language mapping
        self.language_to_idx = {'tamil': 0, 'telugu': 1, 'english': 2}
        
        print(f"📊 HD-SIN Triplet dataset: {len(self.data)} files, {len(self.valid_speakers)} valid speakers")
        self.num_triplets = len(self.valid_speakers) * 20
    
    def __len__(self):
        return self.num_triplets
    
    def __getitem__(self, idx):
        # Select anchor speaker
        anchor_speaker = random.choice(self.valid_speakers)
        anchor_files = self.speaker_to_files[anchor_speaker]
        
        # Select anchor and positive (different files)
        anchor_item, positive_item = random.sample(anchor_files, 2)
        
        # Select negative from different speaker
        negative_speaker = random.choice([s for s in self.valid_speakers if s != anchor_speaker])
        negative_item = random.choice(self.speaker_to_files[negative_speaker])
        
        # Load audio helper
        def load_audio(item):
            audio, sr = sf.read(item['audio_path'])
            if sr != self.config.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
            if len(audio) > self.config.MAX_LENGTH:
                audio = audio[:self.config.MAX_LENGTH]
            else:
                audio = np.pad(audio, (0, self.config.MAX_LENGTH - len(audio)), 'constant')
            return torch.FloatTensor(audio)
        
        return {
            'anchor': load_audio(anchor_item),
            'positive': load_audio(positive_item),
            'negative': load_audio(negative_item),
            'anchor_language': self.language_to_idx[anchor_item['language']],
            'positive_language': self.language_to_idx[positive_item['language']],
            'negative_language': self.language_to_idx[negative_item['language']],
            'anchor_speaker': anchor_speaker,
            'negative_speaker': negative_speaker
        }