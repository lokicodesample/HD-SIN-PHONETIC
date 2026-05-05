# ============================================
# dataset.py - Dataset classes for all architectures
# ============================================

import torch
from torch.utils.data import Dataset
import json
import soundfile as sf
import numpy as np
import random
from collections import defaultdict

class SpeakerDataset(Dataset):
    """Dataset for Architecture A (classification)"""
    
    def __init__(self, manifest_path, config):
        self.config = config
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Create speaker to index mapping
        self.speaker_to_idx = {}
        for item in self.data:
            if item['speaker_id'] not in self.speaker_to_idx:
                self.speaker_to_idx[item['speaker_id']] = len(self.speaker_to_idx)
        
        print(f"📊 Dataset loaded: {len(self.data)} files, {len(self.speaker_to_idx)} speakers")
    
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
            'label': self.speaker_to_idx[item['speaker_id']],
            'speaker_id': item['speaker_id']
        }


class TripletDataset(Dataset):
    """Dataset for Architectures B and C (triplet loss)"""
    
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
        
        print(f"📊 Triplet dataset: {len(self.data)} files, {len(self.valid_speakers)} valid speakers")
        self.num_triplets = len(self.valid_speakers) * 20  # Generate multiple triplets
    
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
            'negative': load_audio(negative_item)
        }