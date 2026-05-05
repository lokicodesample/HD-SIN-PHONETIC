# ============================================
# architectures.py - All baseline architectures
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig

class ArchitectureA(nn.Module):
    """Architecture A: Classifier Fine-tuning"""
    
    def __init__(self, config):
        super().__init__()
        
        print("🔄 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        
        # Freeze all WavLM parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable components
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.NUM_SPEAKERS)
        )
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Architecture A: {self.trainable_params:,} trainable params")
    
    def forward(self, audio):
        outputs = self.wavlm(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        
        # Weighted sum of layers
        weighted_sum = torch.zeros_like(hidden_states[-1])
        for i, hidden in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * hidden
        
        # Mean pooling
        pooled = weighted_sum.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits, pooled


class ArchitectureB(nn.Module):
    """Architecture B: PEFT Metric Learning"""
    
    def __init__(self, config):
        super().__init__()
        
        print("🔄 Loading WavLM-large (frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        
        # Freeze all WavLM parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable components
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        self.embedding_proj = nn.Linear(hidden_size, config.EMBEDDING_DIM)
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Architecture B: {self.trainable_params:,} trainable params")
    
    def forward(self, audio):
        outputs = self.wavlm(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        
        # Weighted sum
        weighted_sum = torch.zeros_like(hidden_states[-1])
        for i, hidden in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * hidden
        
        # Mean pooling
        pooled = weighted_sum.mean(dim=1)
        
        # Project and normalize
        embedding = self.embedding_proj(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class ArchitectureC(nn.Module):
    """Architecture C: Full Model Fine-tuning"""
    
    def __init__(self, config, unfrozen_layers=3):
        super().__init__()
        
        print("🔄 Loading WavLM-large (partially frozen)...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        
        # Freeze all first
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers
        if hasattr(self.wavlm, 'encoder') and hasattr(self.wavlm.encoder, 'layers'):
            total_layers = len(self.wavlm.encoder.layers)
            for i in range(total_layers - unfrozen_layers, total_layers):
                for param in self.wavlm.encoder.layers[i].parameters():
                    param.requires_grad = True
            print(f"   Unfroze top {unfrozen_layers} of {total_layers} layers")
        
        # Get hidden size
        wavlm_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        hidden_size = wavlm_config.hidden_size
        
        # Trainable projection
        self.embedding_proj = nn.Linear(hidden_size, config.EMBEDDING_DIM)
        
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Architecture C: {self.trainable_params:,} / {self.total_params:,} trainable ({100.*self.trainable_params/self.total_params:.2f}%)")
    
    def forward(self, audio):
        outputs = self.wavlm(audio)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        pooled = last_hidden.mean(dim=1)
        
        # Project and normalize
        embedding = self.embedding_proj(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding 