# create_standalone_5percent_dataset.py
# Run this on your current system to extract only needed audio files

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths on your current system
SOURCE_AUDIO = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/audio"
SOURCE_MANIFESTS = "/home/ladmin/new_env_nemo/my_local_data/processed_dataset/manifests_subsets"

# Destination for standalone 5% dataset
DEST_BASE = "/home/ladmin/new_env_nemo/my_local_data/5percent_dataset_standalone"
DEST_AUDIO = os.path.join(DEST_BASE, "audio")
DEST_MANIFESTS = os.path.join(DEST_BASE, "manifests_subsets")

# Create directories
os.makedirs(DEST_AUDIO, exist_ok=True)
os.makedirs(DEST_MANIFESTS, exist_ok=True)
os.makedirs(os.path.join(DEST_AUDIO, "tamil"), exist_ok=True)
os.makedirs(os.path.join(DEST_AUDIO, "telugu"), exist_ok=True)
os.makedirs(os.path.join(DEST_AUDIO, "english"), exist_ok=True)

print("="*60)
print("📦 Creating Standalone 5% Dataset")
print("="*60)

# Track all audio files we need to copy
needed_audio_files = set()

# First, read all JSONL files to find needed audio files
print("\n📋 Reading JSONL files to find required audio files...")
jsonl_files = [
    "5percent_split1_train.jsonl",
    "5percent_split1_val.jsonl",
    "5percent_split1_test.jsonl",
    "5percent_split2_train.jsonl",
    "5percent_split2_val.jsonl",
    "5percent_split2_test.jsonl"
]

for jsonl_file in jsonl_files:
    src_path = os.path.join(SOURCE_MANIFESTS, jsonl_file)
    if os.path.exists(src_path):
        with open(src_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                audio_path = data['audio_path']
                needed_audio_files.add(audio_path)
        print(f"  ✅ {jsonl_file}: found entries")

print(f"\n📊 Total unique audio files needed: {len(needed_audio_files):,}")

# Copy audio files
print("\n🎵 Copying audio files to standalone folder...")
copied_count = 0
missing_count = 0

for audio_path in tqdm(needed_audio_files, desc="Copying audio"):
    if os.path.exists(audio_path):
        # Determine destination path (preserve folder structure)
        # audio_path example: /home/.../audio/tamil/tamil_spk_xxx.wav
        rel_path = audio_path.split("/audio/")[-1]  # "tamil/tamil_spk_xxx.wav"
        dest_path = os.path.join(DEST_AUDIO, rel_path)
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(audio_path, dest_path)
        copied_count += 1
    else:
        print(f"  ⚠️ Missing: {audio_path}")
        missing_count += 1

print(f"\n✅ Copied: {copied_count:,} audio files")
print(f"⚠️ Missing: {missing_count}")

# Copy and update JSONL files (update paths to new location)
print("\n📝 Creating updated JSONL files with new paths...")

for jsonl_file in jsonl_files:
    src_path = os.path.join(SOURCE_MANIFESTS, jsonl_file)
    dest_path = os.path.join(DEST_MANIFESTS, jsonl_file)
    
    if not os.path.exists(src_path):
        continue
    
    with open(src_path, 'r') as f_in:
        with open(dest_path, 'w') as f_out:
            for line in f_in:
                data = json.loads(line)
                old_path = data['audio_path']
                
                # Create new path
                rel_path = old_path.split("/audio/")[-1]
                new_path = os.path.join(DEST_BASE, "audio", rel_path)
                data['audio_path'] = new_path
                
                f_out.write(json.dumps(data) + '\n')
    
    print(f"  ✅ Updated: {jsonl_file}")

# Calculate size
print("\n📊 Calculating final size...")
total_size = 0
for root, dirs, files in os.walk(DEST_AUDIO):
    for f in files:
        filepath = os.path.join(root, f)
        total_size += os.path.getsize(filepath)

size_gb = total_size / (1024**3)
print(f"✅ Standalone dataset size: {size_gb:.2f} GB")
print(f"   Audio files: {copied_count:,}")
print(f"   JSONL files: 6")

print("\n" + "="*60)
print("🎉 Standalone 5% dataset created at:")
print(f"   {DEST_BASE}")
print("="*60)
print("\nNow you can copy ONLY this folder to your DGX server!")
print(f"   tar -czf 5percent_dataset_standalone.tar.gz {DEST_BASE}")