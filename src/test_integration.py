"""
Integration Test: Raid's DataLoader + Moeed's Feature Extractor
---------------------------------------------------------------
This script verifies that the raw AmbiStory JSON data can be successfully 
tokenized by Raid's code, passed into Moeed's LoRA-DeBERTa model, 
and output the correct dense vectors for Areeb's BiLSTM.
"""

import torch
from data_loader import build_dataloader, build_tokenizer
from feature_extraction import HierarchicalDebertaEncoder

def run_integration_test():
    print("=== 1. Initializing Raid's Tokenizer & DataLoader ===")
    
    # We use DeBERTa V1 tokenizer to match our model
    tokenizer = build_tokenizer("microsoft/deberta-large")
    
    # Pointing to the local AmbiStory JSON file. Batch size is 8 as per the literature.
    data_path = "data/train.json"
    dataloader = build_dataloader(
        data_path=data_path, 
        tokenizer=tokenizer, 
        batch_size=8
    )
    
    print("DataLoader built successfully! Fetching the first batch of real data...")
    # Fetch a single batch of real data
    batch = next(iter(dataloader))
    print(f"Batch loaded! Target scores for this batch: {batch['target_score'].tolist()}")
    
    print("\n=== 2. Initializing Moeed's Hierarchical DeBERTa Encoder ===")
    encoder = HierarchicalDebertaEncoder()
    
    print("\n=== 3. Running the Forward Pass ===")
    outputs = encoder(batch)
    
    print("\nINTEGRATION SUCCESSFUL! Final shapes ready for Areeb's BiLSTM:")
    for key, tensor in outputs.items():
        print(f" -> {key}: {tensor.shape}")

if __name__ == "__main__":
    run_integration_test()