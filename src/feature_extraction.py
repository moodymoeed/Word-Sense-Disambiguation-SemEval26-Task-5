"""Feature extraction module for the CS445 Group 2 SemEval system.

Project Phase: Step 2 - Disentangled Feature Extraction
-------------------------------------------------------
This file serves as the Comprehension Engine of our architecture. 
Upstream, `data_loader.py` parses the AmbiStory JSON files and converts text into 
integer IDs. This file takes those raw integer IDs and translates them into 
rich, dense mathematical vectors (embeddings) that capture deep semantic meaning.
Downstream, these dense vectors are passed through BiLSTM.

Architectural Reasoning:
------------------------
1. Why DeBERTa-large? 
   We chose DeBERTa because of its Disentangled Attention mechanism, which separately 
   encodes word content and relative word position. This is critical for Word Sense 
   Disambiguation (WSD) where the distance between a homonym and its context clues matters.
2. Why LoRA? 
   DeBERTa-large has ~350M parameters. Training it fully would cause our GPUs to crash 
   with Out-Of-Memory (OOM) errors. We are applying Low-Rank Adaptation (LoRA) to freeze the 
   base model and only train a few million injected parameters in the attention layers.
3. Why Independent Encoding? 
   Instead of flattening the story, we process the precontext, sentence, ending, and meaning 
   independently. This preserves the temporal components for the downstream sequence model.
"""

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType


class HierarchicalDebertaEncoder(nn.Module):
    """Encodes narrative components independently using a LoRA-adapted DeBERTa model.

     We designed this class to take the batched dictionary from DataLoader and 
    output four distinct, pooled vector representations per story.

    Attributes:
        base_model (DebertaV2Model): The frozen pre-trained Hugging Face model.
        lora_config (LoraConfig): The configuration dictating how LoRA is applied.
        model (PeftModel): The combined model with trainable LoRA adapters.
    """

    def __init__(
        self, 
        model_name: str = "microsoft/deberta-large", 
        lora_r: int = 8, 
        lora_alpha: int = 32, 
        lora_dropout: float = 0.1
    ) -> None:
        """Initializes the DeBERTa model and applies Low-Rank Adaptation (LoRA).

        Args:
            model_name (str): The Hugging Face model repository string.
            lora_r (int): The rank of the LoRA update matrices. We default this to 8 
                based on the NCL-UoR paper referenced in our milestone.
            lora_alpha (int): The scaling factor for LoRA.
            lora_dropout (float): Dropout probability for LoRA layers to prevent overfitting.
        """
        super().__init__()
        
        print(f"Loading base model: {model_name}...")
        
        # We use AutoModel.from_pretrained. This dynamically checks the config.json 
        # and realizes "microsoft/deberta-large" is a V1 model, loading the exact correct architecture.
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # DeBERTa V1 uses a single matrix called 'in_proj' to handle Query, Key, and Value.
        # We target 'in_proj' so LoRA injects the trainable matrices correctly into the attention head.
        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["in_proj"], 
            bias="none"
        )
        
        # We wrap the base model with PEFT. This executes the freezing and injecting process.
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # We print this to the console so my team can verify that our GPU memory will be safe.
        self.model.print_trainable_parameters()

    def _mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Performs mean pooling on token embeddings, ignoring padding tokens.

        When DeBERTa processes text, it outputs a vector for every single token. 
        BiLSTM needs exactly one vector per narrative component. We use this 
        mathematical function to average the token vectors safely.

        Args:
            token_embeddings (torch.Tensor): The raw output from DeBERTa. 
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): The mask indicating real tokens (1) vs padding (0). 
                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: The averaged vector for the sequence. 
                Shape: (batch_size, hidden_size).
        """
        # We expand the attention mask to match the 3D shape of the embeddings so we can multiply them.
        # Shape becomes: (batch_size, sequence_length, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # We multiply the embeddings by the mask. This instantly zeroes out all the padding tokens.
        # Then, we sum the embeddings along the sequence_length dimension (dim=1).
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # We calculate how many real (non-padding) tokens exist in each sequence to find the denominator.
        # We use torch.clamp with a tiny minimum value (1e-9) to guarantee we never divide by zero.
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # We return the mathematically averaged vector.
        return sum_embeddings / sum_mask

    def _encode_component(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Passes a single text component through the model and pools the output.

        Args:
            input_ids (torch.Tensor): Token IDs for a specific component.
            attention_mask (torch.Tensor): Attention mask for a specific component.

        Returns:
            torch.Tensor: A single pooled dense vector for the component.
        """
        # We pass the token IDs through the LoRA-adapted DeBERTa model.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # We extract the last hidden state and pass it to our pooling function.
        return self._mean_pooling(
            token_embeddings=outputs.last_hidden_state, 
            attention_mask=attention_mask
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Executes the forward pass for all narrative components independently.

        We designed this forward pass to perfectly accept the flat dictionary output 
        generated by `data_loader.py`. We extract the relevant keys for each 
        component, encode them independently, and return them.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing the batched 
                input IDs and attention masks for all 4 components.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the 4 pooled vectors, 
                ready to be passed to the Sequence Modeler.
        """
        # 1. Encode the Precontext (Sentences 1-3)
        precontext_vec = self._encode_component(
            input_ids=batch["precontext_input_ids"],
            attention_mask=batch["precontext_attention_mask"]
        )
        
        # 2. Encode the Ambiguous Sentence (Sentence 4)
        sentence_vec = self._encode_component(
            input_ids=batch["sentence_input_ids"],
            attention_mask=batch["sentence_attention_mask"]
        )
        
        # 3. Encode the Ending (Sentence 5 or Empty)
        ending_vec = self._encode_component(
            input_ids=batch["ending_input_ids"],
            attention_mask=batch["ending_attention_mask"]
        )
        
        # 4. Encode the Judged Meaning (Dictionary Definition)
        meaning_vec = self._encode_component(
            input_ids=batch["meaning_input_ids"],
            attention_mask=batch["meaning_attention_mask"]
        )
        
        # We package the resulting vectors into a dictionary for seamless downstream integration.
        return {
            "precontext_vec": precontext_vec,
            "sentence_vec": sentence_vec,
            "ending_vec": ending_vec,
            "meaning_vec": meaning_vec
        }


# =========================================================================================
# SANITY CHECK / DUMMY DATA BLOCK
# I wrote this block so I can execute `python feature_extraction.py` directly in my terminal 
# to test my code for compilation errors, GPU memory leaks, and shape mismatches.
# =========================================================================================
if __name__ == "__main__":
    
    print("Initializing the Hierarchical DeBERTa Encoder with LoRA...")
    encoder = HierarchicalDebertaEncoder()
    
    # We are simulating the exact batch size (8) and a standard sequence length (128)
    BATCH_SIZE = 8
    SEQ_LEN = 128
    
    print("\nGenerating dummy tensors matching `data_loader.py` output format...")
    # We create a dummy dictionary that perfectly mimics what Raid's collator yields.
    dummy_batch: Dict[str, torch.Tensor] = {
        "precontext_input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
        "precontext_attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN)),
        
        "sentence_input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
        "sentence_attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN)),
        
        "ending_input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
        "ending_attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN)),
        
        "meaning_input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
        "meaning_attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN)),
    }
    
    print("Running the forward pass through the LoRA-adapted model...")
    # I pass the dummy batch directly into the forward method.
    pooled_outputs = encoder(dummy_batch)
    
    print("\nFeature Extraction Successful! Validating output tensor shapes for Areeb:")
    for component_name, tensor in pooled_outputs.items():
        # DeBERTa-large hidden size is exactly 1024.
        expected_shape = (BATCH_SIZE, 1024)
        match_status = "OK" if tensor.shape == expected_shape else "MISMATCH"
        print(f" -> {component_name}: {tensor.shape} [{match_status}]")