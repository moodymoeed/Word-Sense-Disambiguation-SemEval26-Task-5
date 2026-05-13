"""Sequence Modeling Module for the CS445 Group 2 SemEval system.

Project Phase: Step 3 - Temporal Reasoning with Bidirectional LSTM
-------------------------------------------------------------------
This file serves as the Sequence Modeler of our architecture.
Upstream, `feature_extraction.py` encodes narrative components into dense vectors.
This file takes those 4 vectors and models the chronological flow of the story
using a Bidirectional LSTM to capture how the ending retroactively changes 
interpretation of the past (retroactive contextualization).
Downstream, the final prediction goes to Ramzy's multi-objective loss function.

Architectural Reasoning:
------------------------
1. Why Bidirectional LSTM?
   Transformers are great for parallel processing but cannot model ordered temporal sequences.
   The AmbiStory dataset relies on retroactive contextualization: the ending sentence 
   alters how we interpret the target sentence. BiLSTM captures this by having:
   - Forward pass: How setup affects climax
   - Backward pass: How resolution reinterprets the past
   
2. Why Stack [precontext, sentence, ending]?
   A story is not a random bag of sentences. It has structure: setup → climax → resolution.
   By stacking these 3 components in order, we preserve this narrative logic.
   
3. Why Concatenate with meaning_vec?
   The BiLSTM learns the narrative context. But we need to evaluate a SPECIFIC definition.
   Concatenating the final narrative state with the meaning vector combines:
   [narrative_context] + [target_definition] → plausibility_score
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """Bidirectional LSTM sequence model for temporal narrative reasoning.

    This class takes 4 encoded narrative components from Moeed's DeBERTa encoder
    and outputs continuous plausibility scores (1-5) by modeling the chronological
    flow of the story and how each definition fits into the narrative context.

    Architecture Flow:
    1. Stack temporal components (precontext, sentence, ending) into a sequence
    2. Feed through Bidirectional LSTM to capture narrative flow
    3. Extract final hidden state (contains full narrative interpretation)
    4. Concatenate with the meaning vector (the specific definition)
    5. Pass through fully connected regression head to output plausibility score

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM layer for sequence processing
        regression_head (nn.Sequential): MLP for regression to [1, 5] range
        output_clamp (tuple): Valid output range for plausibility scores
    """

    def __init__(
        self,
        encoder_hidden_size: int = 1024,
        lstm_hidden_size: int = 512,
        num_lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        regression_hidden_size: int = 256,
        output_clamp: Optional[Tuple[float, float]] = (1.0, 5.0),
    ) -> None:
        """Initializes the Sequence Model with BiLSTM and regression head.

        Args:
            encoder_hidden_size (int): Hidden dimension from DeBERTa-large encoder.
                Moeed's encoder outputs vectors of this size. Default: 1024
            lstm_hidden_size (int): Hidden dimension of the BiLSTM layer.
                Balances memory and expressiveness. Default: 512
            num_lstm_layers (int): Number of stacked LSTM layers.
                Allows learning hierarchical temporal patterns. Default: 2
            lstm_dropout (float): Dropout probability in LSTM and regression head
                to prevent overfitting. Default: 0.1
            regression_hidden_size (int): Hidden dimension of the regression MLP.
                First dense layer: (lstm_hidden_size * 2 + encoder_hidden_size) → this value
                Second dense layer: this value → 1. Default: 256
            output_clamp (tuple): Min and max values for clamping output scores.
                Plausibility is rated 1-5, so we clamp to [1.0, 5.0].
                Set to None to disable clamping. Default: (1.0, 5.0)
        """
        super().__init__()

        print(f"Initializing Sequence Model with BiLSTM (hidden_size={lstm_hidden_size})...")

        # ========== PHASE 2: Initialize Bidirectional LSTM ==========
        # The BiLSTM will receive stacked narrative components: (batch_size, 3, 1024)
        # It processes these 3 sequential narrative steps bidirectionally to capture:
        #   - Forward: how precontext and sentence build toward climax
        #   - Backward: how ending changes interpretation of previous context
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_size,  # 1024 from DeBERTa
            hidden_size=lstm_hidden_size,  # 512 (configurable)
            num_layers=num_lstm_layers,  # 2 (stacked layers)
            bidirectional=True,  # CRITICAL: enables forward + backward passes
            dropout=lstm_dropout,  # 0.1 for regularization
            batch_first=True,  # Input shape: (batch, seq_len, features)
        )

        # ========== PHASE 3: Initialize Regression Head ==========
        # After BiLSTM processes the temporal sequence, we have:
        #   - BiLSTM hidden state: (batch_size, lstm_hidden_size * 2) = (batch_size, 1024)
        #       [Doubled because bidirectional: forward(512) + backward(512)]
        #   - Meaning vector from Moeed: (batch_size, encoder_hidden_size) = (batch_size, 1024)
        #
        # Concatenating these: (batch_size, 1024 + 1024) = (batch_size, 2048)
        # This combined vector is fed through the regression head to predict plausibility.

        regression_input_size = (lstm_hidden_size * 2) + encoder_hidden_size

        self.regression_head = nn.Sequential(
            # Dense layer 1: Project from combined features to hidden dimension
            nn.Linear(regression_input_size, regression_hidden_size),  # 2048 → 256
            nn.ReLU(),  # Non-linearity allows learning complex score patterns
            nn.Dropout(lstm_dropout),  # Regularization
            # Dense layer 2: Project from hidden to single output score
            nn.Linear(regression_hidden_size, 1),  # 256 → 1
            # No activation here; we'll clamp the output instead
        )

        self.output_clamp = output_clamp
        print(f"Sequence Model initialized. Ready to process narrative sequences.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute the forward pass through the sequence model.

        Processes narrative components through BiLSTM temporal reasoning and
        produces continuous plausibility scores for each sample in the batch.

        Args:
            batch (Dict[str, torch.Tensor]): Dictionary from Moeed's encoder containing:
                - 'precontext_vec': (batch_size, 1024) - Sentences 1-3 context
                - 'sentence_vec': (batch_size, 1024) - The ambiguous sentence
                - 'ending_vec': (batch_size, 1024) - Sentence 5 or empty
                - 'meaning_vec': (batch_size, 1024) - Dictionary definition to evaluate

        Returns:
            torch.Tensor: Plausibility scores, shape (batch_size, 1).
                Values are typically in range [1.0, 5.0] after clamping.
                Each value represents how well the meaning fits in the narrative context.

        Dimension Tracking Through the Forward Pass:
        ============================================
        Input from Moeed:
          precontext_vec: (8, 1024)
          sentence_vec:   (8, 1024)
          ending_vec:     (8, 1024)
          meaning_vec:    (8, 1024)

        After stacking temporal components:
          temporal_sequence: (8, 3, 1024)  [batch_size, seq_len=3, hidden=1024]

        After BiLSTM (bidirectional, so hidden is doubled):
          lstm_output: (8, 3, 1024)        [batch_size, seq_len, hidden*2]
          h_n:         (4, 8, 512)         [num_layers*2, batch_size, hidden]

        After extracting final hidden state:
          final_hidden: (8, 1024)          [batch_size, hidden*2]

        After concatenating with meaning_vec:
          combined: (8, 2048)              [batch_size, hidden*2 + encoder_hidden]

        After regression head:
          logits: (8, 1)                   [batch_size, 1] ← FINAL OUTPUT
        """

        # ========== PHASE 4: Extract Inputs from Batch ==========
        # These 4 vectors come directly from Moeed's HierarchicalDebertaEncoder
        precontext_vec = batch["precontext_vec"]  # (batch_size, 1024)
        sentence_vec = batch["sentence_vec"]  # (batch_size, 1024)
        ending_vec = batch["ending_vec"]  # (batch_size, 1024)
        meaning_vec = batch["meaning_vec"]  # (batch_size, 1024)

        # ========== PHASE 5A: Stack Temporal Components ==========
        # This is the critical architectural decision: preserve narrative order.
        # We're saying: "These 3 components form a sequence in THIS order"
        #   1. Precontext (setup) - Sentences 1-3
        #   2. Sentence (climax) - The target ambiguous sentence
        #   3. Ending (resolution) - Sentence 5
        # The BiLSTM will learn how these relate chronologically.
        temporal_sequence = torch.stack(
            [precontext_vec, sentence_vec, ending_vec], dim=1
        )
        # Shape after stack: (batch_size, 3, 1024)
        # The 3 is a new dimension at position 1 (sequence length)

        # ========== PHASE 5B: Pass Through Bidirectional LSTM ==========
        # The LSTM processes this sequence in TWO directions simultaneously:
        #
        # FORWARD PASS: precontext → sentence → ending
        #   Captures how the setup influences the climax
        #
        # BACKWARD PASS: ending → sentence → precontext
        #   Captures how the resolution retroactively reinterprets what came before
        #   This models the "retroactive contextualization" the proposal mentions
        #
        lstm_output, (h_n, c_n) = self.lstm(temporal_sequence)
        # lstm_output: (batch_size, seq_len=3, lstm_hidden_size*2=1024)
        #   Hidden state at each of the 3 sequence steps (we won't use this)
        # h_n: (num_layers*bidirectional=4, batch_size, lstm_hidden_size=512)
        #   Final hidden state for each layer and direction
        #   Shape breakdown: [layer_0_fwd, layer_0_bwd, layer_1_fwd, layer_1_bwd]
        # c_n: (num_layers*bidirectional=4, batch_size, lstm_hidden_size=512)
        #   Final cell state (not used, but returned by LSTM)

        # ========== PHASE 5C: Extract Final Hidden State ==========
        # We want the final hidden state from the TOP layer of the bidirectional LSTM.
        # h_n[-2] is the forward direction of the top layer
        # h_n[-1] is the backward direction of the top layer
        # Concatenating them gives us the complete narrative interpretation.
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        # Shape: (batch_size, lstm_hidden_size*2) = (batch_size, 512*2) = (batch_size, 1024)

        # ========== PHASE 5D: Concatenate with Meaning Vector ==========
        # Now we combine:
        #   - final_hidden: The narrative context after bidirectional processing
        #   - meaning_vec: The specific definition we're evaluating
        # This combined representation contains both the story interpretation
        # and the candidate meaning, allowing the regression head to learn
        # how well they fit together.
        combined = torch.cat([final_hidden, meaning_vec], dim=1)
        # Shape: (batch_size, 1024 + 1024) = (batch_size, 2048)

        # ========== PHASE 5E: Pass Through Regression Head ==========
        # The regression head is a small MLP that learns the non-linear mapping from
        # [narrative_context + meaning_vector] → plausibility_score
        logits = self.regression_head(combined)
        # Shape: (batch_size, 1)

        # ========== PHASE 6: Clamp Output to Valid Range ==========
        # Plausibility scores should be between 1 (highly implausible) and 5 (highly plausible).
        # We clamp the output to enforce this constraint.
        if self.output_clamp is not None:
            logits = torch.clamp(
                logits, min=self.output_clamp[0], max=self.output_clamp[1]
            )

        return logits  # (batch_size, 1)


# =========================================================================================
# SANITY CHECK / DUMMY DATA BLOCK
# I wrote this block so I can execute `python sequence_model.py` directly in my terminal
# to test my code for compilation errors, shape mismatches, and numerical stability.
# =========================================================================================
if __name__ == "__main__":

    print("Initializing the Sequence Model (BiLSTM + Regression Head)...")
    sequence_model = SequenceModel()

    # We are simulating the exact batch size (8) and hidden dimension (1024)
    # that Moeed's encoder produces
    BATCH_SIZE = 8

    print(
        "\nGenerating dummy tensors matching `feature_extraction.py` output format..."
    )
    # We create a dummy dictionary that perfectly mimics what Moeed's encoder returns.
    dummy_batch: Dict[str, torch.Tensor] = {
        "precontext_vec": torch.randn(BATCH_SIZE, 1024),
        "sentence_vec": torch.randn(BATCH_SIZE, 1024),
        "ending_vec": torch.randn(BATCH_SIZE, 1024),
        "meaning_vec": torch.randn(BATCH_SIZE, 1024),
    }

    print("Running the forward pass through the BiLSTM sequence model...")
    # I pass the dummy batch directly into the forward method.
    scores = sequence_model(dummy_batch)

    print("\nSequence Modeling Successful! Validating output tensor shape for Ramzy:")
    expected_shape = (BATCH_SIZE, 1)
    match_status = "OK" if scores.shape == expected_shape else "MISMATCH"
    print(
        f" -> Plausibility scores: {scores.shape} [{match_status}]"
    )

    print(f"\nSample output values (should be in range [1.0, 5.0]):")
    print(f" -> Min: {scores.min().item():.4f}")
    print(f" -> Max: {scores.max().item():.4f}")
    print(f" -> Mean: {scores.mean().item():.4f}")

    print("\nAll checks passed!")
