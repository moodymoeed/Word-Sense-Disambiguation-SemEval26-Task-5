"""PyTorch data pipeline for AmbiStory / SemEval 2026 Task 5.

The project architecture encodes the narrative components separately:
precontext, ambiguous sentence, ending, and candidate meaning.  The collator
therefore tokenizes each component independently and returns the exact batch
keys expected by the hierarchical DeBERTa + BiLSTM model.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # Lets lightweight tooling import the module before deps are installed.
    torch = None
    DataLoader = None
    Dataset = object


DEFAULT_MODEL_NAME = "microsoft/deberta-large"
DEFAULT_MAX_LENGTH = 256

COMPONENT_FIELDS: Mapping[str, str] = {
    "precontext": "precontext",
    "sentence": "sentence",
    "ending": "ending",
    "meaning": "judged_meaning",
}

TEXT_FIELDS = tuple(COMPONENT_FIELDS.values())
TARGET_FIELDS = ("average", "stdev")

EXPECTED_BATCH_KEYS = (
    "precontext_input_ids",
    "precontext_attention_mask",
    "sentence_input_ids",
    "sentence_attention_mask",
    "ending_input_ids",
    "ending_attention_mask",
    "meaning_input_ids",
    "meaning_attention_mask",
    "target_score",
    "target_stdev",
)


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is required for data loading. Install project dependencies "
            "with `pip install -r requirements.txt`."
        )


def build_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    prefer_deberta_v2_tokenizer: bool = True,
    strict_tokenizer_class: bool = False,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Create the tokenizer used by the encoder.

    The team handoff requested ``DebertaV2Tokenizer.from_pretrained``.  That is
    attempted first by default.  If the model checkpoint is incompatible with
    that tokenizer class, the function falls back to ``AutoTokenizer`` so the
    tokenizer still matches the checkpoint vocabulary/configuration.
    """

    try:
        from transformers import AutoTokenizer, DebertaV2Tokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformers is required for tokenization. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    if prefer_deberta_v2_tokenizer:
        try:
            return DebertaV2Tokenizer.from_pretrained(model_name, **from_pretrained_kwargs)
        except Exception:
            if strict_tokenizer_class:
                raise
            warnings.warn(
                f"Could not load {model_name!r} with DebertaV2Tokenizer; "
                "falling back to AutoTokenizer for checkpoint-compatible IDs.",
                RuntimeWarning,
                stacklevel=2,
            )

    return AutoTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)


class AmbiStoryDataset(Dataset):
    """Dataset wrapper that reads one official split JSON file.

    No split is created inside this class.  Pass ``data/train.json``,
    ``data/dev.json``, or ``data/test.json`` explicitly so train/dev/test
    isolation stays strict.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        require_targets: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.require_targets = require_targets
        self.records = self._load_records(self.data_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]

    def _load_records(self, data_path: Path) -> List[Dict[str, Any]]:
        with data_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_items = list(raw_data.items())
            if all(str(key).isdigit() for key, _ in raw_items):
                raw_items = sorted(raw_items, key=lambda item: int(item[0]))
        elif isinstance(raw_data, list):
            raw_items = [(str(i), sample) for i, sample in enumerate(raw_data)]
        else:
            raise ValueError(f"{data_path} must contain a JSON object or list.")

        records: List[Dict[str, Any]] = []
        for sample_id, sample in raw_items:
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {sample_id} in {data_path} is not a JSON object.")

            required_fields: Tuple[str, ...] = TEXT_FIELDS
            if self.require_targets:
                required_fields = required_fields + TARGET_FIELDS

            missing = [field for field in required_fields if field not in sample]
            if missing:
                raise ValueError(
                    f"Sample {sample_id} in {data_path} is missing required fields: {missing}"
                )

            records.append(
                {
                    "id": str(sample_id),
                    "precontext": str(sample["precontext"]),
                    "sentence": str(sample["sentence"]),
                    "ending": "" if sample["ending"] is None else str(sample["ending"]),
                    "meaning": str(sample["judged_meaning"]),
                    "target_score": self._parse_target(sample_id, sample, "average"),
                    "target_stdev": self._parse_target(sample_id, sample, "stdev"),
                }
            )

        return records

    def _parse_target(self, sample_id: str, sample: Mapping[str, Any], field: str) -> Optional[float]:
        if not self.require_targets:
            return None
        try:
            return float(sample[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sample {sample_id} in {self.data_path} has a non-numeric {field!r}: "
                f"{sample[field]!r}. Use require_targets=False for unlabeled inference splits."
            ) from exc


@dataclass
class AmbiStoryCollator:
    """Tokenize a batch into the model-facing tensor contract."""

    tokenizer: Any
    max_length: Optional[Union[int, Mapping[str, Optional[int]]]] = DEFAULT_MAX_LENGTH
    include_ids: bool = False
    require_targets: bool = True

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        _require_torch()

        batch: Dict[str, Any] = {}
        for component_name in COMPONENT_FIELDS:
            texts = [str(example[component_name]) for example in examples]
            encoded = self._tokenize_component(component_name, texts)
            batch[f"{component_name}_input_ids"] = encoded["input_ids"].long()
            batch[f"{component_name}_attention_mask"] = encoded["attention_mask"].long()

        if self.require_targets:
            batch["target_score"] = torch.tensor(
                [float(example["target_score"]) for example in examples],
                dtype=torch.float32,
            )
            batch["target_stdev"] = torch.tensor(
                [float(example["target_stdev"]) for example in examples],
                dtype=torch.float32,
            )

        if self.include_ids:
            batch["id"] = [str(example["id"]) for example in examples]

        return batch

    def _tokenize_component(self, component_name: str, texts: Sequence[str]) -> Mapping[str, Any]:
        tokenizer_kwargs: Dict[str, Any] = {
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
        }

        max_length = self._max_length_for(component_name)
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        return self.tokenizer(list(texts), **tokenizer_kwargs)

    def _max_length_for(self, component_name: str) -> Optional[int]:
        if isinstance(self.max_length, Mapping):
            return self.max_length.get(component_name)
        return self.max_length


def build_dataloader(
    data_path: Union[str, Path],
    tokenizer: Optional[Any] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    max_length: Optional[Union[int, Mapping[str, Optional[int]]]] = DEFAULT_MAX_LENGTH,
    num_workers: int = 0,
    include_ids: bool = False,
    require_targets: bool = True,
    **dataloader_kwargs: Any,
) -> Any:
    """Build one DataLoader for an explicit split file."""

    _require_torch()

    if tokenizer is None:
        tokenizer = build_tokenizer()

    dataset = AmbiStoryDataset(data_path, require_targets=require_targets)
    collator = AmbiStoryCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        include_ids=include_ids,
        require_targets=require_targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        **dataloader_kwargs,
    )


def build_split_dataloaders(
    data_dir: Union[str, Path] = "data",
    tokenizer: Optional[Any] = None,
    batch_size: int = 8,
    splits: Iterable[str] = ("train", "dev", "test"),
    max_length: Optional[Union[int, Mapping[str, Optional[int]]]] = DEFAULT_MAX_LENGTH,
    shuffle_train: bool = True,
    num_workers: int = 0,
    include_ids: bool = False,
    require_targets: Union[bool, Mapping[str, bool]] = True,
    **dataloader_kwargs: Any,
) -> Dict[str, Any]:
    """Build loaders for official split files without re-splitting the data.

    The released ``test.json`` uses placeholder labels, so test targets are
    disabled automatically when ``require_targets`` is left as ``True``.
    Pass a mapping such as ``{"train": True, "dev": True, "test": False}``
    to control this explicitly.
    """

    _require_torch()

    if tokenizer is None:
        tokenizer = build_tokenizer()

    data_root = Path(data_dir)
    loaders: Dict[str, Any] = {}
    for split in splits:
        split_requires_targets = _requires_targets_for_split(split, require_targets)
        loaders[split] = build_dataloader(
            data_path=data_root / f"{split}.json",
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle=split == "train" and shuffle_train,
            max_length=max_length,
            num_workers=num_workers,
            include_ids=include_ids,
            require_targets=split_requires_targets,
            **dataloader_kwargs,
        )

    return loaders


def _requires_targets_for_split(
    split: str,
    require_targets: Union[bool, Mapping[str, bool]],
) -> bool:
    if isinstance(require_targets, Mapping):
        return bool(require_targets.get(split, True))
    if split == "test":
        return False
    return bool(require_targets)
