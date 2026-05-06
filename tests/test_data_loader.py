import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTensor(list):
    def long(self):
        return self


class FakeTorch(types.ModuleType):
    float32 = "float32"

    def tensor(self, values, dtype=None):
        tensor = FakeTensor(values)
        tensor.dtype = dtype
        return tensor


class FakeDataset:
    pass


class FakeDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        num_workers,
        collate_fn,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.kwargs = kwargs

    def __iter__(self):
        examples = [self.dataset[i] for i in range(self.batch_size)]
        yield self.collate_fn(examples)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        width = max(2, max(len(text.split()) + 2 for text in texts))
        return {
            "input_ids": FakeTensor([[1] * width for _ in texts]),
            "attention_mask": FakeTensor([[1] * width for _ in texts]),
        }


def install_fake_torch():
    fake_torch = FakeTorch("torch")
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")

    fake_data.Dataset = FakeDataset
    fake_data.DataLoader = FakeDataLoader
    fake_utils.data = fake_data

    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.data"] = fake_data


class DataLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_fake_torch()
        sys.path.insert(0, str(REPO_ROOT))
        cls.dl = importlib.import_module("src.data_loader")

    def test_dataset_parses_official_splits_without_resplitting(self):
        train = self.dl.AmbiStoryDataset(REPO_ROOT / "data/train.json")
        dev = self.dl.AmbiStoryDataset(REPO_ROOT / "data/dev.json")
        test = self.dl.AmbiStoryDataset(REPO_ROOT / "data/test.json", require_targets=False)

        self.assertEqual(len(train), 2280)
        self.assertEqual(len(dev), 588)
        self.assertEqual(len(test), 930)
        self.assertEqual(train[0]["id"], "0")
        self.assertEqual(dev[0]["id"], "0")
        self.assertEqual(test[0]["id"], "0")
        self.assertEqual(train[0]["target_score"], 3.0)
        self.assertAlmostEqual(train[0]["target_stdev"], 1.5811388300841898)

    def test_empty_endings_are_preserved(self):
        train = self.dl.AmbiStoryDataset(REPO_ROOT / "data/train.json")

        self.assertEqual(sum(1 for record in train.records if record["ending"] == ""), 760)
        self.assertEqual(train[2278]["ending"], "")

    def test_labeled_test_split_requires_numeric_targets(self):
        with self.assertRaisesRegex(ValueError, "non-numeric 'average'"):
            self.dl.AmbiStoryDataset(REPO_ROOT / "data/test.json")

    def test_collator_returns_exact_training_contract(self):
        tokenizer = FakeTokenizer()
        loader = self.dl.build_dataloader(
            REPO_ROOT / "data/train.json",
            tokenizer=tokenizer,
            batch_size=8,
            shuffle=False,
            max_length=256,
        )

        batch = next(iter(loader))

        self.assertEqual(tuple(batch.keys()), self.dl.EXPECTED_BATCH_KEYS)
        self.assertEqual(len(batch["target_score"]), 8)
        self.assertEqual(len(batch["target_stdev"]), 8)
        self.assertEqual(batch["target_score"].dtype, "float32")
        self.assertEqual(batch["target_stdev"].dtype, "float32")
        self.assertEqual(len(tokenizer.calls), 4)

        for _, kwargs in tokenizer.calls:
            self.assertEqual(kwargs["padding"], "longest")
            self.assertTrue(kwargs["truncation"])
            self.assertEqual(kwargs["return_tensors"], "pt")
            self.assertEqual(kwargs["max_length"], 256)

    def test_components_are_tokenized_independently_and_in_temporal_order(self):
        tokenizer = FakeTokenizer()
        train = self.dl.AmbiStoryDataset(REPO_ROOT / "data/train.json")
        collator = self.dl.AmbiStoryCollator(tokenizer=tokenizer, max_length=128)

        collator([train[0], train[1]])

        called_texts = [texts for texts, _ in tokenizer.calls]
        self.assertEqual(called_texts[0], [train[0]["precontext"], train[1]["precontext"]])
        self.assertEqual(called_texts[1], [train[0]["sentence"], train[1]["sentence"]])
        self.assertEqual(called_texts[2], [train[0]["ending"], train[1]["ending"]])
        self.assertEqual(called_texts[3], [train[0]["meaning"], train[1]["meaning"]])

    def test_per_component_max_lengths_are_supported(self):
        tokenizer = FakeTokenizer()
        train = self.dl.AmbiStoryDataset(REPO_ROOT / "data/train.json")
        collator = self.dl.AmbiStoryCollator(
            tokenizer=tokenizer,
            max_length={
                "precontext": 256,
                "sentence": 64,
                "ending": 128,
                "meaning": 128,
            },
        )

        collator([train[0], train[1]])

        self.assertEqual([kwargs["max_length"] for _, kwargs in tokenizer.calls], [256, 64, 128, 128])

    def test_split_loader_disables_test_targets_for_inference(self):
        tokenizer = FakeTokenizer()
        loaders = self.dl.build_split_dataloaders(
            data_dir=REPO_ROOT / "data",
            tokenizer=tokenizer,
            batch_size=2,
            splits=("train", "dev", "test"),
            shuffle_train=True,
        )

        self.assertTrue(loaders["train"].shuffle)
        self.assertFalse(loaders["dev"].shuffle)
        self.assertFalse(loaders["test"].shuffle)

        test_batch = next(iter(loaders["test"]))
        self.assertNotIn("target_score", test_batch)
        self.assertNotIn("target_stdev", test_batch)

    def test_optional_ids_are_available_for_prediction_writing(self):
        tokenizer = FakeTokenizer()
        loader = self.dl.build_dataloader(
            REPO_ROOT / "data/dev.json",
            tokenizer=tokenizer,
            batch_size=2,
            include_ids=True,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["id"], ["0", "1"])


class TokenizerFactoryTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("transformers", None)

    def test_tokenizer_factory_prefers_requested_deberta_v2_class(self):
        calls = []

        class FakeDebertaV2Tokenizer:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("deberta_v2", model_name, kwargs))
                return "tokenizer"

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.DebertaV2Tokenizer = FakeDebertaV2Tokenizer
        fake_transformers.AutoTokenizer = object
        sys.modules["transformers"] = fake_transformers

        from src import data_loader

        tokenizer = data_loader.build_tokenizer()

        self.assertEqual(tokenizer, "tokenizer")
        self.assertEqual(calls[0][0], "deberta_v2")
        self.assertEqual(calls[0][1], "microsoft/deberta-large")

    def test_tokenizer_factory_can_fall_back_to_checkpoint_compatible_auto_tokenizer(self):
        calls = []

        class FakeDebertaV2Tokenizer:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("deberta_v2", model_name, kwargs))
                raise OSError("incompatible tokenizer files")

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("auto", model_name, kwargs))
                return "auto-tokenizer"

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.DebertaV2Tokenizer = FakeDebertaV2Tokenizer
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        sys.modules["transformers"] = fake_transformers

        from src import data_loader

        with self.assertWarns(RuntimeWarning):
            tokenizer = data_loader.build_tokenizer()

        self.assertEqual(tokenizer, "auto-tokenizer")
        self.assertEqual([call[0] for call in calls], ["deberta_v2", "auto"])


if __name__ == "__main__":
    unittest.main()
