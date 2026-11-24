import importlib
from io import StringIO
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import pandas as pd

from dataprep.class_splits import assert_disjoint


build_splits = importlib.import_module("scripts.05_build_class_splits")
validate_dataprep = importlib.import_module("scripts.07_validate_dataprep")


class ClassIdentityTests(unittest.TestCase):
    def test_distinct_synsets_with_same_normalized_name_are_disjoint(self):
        assert_disjoint(
            [{"wnid": "n02113186", "class_name": "Cardigan"}],
            [{"wnid": "n02963159", "class_name": " cardigan "}],
        )

    def test_same_synset_with_different_names_is_leakage(self):
        with self.assertRaisesRegex(ValueError, "wnid 'n02113186'"):
            assert_disjoint(
                [{"wnid": "n02113186", "class_name": "Cardigan"}],
                [{"wnid": "n02113186", "class_name": "Welsh corgi"}],
            )

    def test_name_fallback_applies_when_either_id_is_missing(self):
        for seen_id, unseen_id in (("n1", None), (None, "n2"), (None, None),
                                   ("n1", ""), ("", "n2")):
            with self.subTest(seen_id=seen_id, unseen_id=unseen_id):
                seen = {"class_name": " Red_Fox "}
                unseen = {"class_name": "red fox"}
                if seen_id is not None:
                    seen["wnid"] = seen_id
                if unseen_id is not None:
                    unseen["wnid"] = unseen_id
                with self.assertRaisesRegex(ValueError, "normalized label 'red fox'"):
                    assert_disjoint([seen], [unseen])

    def test_mixed_entries_are_checked_individually(self):
        seen = [
            {"wnid": "n1", "class_name": "Cardigan"},
            {"class_name": "Red_Fox"},
        ]
        assert_disjoint(seen, [{"wnid": "n2", "class_name": "cardigan"}])
        for unseen in (
            {"class_name": "cardigan"},
            {"wnid": "n3", "class_name": "red fox"},
        ):
            with self.subTest(unseen=unseen):
                with self.assertRaisesRegex(ValueError, "leakage"):
                    assert_disjoint(seen, [unseen])

    def test_disjoint_and_empty_groups_pass(self):
        assert_disjoint(
            [{"wnid": "n1", "class_name": "dog"}],
            [],
            [{"class_name": "bird"}],
            [{"wnid": "n2", "class_name": "cat"}],
        )
        assert_disjoint([], [{"class_name": "bird"}])


class SplitScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifests = self.root / "manifests"
        self.splits = self.root / "splits"
        self.manifests.mkdir()
        self.splits.mkdir()

    def write_splits(self, seen, unseen):
        (self.splits / "seen.json").write_text(json.dumps({"classes": seen}))
        (self.splits / "unseen_imagenet.json").write_text(json.dumps(unseen))

    def test_generation_and_validation_accept_homonyms_and_preserve_selection(self):
        classes = pd.DataFrame({
            "wnid": [f"n{i:08d}" for i in range(1000)],
            "class_name": [f"class {i}" for i in range(1000)],
        })
        shuffled = classes["wnid"].tolist()
        random.Random(541).shuffle(shuffled)
        seen_wnid, unseen_wnid = shuffled[0], shuffled[800]
        classes.loc[classes["wnid"] == seen_wnid, "class_name"] = "Cardigan"
        classes.loc[classes["wnid"] == unseen_wnid, "class_name"] = "cardigan"
        classes.to_parquet(self.manifests / "imagenet_classes.parquet", index=False)
        with (
            patch.object(build_splits, "MANIFESTS_DIR", self.manifests),
            patch.object(build_splits, "SPLITS_DIR", self.splits),
            patch.object(validate_dataprep, "SPLITS_DIR", self.splits),
            redirect_stdout(StringIO()),
        ):
            build_splits.main(k=800, seed=541)
            first = (self.splits / "seen.json").read_text()
            build_splits.main(k=800, seed=541)
            self.assertEqual((self.splits / "seen.json").read_text(), first)
            validate_dataprep.check_splits_disjoint()

        seen = json.loads(first)
        unseen = json.loads((self.splits / "unseen_imagenet.json").read_text())
        self.assertEqual((seen["k"], seen["seed"]), (800, 541))
        self.assertEqual([entry["wnid"] for entry in seen["classes"]], sorted(shuffled[:800]))
        self.assertEqual([entry["wnid"] for entry in unseen], sorted(shuffled[800:]))
        for domain in ("awa2", "cub", "sun"):
            self.assertEqual(
                json.loads((self.splits / f"unseen_{domain}.json").read_text()), []
            )

    def test_generation_rejects_cross_domain_name_overlap_before_writing(self):
        pd.DataFrame({
            "wnid": ["n1"], "class_name": ["Red_Fox"],
        }).to_parquet(self.manifests / "imagenet_classes.parquet", index=False)
        pd.DataFrame({
            "class_name": [" red fox "], "is_unseen_ps": [True],
        }).to_parquet(self.manifests / "awa2.parquet", index=False)
        with (
            patch.object(build_splits, "MANIFESTS_DIR", self.manifests),
            patch.object(build_splits, "SPLITS_DIR", self.splits),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "normalized label 'red fox'"):
                build_splits.main(k=1, seed=541)
        self.assertFalse((self.splits / "seen.json").exists())

    def test_validator_rejects_true_id_overlap_even_with_different_names(self):
        self.write_splits(
            [{"wnid": "n1", "class_name": "Cardigan"}],
            [{"wnid": "n1", "class_name": "Welsh corgi"}],
        )
        with patch.object(validate_dataprep, "SPLITS_DIR", self.splits):
            with self.assertRaisesRegex(ValueError, "unseen_imagenet.json:.*wnid 'n1'"):
                validate_dataprep.check_splits_disjoint()

    def test_validator_preserves_normalized_name_checks_for_domain_splits(self):
        self.write_splits([{"wnid": "n1", "class_name": "Red_Fox"}], [])
        (self.splits / "unseen_awa2.json").write_text(
            json.dumps([{"class_name": " red fox "}])
        )
        with patch.object(validate_dataprep, "SPLITS_DIR", self.splits):
            with self.assertRaisesRegex(ValueError, "unseen_awa2.json:.*red fox"):
                validate_dataprep.check_splits_disjoint()


if __name__ == "__main__":
    unittest.main()
