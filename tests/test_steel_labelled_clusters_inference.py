#!/usr/bin/env python
# coding: utf-8

import unittest
from pathlib import Path

import pandas as pd

from regression.steel_labelled_clusters_inference import (
    checkpoint_path_for_target,
    validate_inference_dataset,
)
from regression.steel_labelled_clusters_regression import ELEMENT_COLS


class TestSteelLabelledClustersInference(unittest.TestCase):
    def test_validate_inference_dataset_accepts_text_and_elements(self):
        row = {"Text": "Oil quenched and tempered"}
        row.update({element: 0.0 for element in ELEMENT_COLS})
        row["Fe"] = 99.0
        row["C"] = 1.0

        validate_inference_dataset(pd.DataFrame([row]))

    def test_validate_inference_dataset_rejects_missing_element(self):
        row = {"Text": "Oil quenched and tempered"}
        row.update({element: 0.0 for element in ELEMENT_COLS if element != "Fe"})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_inference_dataset(pd.DataFrame([row]))

    def test_checkpoint_path_for_target(self):
        path = checkpoint_path_for_target(Path("/tmp/ckpt"), "Tensile_value")

        self.assertEqual(path, Path("/tmp/ckpt/Tensile_value_best_model.pt"))


if __name__ == "__main__":
    unittest.main()
