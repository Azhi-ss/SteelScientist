#!/usr/bin/env python
# coding: utf-8

import unittest

import pandas as pd
import torch

from regression.steel_labelled_clusters_regression import (
    ELEMENT_COLS,
    TARGET_COLS,
    TextElementGatedFusionRegressor,
    build_feature_bank,
    validate_dataset,
)


class TestSteelLabelledClustersRegression(unittest.TestCase):
    def _frame(self):
        rows = []
        for i in range(3):
            row = {
                "Text": f"heat treatment {i}",
                "Tensile_value": 700.0 + i,
                "Yield_value": 500.0 + i,
                "Elongation_value": 20.0 + i,
            }
            for element in ELEMENT_COLS:
                row[element] = 0.0
            row["Fe"] = 99.0
            row["C"] = 1.0
            rows.append(row)
        return pd.DataFrame(rows)

    def test_validate_dataset_accepts_required_columns(self):
        df = self._frame()

        validate_dataset(df, "unit-test")

    def test_validate_dataset_rejects_missing_columns(self):
        df = self._frame().drop(columns=["Text"])

        with self.assertRaisesRegex(ValueError, "unit-test missing required columns"):
            validate_dataset(df, "unit-test")

    def test_build_feature_bank_uses_all_36_elements(self):
        df = self._frame()
        text_embeds = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ele_embeds = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        features = build_feature_bank(df, text_embeds, ele_embeds)

        self.assertEqual(features["text_embeds"].shape, (3, 2))
        self.assertEqual(features["ele_embeds"].shape, (3, 2))
        self.assertEqual(features["raw_composition"].shape, (3, len(ELEMENT_COLS)))
        self.assertEqual(list(TARGET_COLS), ["Tensile_value", "Yield_value", "Elongation_value"])

    def test_text_element_gate_forward_pass_excludes_raw_composition(self):
        model = TextElementGatedFusionRegressor(
            text_input_dim=2,
            ele_input_dim=2,
            latent_dim=4,
            text_hidden_dims=[4],
            ele_hidden_dims=[4],
            gate_hidden_dim=4,
            head_hidden_dims=[4],
            dropout=0.0,
        )

        pred, aux = model(
            torch.randn(3, 2),
            torch.randn(3, 2),
            return_aux=True,
        )

        self.assertEqual(pred.shape, (3, 1))
        self.assertEqual(aux["gate_weights"].shape, (3, 2))
        self.assertTrue(torch.allclose(aux["gate_weights"].sum(dim=1), torch.ones(3)))
        self.assertNotIn("raw_latent", aux)


if __name__ == "__main__":
    unittest.main()
