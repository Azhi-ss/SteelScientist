#!/usr/bin/env python
# coding: utf-8

import unittest

import pandas as pd

from scripts.prepare_labelled_clusters_data import (
    clean_labelled_clusters,
    drop_strict_duplicates,
    split_regression_dataset,
)


class TestPrepareLabelledClustersData(unittest.TestCase):
    def _raw_frame(self):
        return pd.DataFrame(
            {
                "Entry": [1, 2, 3, 4, 5],
                "Name": [
                    "AISI 4023 Steel",
                    "AISI 4027 Steel",
                    "AISI 4023 Steel",
                    "AISI 1050 Steel",
                    "AISI 1040 Steel",
                ],
                "Processing condition": [
                    "Oil quenched, 150C temper",
                    "Water quenched, 540C temper",
                    "Oil quenched, 150C temper",
                    "Normalized",
                    "Annealed",
                ],
                "Cluster Number (0 to 11)": [0, 10, 0, 4, 7],
                "Al": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Cu": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Mn": [0.8, 0.8, 0.8, 0.7, 0.6],
                "N": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Ni": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Ti": [0.0, 0.0, 0.0, 0.0, 0.0],
                "S": [0.04, 0.04, 0.04, 0.03, 0.02],
                "Fe": [98.4, 98.35, 98.4, 98.5, 98.6],
                "Zr": [0.0, 0.0, 0.0, 0.0, 0.0],
                "P": [0.035, 0.035, 0.035, 0.03, 0.02],
                "Si": [0.25, 0.25, 0.25, 0.2, 0.2],
                "V": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Mo": [0.25, 0.25, 0.25, 0.2, 0.1],
                "Co": [0.0, 0.0, 0.0, 0.0, 0.0],
                "C": [0.225, 0.275, 0.225, 0.34, 0.46],
                "Nb": [0.0, 0.0, 0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Cr": [0.0, 0.0, 0.0, 0.0, 0.0],
                "(Ultimate) Tensile strength (MPa)": [745, 689, 745, 700, 620],
                "Yield strength (MPa)": [440, 510, 440, 420, 350],
                "Ductility (%)": [21, 25, 21, 20, 30],
                "Cluster Number": [0, 10, 0, 4, 7],
                "Cluster label": [
                    "Carburized",
                    "Water quenched and tempered",
                    "Carburized",
                    "Normalized",
                    "Annealed",
                ],
            }
        )

    def test_clean_labelled_clusters_preserves_rows_and_generates_actions(self):
        raw = self._raw_frame().iloc[:2]

        cleaned = clean_labelled_clusters(raw)

        self.assertEqual(len(cleaned), 2)
        self.assertIn("actions", cleaned.columns)
        self.assertEqual(cleaned.loc[0, "Text"], "Oil quenched, 150C temper")
        self.assertEqual(cleaned.loc[0, "actions"], "Oil quenched, 150C temper")
        self.assertEqual(cleaned.loc[1, "Tensile_value"], 689.0)
        self.assertEqual(cleaned.loc[1, "Yield_value"], 510.0)
        self.assertEqual(cleaned.loc[1, "Elongation_value"], 25.0)
        self.assertEqual(cleaned.loc[0, "cluster_number"], 0)
        self.assertEqual(cleaned.loc[1, "cluster_label"], "Water quenched and tempered")
        self.assertEqual(cleaned.loc[0, "H"], 0.0)
        self.assertEqual(cleaned.loc[0, "Fe"], 98.4)

    def test_drop_strict_duplicates_removes_identical_training_records(self):
        cleaned = clean_labelled_clusters(self._raw_frame())

        deduped, duplicate_rows, duplicate_groups = drop_strict_duplicates(cleaned)

        self.assertEqual(len(cleaned), 5)
        self.assertEqual(len(deduped), 4)
        self.assertEqual(len(duplicate_rows), 2)
        self.assertEqual(len(duplicate_groups), 1)
        self.assertEqual(duplicate_groups.loc[0, "duplicate_count"], 2)
        self.assertEqual(duplicate_groups.loc[0, "source_entries"], "1,3")

    def test_split_regression_dataset_uses_fixed_8_2_split(self):
        cleaned = clean_labelled_clusters(self._raw_frame())
        deduped, _, _ = drop_strict_duplicates(cleaned)

        train_df, val_df = split_regression_dataset(deduped, train_ratio=0.8, seed=42)

        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(set(train_df.index), set(range(len(train_df))))
        self.assertEqual(set(val_df.index), set(range(len(val_df))))


if __name__ == "__main__":
    unittest.main()
