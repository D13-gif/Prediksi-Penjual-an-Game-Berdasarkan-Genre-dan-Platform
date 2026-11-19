import unittest

import pandas as pd

from src.preprocessing import create_target


class TargetBoundaryTests(unittest.TestCase):
    def test_sales_categories_follow_inclusive_project_thresholds(self):
        sales = [0.499999, 0.5, 0.500001, 2.0, 2.000001]
        result = create_target(pd.DataFrame({"Global_Sales": sales}))

        self.assertEqual(
            result["Sales_Category"].astype(str).tolist(),
            ["Rendah", "Sedang", "Sedang", "Sedang", "Tinggi"],
        )

    def test_missing_sales_remains_unclassified(self):
        result = create_target(pd.DataFrame({"Global_Sales": [None]}))

        self.assertTrue(pd.isna(result["Sales_Category"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
