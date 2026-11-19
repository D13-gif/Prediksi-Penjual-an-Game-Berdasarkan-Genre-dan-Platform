import unittest
from pathlib import Path

from app.services import ModelService


class PredictionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService(Path(__file__).resolve().parents[1])

    def test_fractional_year_is_rejected(self):
        cleaned, errors = self.service.validate_input(
            {
                "Platform": "PS4",
                "Genre": "Action",
                "Publisher": "Nintendo",
                "Year": "2015.6",
            }
        )

        self.assertIsNone(cleaned)
        self.assertTrue(any("bilangan bulat" in error for error in errors))

    def test_nan_year_is_rejected(self):
        cleaned, errors = self.service.validate_input(
            {
                "Platform": "PS4",
                "Genre": "Action",
                "Publisher": "Nintendo",
                "Year": "NaN",
            }
        )

        self.assertIsNone(cleaned)
        self.assertTrue(any("bilangan bulat" in error for error in errors))

    def test_out_of_range_year_is_rejected(self):
        cleaned, errors = self.service.validate_input(
            {
                "Platform": "PS4",
                "Genre": "Action",
                "Publisher": "Nintendo",
                "Year": "1900",
            }
        )

        self.assertIsNone(cleaned)
        self.assertTrue(any("rentang wajar" in error for error in errors))

    def test_missing_required_input_is_rejected(self):
        cleaned, errors = self.service.validate_input(
            {
                "Platform": "",
                "Genre": "Action",
                "Publisher": "Nintendo",
                "Year": "2015",
            }
        )

        self.assertIsNone(cleaned)
        self.assertTrue(any("wajib diisi" in error for error in errors))

    def test_unknown_categories_remain_supported(self):
        cleaned, errors = self.service.validate_input(
            {
                "Platform": "Unreleased Platform",
                "Genre": "New Genre",
                "Publisher": "New Publisher",
                "Year": "2015",
            }
        )

        self.assertEqual(errors, [])
        self.assertEqual(cleaned["Year"], 2015)
        self.assertEqual(cleaned["Publisher"], "New Publisher")


if __name__ == "__main__":
    unittest.main()
