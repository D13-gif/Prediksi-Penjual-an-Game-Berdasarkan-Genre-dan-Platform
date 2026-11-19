import unittest
from pathlib import Path

from app import create_app


class TreeImageRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.app = create_app(cls.root)
        cls.client = cls.app.test_client()

    def test_tree_page_uses_read_only_image_route(self):
        response = self.client.get("/tree")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/tree/image", response.data)

    def test_tree_image_is_served_from_model_artifact(self):
        image_path = self.root / "data" / "models" / "decision_tree_viz.png"
        response = self.client.get("/tree/image", buffered=True)
        payload = response.get_data()
        status = response.status_code
        mimetype = response.mimetype
        response.close()

        self.assertEqual(status, 200)
        self.assertEqual(mimetype, "image/png")
        self.assertEqual(payload, image_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
