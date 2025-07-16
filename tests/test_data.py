import unittest
from src.data.fetcher import fetch_data
from src.data.data_processor import process_data

class TestDataProcessing(unittest.TestCase):

    def test_fetch_data(self):
        data = fetch_data("some_api_endpoint")
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

    def test_process_data(self):
        raw_data = {"price": [100, 101, 102], "volume": [10, 15, 20]}
        processed_data = process_data(raw_data)
        self.assertIn("average_price", processed_data)
        self.assertEqual(processed_data["average_price"], 101)  # Example expected value

if __name__ == '__main__':
    unittest.main()