import unittest
from src.bot.notification_manager import NotificationManager
from src.bot.handlers import handle_start, handle_help

class TestBot(unittest.TestCase):

    def setUp(self):
        self.notification_manager = NotificationManager()

    def test_handle_start(self):
        response = handle_start()
        self.assertEqual(response, "Welcome to the Market Signals Bot! Use /help to see available commands.")

    def test_handle_help(self):
        response = handle_help()
        expected_response = (
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/signals - Get market signals"
        )
        self.assertEqual(response, expected_response)

    def test_send_notification(self):
        user_id = 123456789
        message = "Test notification"
        result = self.notification_manager.send_notification(user_id, message)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()