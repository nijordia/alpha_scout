import json
import os
from typing import Dict, List, Any
from datetime import datetime

class UserPreferencesManager:
    """
    Manages user preferences for stock tracking and signal types
    """
    def __init__(self, storage_path="user_preferences"):
        """
        Initialize the preferences manager
        
        Parameters:
        -----------
        storage_path : str
            Path to the directory where user preferences will be stored
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a specific user
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
            
        Returns:
        --------
        dict
            User preferences including tracked stocks and signal types
        """
        file_path = os.path.join(self.storage_path, f"{user_id}.json")
        
        if not os.path.exists(file_path):
            # Return default preferences for new users
            return {
                "tracked_stocks": [],
                "signal_types": ["mean_reversion"],
                "notification_time": "08:00",  # Default notification time (UTC)
                "signal_params": {
                    "mean_reversion": {
                        "window": 50,
                        "threshold": 1.5
                    }
                },
                "created_at": "",
                "last_modified": ""
            }
        
        try:
            with open(file_path, 'r') as f:
                preferences = json.load(f)
                
                # Ensure signal_params exist even in older preference files
                if "signal_params" not in preferences:
                    preferences["signal_params"] = {
                        "mean_reversion": {
                            "window": 50,
                            "threshold": 1.5
                        }
                    }
                
                return preferences
        except Exception as e:
            print(f"Error loading user preferences: {e}")
            return {
                "tracked_stocks": [],
                "signal_types": ["mean_reversion"],
                "notification_time": "08:00",
                "signal_params": {
                    "mean_reversion": {
                        "window": 50,
                        "threshold": 1.5
                    }
                }
            }
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Save preferences for a specific user
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        preferences : dict
            User preferences to save
            
        Returns:
        --------
        bool
            True if preferences were saved successfully, False otherwise
        """
        file_path = os.path.join(self.storage_path, f"{user_id}.json")
        
        try:
            # Add timestamps
            now = datetime.now().isoformat()
            
            # Check if the file already exists to determine if this is a new user
            if not os.path.exists(file_path):
                preferences["created_at"] = now
            
            preferences["last_modified"] = now
            
            with open(file_path, 'w') as f:
                json.dump(preferences, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving user preferences: {e}")
            return False

    def add_stock_for_user(self, user_id: str, stock_symbol: str) -> bool:
        """
        Add a stock to a user's tracked stocks
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        stock_symbol : str
            Stock symbol to add (e.g., 'AAPL')
            
        Returns:
        --------
        bool
            True if stock was added successfully, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        
        # Convert to uppercase for consistency
        stock_symbol = stock_symbol.upper()
        
        # Add stock if not already tracked
        if stock_symbol not in preferences.get("tracked_stocks", []):
            preferences.setdefault("tracked_stocks", []).append(stock_symbol)
            return self.save_user_preferences(user_id, preferences)
        
        return True  # Stock was already in the list
    
    def remove_stock_for_user(self, user_id: str, stock_symbol: str) -> bool:
        """
        Remove a stock from a user's tracked stocks
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        stock_symbol : str
            Stock symbol to remove
            
        Returns:
        --------
        bool
            True if stock was removed successfully, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        
        # Convert to uppercase for consistency
        stock_symbol = stock_symbol.upper()
        
        # Remove stock if tracked
        if stock_symbol in preferences.get("tracked_stocks", []):
            preferences["tracked_stocks"].remove(stock_symbol)
            return self.save_user_preferences(user_id, preferences)
        
        return True  # Stock was not in the list
    
    def update_notification_time(self, user_id: str, notification_time: str) -> bool:
        """
        Update notification time for a user
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        notification_time : str
            Time for daily notifications in 'HH:MM' format (UTC)
            
        Returns:
        --------
        bool
            True if notification time was updated successfully, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        preferences["notification_time"] = notification_time
        return self.save_user_preferences(user_id, preferences)
    
    def update_signal_types(self, user_id: str, signal_types: List[str]) -> bool:
        """
        Update signal types for a user
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        signal_types : list
            List of signal types (e.g., ['mean_reversion', 'momentum'])
            
        Returns:
        --------
        bool
            True if signal types were updated successfully, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        preferences["signal_types"] = signal_types
        return self.save_user_preferences(user_id, preferences)
    
    def update_signal_params(self, user_id: str, signal_type: str, params: Dict[str, Any]) -> bool:
        """
        Update parameters for a specific signal type
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        signal_type : str
            Type of signal (e.g., 'mean_reversion')
        params : dict
            Parameters for the signal type
            
        Returns:
        --------
        bool
            True if parameters were updated successfully, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        
        # Initialize signal_params if it doesn't exist
        if "signal_params" not in preferences:
            preferences["signal_params"] = {}
        
        # Initialize parameters for this signal type if it doesn't exist
        if signal_type not in preferences["signal_params"]:
            preferences["signal_params"][signal_type] = {}
        
        # Update the parameters
        for key, value in params.items():
            preferences["signal_params"][signal_type][key] = value
        
        return self.save_user_preferences(user_id, preferences)
    
    def get_signal_params(self, user_id: str, signal_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific signal type
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
        signal_type : str
            Type of signal (e.g., 'mean_reversion')
            
        Returns:
        --------
        dict
            Parameters for the signal type
        """
        preferences = self.get_user_preferences(user_id)
        
        # Return default parameters if not set
        if "signal_params" not in preferences or signal_type not in preferences["signal_params"]:
            if signal_type == "mean_reversion":
                return {
                    "window": 50,
                    "threshold": 1.5
                }
            
            # Add defaults for other signal types here
            
            return {}
        
        return preferences["signal_params"][signal_type]
    
    def get_all_users(self) -> List[str]:
        """
        Get a list of all user IDs
        
        Returns:
        --------
        list
            List of user IDs
        """
        try:
            users = []
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    users.append(filename.split('.')[0])
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
            return []
    
    def get_all_tracked_stocks(self) -> List[str]:
        """
        Get a list of all stocks tracked by all users
        
        Returns:
        --------
        list
            List of unique stock symbols
        """
        all_stocks = set()
        
        for user_id in self.get_all_users():
            preferences = self.get_user_preferences(user_id)
            all_stocks.update(preferences.get("tracked_stocks", []))
        
        return list(all_stocks)