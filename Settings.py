import json
import os
import psutil
import GPUtil  # Install with: pip install gputil
import re

SETTINGS_FILE = "settings.json"

class Settings:
    def __init__(self, user_vram=None, use_internet=None, api_cost_tolerance=None):
        """Initializes settings with given or default values."""
        self.user_vram = user_vram if user_vram is not None else detect_system_vram()
        self.use_internet = use_internet if use_internet is not None else False
        self.api_cost_tolerance = api_cost_tolerance if api_cost_tolerance is not None else 0

    def __repr__(self):
        return f"Settings(user_vram={self.user_vram}, use_internet={self.use_internet}, api_cost_tolerance={self.api_cost_tolerance})"

def detect_system_vram():
    """Automatically detects the system's VRAM."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            vram_gb = int(gpus[0].memoryTotal / 1024)  # Convert MB to GB
            print(f"Detected GPU VRAM: {vram_gb}GB")
            return vram_gb
        else:
            print("No dedicated GPU detected. Using system RAM instead.")
            return int(psutil.virtual_memory().total / (1024 ** 3) * 0.5)  # Use 50% of system RAM as fallback
    except Exception as e:
        print(f"Error detecting VRAM: {e}. Using default 4GB.")
        return 4  # Default to 4GB if detection fails

def ask_yes_no(question):
    """Asks a Yes/No question and returns True for 'Yes' and False for 'No'."""
    while True:
        response = input(question).strip().lower()
        if response in ["yes", "no"]:
            return response == "yes"
        print("Please enter 'Yes' or 'No'.")

def ask_number(question, min_val, max_val):
    """Asks for a number within a range and ensures valid input."""
    while True:
        try:
            value = int(input(question).strip())
            if min_val <= max_val:
                return max(min(value, max_val), min_val)  # Ensures value is in range
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid integer.")

def load_settings():
    """Loads settings.json using regex, ensures correct data types, and returns a Settings object."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as file:
                content = file.read()  # Read the full file contents

            # Regex to match JSON (non-greedy match to prevent over-extraction)
            match = re.search(r"\{[\s\S]*?\}", content)

            if match:
                json_string = match.group(0)  # Extract the JSON portion
            else:
                print("No JSON found in file. Using empty JSON.")
                json_string = "{}"  # Default to empty JSON

            # Convert to dictionary with correct types
            try:
                settings_dict = json.loads(json_string)
                return Settings(
                    use_internet=bool(settings_dict.get("use_internet", False)),  # Ensure it's a bool
                    user_vram=int(settings_dict.get("user_vram", detect_system_vram())),  # Ensure it's an int
                    api_cost_tolerance=int(settings_dict.get("api_cost_tolerance", 0))  # Ensure it's an int
                )
            except (json.JSONDecodeError, ValueError):
                print("Error: Could not parse JSON. Using default values.")
                return ask_user_for_settings()

        except FileNotFoundError:
            print("settings.json not found. Running initial setup.")
            return ask_user_for_settings()

    else:
        print("\n No settings file found. Running initial setup...")
        return ask_user_for_settings()

def ask_user_for_settings():
    """Asks user for settings and saves them to a file."""
    settings = Settings(
        use_internet=ask_yes_no("Would you like to use internet-based models? (Yes/No): "),
        api_cost_tolerance=ask_number("On a scale from 0-10, how much are you willing to spend on API costs? (0 = none, 10 = highest cost): ", 0, 10)
    )
    save_settings(settings)
    return settings

def save_settings(settings):
    """Saves settings to settings.json."""
    with open(SETTINGS_FILE, "w") as file:
        json.dump({
            "use_internet": settings.use_internet,
            "user_vram": settings.user_vram,
            "api_cost_tolerance": settings.api_cost_tolerance
        }, file, indent=4)
    print(f"\n Settings saved to {SETTINGS_FILE}")

def format_settings_for_modelpicker():
    """Formats settings into a ModelPicker-compatible string."""
    settings = load_settings()
    settings_string = f"user_settings = Settings(user_vram={settings.user_vram}, use_internet={settings.use_internet}, api_cost_tolerance={settings.api_cost_tolerance})"
    print(settings_string)
    return settings_string  # Can be written to a file if needed

if __name__ == "__main__":
    format_settings_for_modelpicker()
