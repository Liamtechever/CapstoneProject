import os
from dotenv import load_dotenv

# Force-load the correct .env file
dotenv_path = "secrets.env"
load_dotenv(dotenv_path)
print(os.getcwd())

# # Debugging: Print environment variables
# print("\n🔍 Checking loaded environment variables:")
# for key, value in os.environ.items():
#     if "OPENAI" in key:
#         print(f"{key}: {value[:5]}... (hidden for security)")

# Retrieve API Key
api_key = os.getenv("OPEN_API_KEY")

if not api_key:
    raise ValueError("🚨 API key still not found. Ensure secrets.env is loaded.")

print("\n✅ API Key successfully loaded!")
