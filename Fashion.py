import sqlite3
import os
import json
import openai
import base64
import cv2  # Import OpenCV for displaying images
from Take_Clothing import TakePicture  # Uses the new module for image capture
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv("secrets.env")
openai.api_key = os.getenv("OPEN_API_KEY")
print("API Key loaded successfully")


# ------------------------------
# Image Encoding and Vision Model Integration
# ------------------------------
def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_image(image_path):
    """
    Processes an image with a vision model via the OpenAI API.
    Returns a description and key details of the clothing.

    Falls back to manual input if the API call fails.
    """
    base64_image = encode_image(image_path)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Adjust this model name as needed.
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Provide a detailed description of the clothing item including its key details and category (e.g., top, bottom, shoes)."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        ai_output = response.choices[0].message.content
        print("AI Vision Model Output:")
        print(ai_output)

        # For now, assume the AI output is the description.
        description = ai_output
        details = ""  # Optionally, parse additional details.

    except Exception as e:
        print("Error calling vision model:", e)
        description = input("Enter a description for the clothing (manual input): ")
        details = input("Enter key details for the clothing (manual input): ")

    return description, details


# ------------------------------
# Database Functions
# ------------------------------
def init_db(db_path='fashion.db'):
    """
    Initializes the SQLite database and creates the clothing table if it does not exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            description TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    return conn


def save_clothing_item(conn, image_path, description, details):
    """
    Saves a clothing item to the database.
    """
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clothing (image_path, description, details)
        VALUES (?, ?, ?)
    ''', (image_path, description, details))
    conn.commit()


def get_all_clothing_items(conn):
    """
    Retrieves all clothing items from the database.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clothing')
    return cursor.fetchall()


def delete_clothing_item(conn):
    """
    Displays all clothing items and prompts the user for the ID of the item to delete.
    Deletes the selected record from the database and (optionally) removes the image file.
    """
    items = get_all_clothing_items(conn)
    if not items:
        print("No clothing items found in the database.")
        return

    print("Clothing Items:")
    for item in items:
        print(f"ID: {item[0]}, Image Path: {item[1]}, Description: {item[2]}, Details: {item[3]}")
    print("-" * 40)

    try:
        id_to_delete = int(input("Enter the ID of the clothing item you want to delete: ").strip())
    except ValueError:
        print("Invalid ID. Please enter a numeric value.")
        return

    item_to_delete = next((item for item in items if item[0] == id_to_delete), None)
    if item_to_delete is None:
        print(f"No clothing item with ID {id_to_delete} was found.")
        return

    cursor = conn.cursor()
    cursor.execute("DELETE FROM clothing WHERE id = ?", (id_to_delete,))
    conn.commit()
    print(f"Clothing item with ID {id_to_delete} has been deleted from the database.")

    image_path = item_to_delete[1]
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image file {image_path} has also been deleted.")
    else:
        print(f"Image file {image_path} not found on disk.")


def display_all_clothing_items(conn):
    """
    Retrieves and displays all clothing items from the database.
    """
    items = get_all_clothing_items(conn)
    if not items:
        print("No clothing items found in the database.")
        return

    for item in items:
        print(f"ID: {item[0]}")
        print(f"Image Path: {item[1]}")
        print(f"Description: {item[2]}")
        print(f"Key Details: {item[3]}")
        print("-" * 40)


# ------------------------------
# AI-Based Full Outfit Generation Using OpenAI API
# ------------------------------
def ai_generate_full_outfit_with_openai(user_prompt, items):
    """
    Uses OpenAI's Chat API to select a full outfit from a list of clothing items.
    The outfit must include exactly one top, one bottom, and one pair of shoes.
    Each clothing item is described as: "ID {id}: {description} - {details}".
    Returns a JSON object with keys 'top', 'bottom', and 'shoes' containing the selected item IDs.
    """
    descriptions_list = []
    for item in items:
        descriptions_list.append(f"ID {item[0]}: {item[2]} - {item[3]}")
    combined_descriptions = "\n".join(descriptions_list)

    system_message = (
        "You are an AI assistant tasked with selecting a full outfit from a list of clothing items. "
        "The outfit must include exactly one top, one bottom, and one pair of shoes. "
        "Each clothing item is described as: 'ID {id}: {description} - {details}'. "
        "Return a JSON object with keys 'top', 'bottom', and 'shoes', each containing the corresponding item ID. "
        "If an item for a category is not available, return null for that key. "
        "Only return the JSON and no additional text."
    )

    prompt_message = (
        f"Clothing Items:\n{combined_descriptions}\n\n"
        f"Task: Based on the prompt '{user_prompt}', choose one clothing item for each category (top, bottom, shoes) to form a full outfit. "
        "Do not select more than one item for any category."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_message},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return {}

    print("\n--- AI Response ---")
    print(response_text)
    print("--- End of AI Response ---\n")

    try:
        result = json.loads(response_text)
        top_id = result.get("top")
        bottom_id = result.get("bottom")
        shoes_id = result.get("shoes")
    except Exception as e:
        print("Error parsing AI response:", e)
        return {}

    selected_items = {}
    if top_id is not None:
        selected_items["top"] = next((item for item in items if item[0] == top_id), None)
    if bottom_id is not None:
        selected_items["bottom"] = next((item for item in items if item[0] == bottom_id), None)
    if shoes_id is not None:
        selected_items["shoes"] = next((item for item in items if item[0] == shoes_id), None)
    return selected_items


def generate_full_outfit_with_ai(conn):
    """
    Retrieves clothing items from the database and uses the OpenAI API to generate a full outfit
    (one top, one bottom, one pair of shoes) based on the user's prompt.
    Displays the selected items along with their details and images.
    """
    items = get_all_clothing_items(conn)
    if not items:
        print("No clothing items found in the database.")
        return

    outfit_prompt = input("Enter the style or type of outfit you'd like (e.g., casual, sporty): ").strip()

    full_outfit = ai_generate_full_outfit_with_openai(outfit_prompt, items)

    print("\nFull Outfit Suggestion Based on Your Prompt:")
    for category, item in full_outfit.items():
        if item:
            print(f"\nCategory: {category.capitalize()}")
            print(f"Clothing Item ID: {item[0]}")
            print(f"Image Path: {item[1]}")
            print(f"Description: {item[2]}")
            print(f"Key Details: {item[3]}")
            # Load and display the image using OpenCV
            image = cv2.imread(item[1])
            if image is not None:
                window_title = f"{category.capitalize()} - {item[2]}"
                cv2.imshow(window_title, image)
                print("Displaying image. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_title)
            else:
                print("Could not load image.")
        else:
            print(f"\nCategory: {category.capitalize()} - No item available.")


# ------------------------------
# Main Application Logic
# ------------------------------
def main():
    conn = init_db()
    print("Welcome to Fashion.ai!\n")

    while True:
        print("\nOptions:")
        print("1. Add a new clothing item")
        print("2. View all clothing items")
        print("3. Get full outfit suggestion (top, bottom, shoes)")
        print("4. Delete a clothing item")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            image_path = TakePicture()  # Uses the new TakePicture function from Take_Clothing.py
            if image_path:
                description, details = process_image(image_path)
                save_clothing_item(conn, image_path, description, details)
                print("Clothing item saved to database.")
            else:
                print("Image capture failed.")
        elif choice == "2":
            display_all_clothing_items(conn)
        elif choice == "3":
            generate_full_outfit_with_ai(conn)
        elif choice == "4":
            delete_clothing_item(conn)
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    conn.close()


if __name__ == "__main__":
    main()
