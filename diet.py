from datetime import datetime, timedelta, UTC 
import json
import re
import sqlite3
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Optional, Union
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv  
import base64

load_dotenv()
 
openai_client = OpenAI()

# --- Database Implementation for SQLite ---
class db:
    _conn = None

    @staticmethod
    def get_connection():
        if db._conn is None:
            db._conn = sqlite3.connect("diet.db", check_same_thread=False, timeout=30)
            db._conn.row_factory = sqlite3.Row
        return db._conn

    @staticmethod
    def insert(table, data):
        """ Insert a row into a table 
        Args:
            table (str): The name of the table to insert the row into.
            data (dict): A dictionary containing the data to insert into the table.
        Returns:
            None
        """
        conn = db.get_connection()
        keys = data.keys()
        values = [data[k] for k in keys]
        placeholders = ', '.join(['?'] * len(keys))
        columns = ', '.join(keys)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})" 
        with conn:
            conn.execute(sql, values)

    @staticmethod
    def query(sql, params):
        """ Query the database with a SQL statement and parameters.
        Args:
            sql (str): The SQL statement to execute.
            params (tuple): A tuple of parameters to pass to the SQL statement.
        Returns:
            A list of dictionaries, each representing a row in the result set.
        """
        conn = db.get_connection()
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def query_one(sql, params):
        conn = db.get_connection()
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else {}

# --- Pydantic Models ---
class FoodItem(BaseModel):
    name: str
    quantity: float
    unit: str = Field(..., description="""
    A unit of measurement for the food item or the word 'item' if it is not a measurable quantity.
    Examples: 'oz', 'g', 'ml', 'item', 'tbsp', 'tsp', 'cup', 'pinch', 'can', 'pkg', 'lb', 'oz', 'lb', 'item'. 
    Example: if three eggs is the message, the quantiy will be 3 and the unit will be 'item'""")

    total_calories: float
    meal_type: Literal['breakfast', 'lunch', 'dinner', 'snack', 'unspecified']

class LLMResponsePicture(BaseModel):
    food_items: List[FoodItem] = Field(..., description="""
    A list of food items consumed by the user in the provided image, if none, return an empty list. 
    If you are confused ask for clarification in your response.""")

    response: Optional[str] = Field(None, description="Only if you need clarification.")

class LLMResponseText(BaseModel):
    food_items: List[FoodItem] = Field(..., description="""
    A list of food items consumed by the user in the current message, if none, return an empty list.
    Only include food items that are mentioned in the user's message that they ate.  
    Do not include food items that are suggestions or recommendations.  If you are confused ask for clarification in your response.""")

    response: Optional[str] = Field(None, description="A response to the user's message")

class FullResponse(LLMResponseText):
    user_message_id:Optional[int] = Field(None, description="The ID of the user's message")
    assistant_message_id:Optional[int] = Field(None, description="The ID of the assistant's message")
    error_message: Optional[str] = Field(None, description="An error message to the user if the response is not valid") 

class Message(BaseModel):
    role: str
    timestamp: datetime
    message: str
    id: int
# --- Utility Functions ---
def get_recent_chat(user_id, application_id, limit=10):
    since = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    messages = db.query("""
        SELECT role, content, timestamp, id FROM messages
        WHERE user_id = ?
          AND application_id = ?
          AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, application_id, since, limit))
    return list(reversed(messages))

def build_chat_history_block(messages):
    """Build a chat history block for the chat history.
    
    Args:
        messages: A list of messages
        
    Returns:
        A JSON string containing the chat history"""
    
    message_list = []
    for msg in messages:
        m = Message(
            role=  msg["role"],
            timestamp=msg.get("timestamp", datetime.now(UTC)),
            message=msg["content"],
            id=msg["id"]
        )
        message_list.append(m)
    return json.dumps([m.model_dump() for m in message_list], default=str)

def extract_json_block(text):
    """Extract a JSON block from the text returned by the LLM.
    
    Args:
        text: A string containing the text
        
    Returns:
        A JSON string containing the JSON block"""
    
    match = re.search(r"```{(.*)}```", text, re.DOTALL)
    if match:
        return "{" + match.group(1).strip() + "}"
    match = re.search(r"{.*}", text, re.DOTALL)
    return match.group(0) if match else None

def get_pic_prompt(schema):
    return f"""
You are a helpful dietary assistant.  
The user has sent you an image of what they are eating. 

Look at the image and extract the food items and their quantities. 
  
---

### 🧾 Response Format

Return **only** a JSON object that matches the following schema, enclosed in triple backticks:

```{{ your JSON here }}```

### JSON Schema
<schema>
{schema}
</schema>
"""

def get_text_prompt(content, restrictions, goal, consumed_calories, 
                    chat_history, schema, foods_consumed, foods_from_picture_json):
    
    if foods_from_picture_json:
        foods_from_picture_prompt_part = f"""
        ### 🍽️ Foods from Picture
        <foods_from_picture>
        {foods_from_picture_json}
        </foods_from_picture>
        """
        foods_from_picture_rule_part = "7. include foods_from_picture in your processing.  \
            these were provided in a picture that was analized in a previous steop."
    else:
        foods_from_picture_prompt_part = ""
        foods_from_picture_rule_part = ""

    return f"""
You are a helpful dietary assistant integrated into a chatbot on a website. Your role is to track the user's food intake and ensure they stay on target to meet their daily calorie goal.

You will receive a message from the user. This message may:
- Mention food they've eaten.
- Ask to adjust calories consumed (e.g., to correct an earlier entry).
- Include conversational elements or questions.

Your tasks:
1. If the message contains food items, extract them and return a JSON object describing each one.
2. If the message is an adjustment, return a food item with **negative calories** to reflect the correction.
3. Respond with a conversational reply **only if appropriate**, but keep it minimal or omit it entirely if the message is strictly functional.
4. If you see a trend in the eating that could be a problem, respond with a subtile warning and add some humor if possible.
5. Use **a few well-placed emojis** to make the response friendly and engaging — but keep it subtle.
6. If the user is eating a lot of calories, ask them what is going on.  Are they not sleeping?  Under a lot of stress? Change in medication? If they provide a resson, let them know you recored it and will look for a trend on the montly review. 
{foods_from_picture_rule_part}

---

### 📜 Recent Chat History
<transcript>
{chat_history}
</transcript>


### Foods Already Consumed Today
<foods_already_consumed>
{foods_consumed}
</foods_already_consumed>


### 💬 Latest Message from the User
<latest_message>
{content}
</latest_message>

{foods_from_picture_prompt_part}

### 👤 User Profile
<user_information>
    <restrictions>{restrictions}</restrictions>
    <daily_calorie_goal>{goal}</daily_calorie_goal>
    <calories_already_consumed_today>{consumed_calories}</calories_already_consumed_today>
</user_information>

---

### 🧾 Response Format

Return **only** a JSON object that matches the following schema, enclosed in triple backticks:

```{{ your JSON here }}```

### JSON Schema
<schema>
{schema}
</schema>
"""


def store_interaction(user_id: str, application_id: str, content: str, response: LLMResponseText) -> dict:
    """Store the user interaction and any food items in the database.
    
    Args:
        user_id: The ID of the user
        application_id: The ID of the application
        content: The original user message
        response: The validated LLMResponse containing the assistant's response and food items
    """
    # Store the conversation messages
    user_message_id = add_message(user_id, application_id, content, "user")
    
    assistant_message_id = None
    if response.response:
        assistant_message_id = add_message(user_id, application_id, response.response, "assistant")
    
    # Store any food items mentioned
    for item in response.food_items:
        add_food_item(
            user_id,
            application_id,
            user_message_id,
            item.name,
            item.quantity,
            item.unit,
            item.total_calories,
            item.meal_type
        )
    return {"user_message_id": user_message_id, "assistant_message_id": assistant_message_id}

def log_to_db(prompt, response):
    db.insert("log", {"prompt": prompt, "response": response})
    clean_log()

def clean_log():
    # keep the last 10 rows and delete the rest
    db.query("DELETE FROM log WHERE id NOT IN (SELECT id FROM log ORDER BY id DESC LIMIT 10)",())


def format_response(response: FullResponse) -> str:
    """Format the response into a markdown table and message.
    
    Args:
        response: The FullResponse object containing either success or error data
        
    Returns:
        A formatted string containing the response
    """
    if response.error_message:
        return response.error_message

    message = ""
    if response.food_items:
        # Create a markdown table for the food items
        table = "| Food Item | Quantity | Unit | Calories | Meal Type |\n"
        table += "|-----------|----------|------|----------|------------|\n"
        for item in response.food_items:
            table += f"| {item.name} | {item.quantity} | {item.unit} | {item.total_calories} | {item.meal_type} |\n"
        message = f"{table}\n\n"
    
    message += response.response
    return message

def process_user_message(content, foods_from_picture_json, user_id, application_id) -> FullResponse:  
    """Process the user's message and return a response.
    
    Args:
        content: The user's message
        user_id: The ID of the user
        application_id: The ID of the application
        
    Returns:
        A FullResponse object containing either:
        - The assistant's response and food items on success
        - An error message if processing failed
    """
    
    MAX_RETRIES = 3
    user = db.query_one("SELECT dietary_restrictions, calorie_goal_per_day FROM users WHERE id = ?", (user_id,))
    restrictions = user.get("dietary_restrictions", "")
    goal = user.get("calorie_goal_per_day", 2000)

    today = datetime.now(UTC).date().isoformat()
    result = db.query_one("""
        SELECT SUM(calories) AS total FROM meal_entries
        WHERE user_id = ? AND application_id = ? AND strftime('%Y-%m-%d', timestamp) = ?
    """, (user_id, application_id, today))
    consumed_calories = result['total'] or 0

    recent = get_recent_chat(user_id, application_id)
    chat_history = build_chat_history_block(recent)
    schema = json.dumps(LLMResponseText.model_json_schema(), indent=2)
    foods_consumed = get_meal_info(user_id, application_id)
    foods_consumed = json.dumps(foods_consumed, indent=2)

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = get_text_prompt(content, restrictions, goal, consumed_calories, 
                                 chat_history, schema, foods_consumed, foods_from_picture_json)
 
        response = openai_client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )
 
        json_response = response.to_json()
        log_to_db(prompt, json_response)

        raw = response.choices[0].message.content
        json_text = extract_json_block(raw)

        try:
            parsed = json.loads(json_text)
            validated = LLMResponseText(**parsed)
            ids = store_interaction(user_id, application_id, content, validated)
            return FullResponse(
                user_message_id=ids["user_message_id"],
                assistant_message_id=ids["assistant_message_id"],
                **validated.model_dump()
            )
        except (json.JSONDecodeError, ValidationError) as e:
            content += f"\n\nThe last response could not be parsed due to this error:\n{str(e)}"
            if attempt == MAX_RETRIES:
                return FullResponse(
                    error_message="I'm having trouble understanding that. Please rephrase.",
                    food_items=[],
                    user_message_id=None,
                    assistant_message_id=None
                )
 
def process_image(image_path: str) -> FullResponse:
    """Process an image using ChatGPT's vision capabilities to identify its contents.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A dictionary containing the identified contents of the image
    """
    try:
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Create the prompt for image analysis
        schema = json.dumps(LLMResponsePicture.model_json_schema(), indent=2)
        prompt =  get_pic_prompt(schema)
        MAX_RETRIES = 1
        for attempt in range(1, 1 + MAX_RETRIES):    
        # Call the OpenAI API with the image
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
        
            # Extract and parse the JSON response
            json_response = response.to_json()
            log_to_db(prompt, json_response)

            raw = response.choices[0].message.content
            json_text = extract_json_block(raw)

            try:
                parsed = json.loads(json_text)
                validated = LLMResponsePicture(**parsed) 
                return FullResponse(
                    user_message_id=None,
                    assistant_message_id=None,
                    **validated.model_dump()
                )
            except (json.JSONDecodeError, ValidationError) as e:
                content += f"\n\nThe last response could not be parsed due to this error:\n{str(e)}"
                if attempt == MAX_RETRIES:
                    return FullResponse(
                        error_message="I'm having trouble understanding that. Please rephrase.",
                        food_items=[],
                        user_message_id=None,
                        assistant_message_id=None
                    )


        
    except Exception as e:
        return {
            "error": str(e),
            "food_items": [],
            "notes": "Failed to process image"
        }

def add_message(user_id: str, application_id: str, content: str, role: str = "user") -> int:
    """Add a message to the messages table and return its ID.
    
    Args:
        user_id (str): The ID of the user sending the message
        application_id (str): The ID of the application
        content (str): The content of the message
        role (str): The role of the message sender (default: "user")
    
    Returns:
        int: The ID of the newly inserted message
    """
    conn = db.get_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (user_id, application_id, content, role, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, application_id, content, role, datetime.now(UTC).isoformat())
        )
        return cursor.lastrowid
 
def add_food_item(
    user_id: str,
    application_id: str,
    message_id: int,
    name: str,
    quantity: float,
    unit: str,
    total_calories: float,
    meal_type: str
) -> int:
    """Add a food item to the user's meal tracking.
    
    Args:
        user_id: The ID of the user
        application_id: The ID of the application
        message_id: The ID of the message
        name: Name of the food item
        quantity: Amount of food
        unit: Unit of measurement
        total_calories: Total calories in the food item
        meal_type: Type of meal (breakfast, lunch, dinner, snack, unspecified)
    
    Returns:
        The ID of the newly inserted food item
    """
    with db._conn:
        cursor = db._conn.cursor()
        cursor.execute(
            "INSERT INTO meal_entries \
            (user_id, application_id, message_id, food_name, quantity, unit, calories, meal_type, timestamp) VALUES \
            (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, application_id, message_id, name, quantity, unit, total_calories, meal_type, datetime.now(UTC).isoformat())
        )
        return cursor.lastrowid
 
def get_calorie_summary(user_id: str, application_id: str) -> dict:
    """Get the total calories consumed today and remaining calories for the user.
    
    Args:
        user_id: The ID of the user
        application_id: The ID of the application
    
    Returns:
        A dictionary containing total_calories_today, remaining_calories, and daily_goal
    """
    today = datetime.now(UTC).date().isoformat()
    
    result = db.query_one("""
        SELECT SUM(calories) AS total 
        FROM meal_entries 
        WHERE user_id = ? 
        AND application_id = ? 
        AND date(timestamp) = ?
    """, (user_id, application_id, today)) 
    
    total_calories = result['total'] or 0
    
    user = db.query_one(
        "SELECT calorie_goal_per_day FROM users WHERE id = ?", 
        (user_id,)
    )
    daily_goal = user.get('calorie_goal_per_day', 2000)
    
    remaining_calories = daily_goal - total_calories


    return {
        "total_calories_today": total_calories,
        "remaining_calories": remaining_calories,
        "daily_goal": daily_goal
    }

def get_meal_info(user_id: str, application_id: str) -> list:
 

    today = datetime.now(UTC).date().isoformat()
    # get the meals for the day
    meals = db.query("""
    SELECT food_name, quantity, unit, calories, meal_type FROM meal_entries
    WHERE user_id = ? AND application_id = ? AND strftime('%Y-%m-%d', timestamp) = ?
    ORDER BY timestamp ASC
    """, (user_id, application_id, today)) 
 
    return [dict(meal) for meal in meals]


def test(p:list):
    if "one" in p:
        message = "6 slices of bacon for breakfast and 2 sunny side eggs.  one cup of coffee with cream."
        responseObject =  process_user_message(message, "1", "1") 
 
    if "two" in p:
         message_id = add_message("1", "1", "test", "assistant")

    if("print" in p):
        print(json.dumps(responseObject.model_dump(), indent=2 ))

if __name__ == "__main__":
    test(["one", "print"])