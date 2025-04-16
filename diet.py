from flask import Flask, request, jsonify
from datetime import datetime, timedelta, UTC 
import json
import re
import sqlite3
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Optional
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
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
    unit: str = Field(..., description=\
                      "A unit of measurement for the food item or the word 'item' if it is not a measurable quantity.\
                      Examples: 'oz', 'g', 'ml', 'item', 'tbsp', 'tsp', 'cup', 'pinch', 'can', 'pkg', 'lb', 'oz', 'lb', 'item'. \
                      Example: if three eggs is the message, the quantiy will be 3 and the unit will be 'item'")
    total_calories: float
    meal_type: Literal['breakfast', 'lunch', 'dinner', 'snack', 'unspecified']

class LLMResponse(BaseModel):
    food_items: List[FoodItem] = Field(..., description="A list of food items consumed by the user, if none, return an empty list")
    response: Optional[str] = Field(None, description="A response to the user's message")

class Message(BaseModel):
    role: str
    timestamp: datetime
    message: str

# --- Utility Functions ---
def get_recent_chat(user_id, application_id, limit=20):
    since = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    messages = db.query("""
        SELECT role, content, timestamp FROM messages
        WHERE user_id = ?
          AND application_id = ?
          AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, application_id, since, limit))
    return list(reversed(messages))

def build_chat_history_block(messages):
    message_list = []
    for msg in messages:
        m = Message(
            role=  msg["role"],
            timestamp=msg.get("timestamp", datetime.now(UTC)),
            message=msg["content"]
        )
        message_list.append(m)
    return json.dumps([m.model_dump() for m in message_list], default=str)

def extract_json_block(text):
    match = re.search(r"```{(.*)}```", text, re.DOTALL)
    if match:
        return "{" + match.group(1).strip() + "}"
    match = re.search(r"{.*}", text, re.DOTALL)
    return match.group(0) if match else None

def get_prompt(content, restrictions, goal, consumed, chat_history, schema):
    return f"""
You are a helpful dietary assistant.

Below is a transcript of the recent conversation:
{chat_history}

Now, based on the user's latest message:
\"\"\"{content}\"\"\"

...extract any food items mentioned and respond in a conversational way.

User dietary restrictions: <restrictions> {restrictions} </restrictions>
Daily calorie goal: <calorie_goal>{goal}</caloire_goal>
Calories already consumed today: <calories_consumed_today>{consumed}</calories_consumed_today>

Respond with ONLY a JSON object, enclosed in triple backticks:
```{{ your JSON here }}```

that conforms to this JSON schema:
{schema}
""" 

def process_user_message(content, user_id, application_id):
    MAX_RETRIES = 3
    user = db.query_one("SELECT dietary_restrictions, calorie_goal_per_day FROM users WHERE id = ?", (user_id,))
    restrictions = user.get("dietary_restrictions", "")
    goal = user.get("calorie_goal_per_day", 2000)

    today = datetime.now(UTC).date().isoformat()
    result = db.query_one("""
        SELECT SUM(calories) AS total FROM meal_entries
        WHERE user_id = ? AND application_id = ? AND strftime('%Y-%m-%d', timestamp) = ?
    """, (user_id, application_id, today))
    consumed = result['total'] or 0

    recent = get_recent_chat(user_id, application_id)
    chat_history = build_chat_history_block(recent)
    schema = json.dumps(LLMResponse.model_json_schema(), indent=2)

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = get_prompt(content, restrictions, goal, consumed, chat_history, schema)
 
        response = openai_client.chat. completions.create(
            model = "gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )

        print(response)
        raw = response.choices[0].message.content
        # extract the json text from the raw response
        json_text = extract_json_block(raw)

        try:
            # parse the json text to a dictionary   
            parsed = json.loads(json_text)
            # validate the dictionary against the LLMResponse schema and return the validated python object
            validated = LLMResponse(**parsed) 
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            content += f"\n\nThe last response could not be parsed due to this error:\n{str(e)}"
            if attempt == MAX_RETRIES:
                return "I'm having trouble understanding that. Please rephrase.", []

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

def add_food_item(user_id: int, application_id: int,message_id:int, food_name: str, quantity: float, unit: str, total_calories: float, meal_type: str) -> int:
    """Add a food item to the meal_entries table and return its ID.
    
    Args:
        user_id (str): The ID of the user
        application_id (str): The ID of the application
        message_id (int): The ID of the message
        food_name (str): Name of the food item
        quantity (float): Amount of food
        unit (str): Unit of measurement
        total_calories (float): Total calories in the food item
        meal_type (str): Type of meal (breakfast, lunch, dinner, snack, unspecified)
    
    Returns:
        int: The ID of the newly inserted food item
    """
    with db._conn:
        cursor = db._conn.cursor()
        cursor.execute(
            "INSERT INTO meal_entries \
            (user_id, application_id, message_id, food_name, quantity, unit, calories, meal_type, timestamp) VALUES \
            (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, application_id, message_id, food_name, quantity, unit, total_calories, meal_type, datetime.now(UTC).isoformat())
        )
        return cursor.lastrowid

def test(p:list):
    if "one" in p:
        message = "6 slices of bacon for breakfast and 2 sunny side eggs.  one cup of coffee with cream."
        responseObject =  process_user_message(message, "1", "1")
        user_message_id = add_message("1", "1", message, "user")
        if responseObject.response:
            assistant_message_id = add_message("1", "1", responseObject.response, "assistant")
        else:
            assistant_message_id = None 

        for item in responseObject.food_items:
            add_food_item("1", "1", user_message_id, item.name, item.quantity, item.unit, item.total_calories, item.meal_type)
    if "two" in p:
         message_id = add_message("1", "1", "test", "assistant")
    if("print" in p):
        print(json.dumps(responseObject.model_dump(), indent=2 ))

if __name__ == "__main__":
    test(["one", "print"])