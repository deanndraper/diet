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
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from memory import DietMemory
from app_tools import update_daily_calorie_limit
import logging

load_dotenv()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
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


# Initialize both OpenAI client (for vision) and LangChain chat model
openai_client = OpenAI()
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Define the tools the agent can use
tools = [update_daily_calorie_limit]

# Create output parser for our LLMResponseText model
response_parser = PydanticOutputParser(pydantic_object=LLMResponseText)

# Create prompt template
text_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful dietary assistant integrated into a chatbot on a website. 
    Your role is to track the user's food intake and ensure they stay on target to meet their daily calorie goal.
    
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
    
    {format_instructions}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """
    ### Foods Already Consumed Today
    <foods_already_consumed>
    {foods_consumed}
    </foods_already_consumed>

    ### 💬 Latest Message from the User
    <latest_message>
    {input}
    </latest_message>

    ### 👤 User Profile
    <user_information>
        <user_id>{user_id}</user_id>
        <restrictions>{restrictions}</restrictions>
        <daily_calorie_goal>{goal}</daily_calorie_goal>
        <calories_already_consumed_today>{consumed_calories}</calories_already_consumed_today>
    </user_information>
    """),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

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


# --- Utility Functions ---
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
 
def store_interaction(user_id: str, application_id: str, response: LLMResponseText) -> dict:
    """Store the user interaction and any food items in the database.
    
    Args:
        user_id: The ID of the user
        application_id: The ID of the application
        response: The validated LLMResponse containing the assistant's response and food items
    """
    # Store any food items mentioned
    if response.food_items:
        for item in response.food_items:
            add_food_item(
                user_id,
                application_id,
                item.name,
                item.quantity,
                item.unit,
                item.total_calories,
                item.meal_type
            )
    # We don't return message IDs anymore as history is handled by LangChain
    return {}

def log_to_db(prompt, response):
    db.insert("log", {"prompt": prompt, "response": response})
    clean_log()

def clean_log():
    # keep the last 10 rows and delete the rest
    db.query("DELETE FROM log WHERE id NOT IN (SELECT id FROM log ORDER BY id DESC LIMIT 10)",())


def format_response(response: Union[FullResponse, LLMResponseText]) -> str:
    """Format the response into a markdown table and message.
    
    Args:
        response: The FullResponse object containing either success or error data
        
    Returns:
        A formatted string containing the response
    """
    if hasattr(response, 'error_message') and response.error_message:
        return response.error_message

    message = ""
    # Check if food_items exists and is not None
    if hasattr(response, 'food_items') and response.food_items:
        # Create a markdown table for the food items
        table = "| Food Item | Quantity | Unit | Calories | Meal Type |\n"
        table += "|-----------|----------|------|----------|------------|\n"
        for item in response.food_items:
            table += f"| {item.name} | {item.quantity} | {item.unit} | {item.total_calories} | {item.meal_type} |\n"
        message = f"{table}\n\n"
    
    # Check if response attribute exists and is not None
    if hasattr(response, 'response') and response.response:
        message += response.response
        
    # If no structured response elements found, return a default message or handle raw content if needed
    if not message:
         # If response itself might be a string (fallback, shouldn't happen with parser)
         if isinstance(response, str):
              return response 
         return "Got it!" # Default if no text response and no food items
         
    return message

def process_user_message(user_id: str, application_id: str, message: str) -> str:
    """Process a user message using a LangChain Agent with history and return a formatted response string."""
    try:
        # 1. Initialize memory manager
        memory_manager = DietMemory(user_id, application_id)
        session_id = memory_manager.session_id
        
        # 2. Create the Agent
        # Ensure the prompt has the required input variables: input, chat_history, agent_scratchpad
        # Add agent_scratchpad placeholder if not present (create_openai_tools_agent adds it implicitly usually)
        # If text_prompt_template needs adjustment, it would happen here. Assuming it's okay for now.
        agent = create_openai_tools_agent(chat_model, tools, text_prompt_template)

        # 3. Create the Agent Executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Added verbose=True for debugging

        # 4. Wrap the executor with history management
        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            lambda sid: memory_manager.get_message_history(),
            input_messages_key="input",
            history_messages_key="chat_history",
            # AgentExecutor output includes 'output' key for final response
            # Ensure memory handles agent's intermediate steps if necessary (DietMemory might need update later)
        )
        
        # 5. Gather context for the prompt
        restrictions = "None" # Placeholder
        summary = get_calorie_summary(user_id, application_id)
        foods_consumed_list = get_meal_info(user_id, application_id)
        foods_consumed_str = "\n".join([f"- {f['food_name']} ({f['calories']} kcal)" for f in foods_consumed_list]) or "None logged yet today."

        # 6. Prepare input dictionary (agent needs 'input' and 'chat_history' primarily)
        # Format instructions are less critical here as agent handles JSON output/tool format
        invoke_input = {
            "input": message,
            # Pass other template variables expected by the prompt directly
            "format_instructions": response_parser.get_format_instructions(), # Still needed for the prompt template? Maybe not for agent. Let's keep it for now.
            "foods_consumed": foods_consumed_str,
            "restrictions": restrictions, 
            "goal": summary['daily_goal'],
            "consumed_calories": summary['total_calories_today'],
            "user_id": user_id
        }

        # 7. Prepare config for session management
        config = {"configurable": {"session_id": session_id}}

        # 8. Invoke the agent with history
        # The result is a dictionary, typically {'input': ..., 'chat_history': ..., 'output': 'Final response text'}
        agent_result = agent_with_history.invoke(invoke_input, config=config)
        
        # 9. Extract the final response from the agent's output
        final_response_text = agent_result.get('output', "Sorry, I couldn't generate a response.")

        # --- Parsing & Storing Logic (Needs Adjustment) ---
        # The agent's final 'output' is usually just the text response.
        # Extracting structured FoodItems after the agent runs requires a different approach.
        # Option 1: Parse the final_response_text itself (less reliable).
        # Option 2: Modify the agent/prompt to *always* return the structured JSON in the 'output'.
        # Option 3: Use the tool outputs recorded by the AgentExecutor (if food logging becomes a tool).
        
        # For now, let's skip the structured parsing and storing from the agent's text output.
        # We need to decide how food logging integrates with the agent (is it a separate tool?)
        # TODO: Re-evaluate food item parsing and storage with the agent setup.
        
        # logger.info(f"Agent raw result: {agent_result}") # Optional logging
        
        # 10. Return the formatted text response
        # Use a simpler formatting for now, just returning the text
        # return format_response(parsed_response) # Old formatting based on parsing
        return final_response_text # Return the direct agent output for now

    except Exception as e:
        logger.error(f"Error processing message with agent: {str(e)}", exc_info=True) # Log traceback
        return "I apologize, but I encountered an error while processing your message with the agent. Please try again."

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

def add_food_item(
    user_id: str,
    application_id: str,
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
        name: Name of the food item
        quantity: Amount of food
        unit: Unit of measurement
        total_calories: Total calories in the food item
        meal_type: Type of meal (breakfast, lunch, dinner, snack, unspecified)
    
    Returns:
        The ID of the newly inserted food item
    """
    conn = db.get_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO meal_entries "
            "(user_id, application_id, food_name, quantity, unit, calories, meal_type, timestamp) VALUES " 
            "(?, ?, ?, ?, ?, ?, ?, ?)", 
            (user_id, application_id, name, quantity, unit, total_calories, meal_type, datetime.now(UTC).isoformat())
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
        # Call the updated process_user_message
        responseString = process_user_message("1", "1", message)
        print(f"Formatted Response:\n{responseString}")
 
    if "two" in p:
         # add_food_item is now called within store_interaction
         print("Skipping direct add_food_item call in test.")
         # message_id = add_food_item("1", "1", None, "test", 0, "", 0, "unspecified")
 
        
    # if("print" in p): # Printing happens within the "one" block now
    #     print(json.dumps(responseObject.model_dump(), indent=2 ))

if __name__ == "__main__":
    # test(["one", "print"]) # "print" part is implicit now
    test(["one"])