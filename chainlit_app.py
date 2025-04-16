import chainlit as cl
from diet import process_user_message, add_message, add_food_item
from datetime import datetime, UTC

@cl.on_chat_start
def start():
    cl.user_session.set("user_id", "1")  # Default user ID
    cl.user_session.set("application_id", "1")  # Default application ID

@cl.on_message
async def main(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    application_id = cl.user_session.get("application_id")
    
    # Process the message using the existing diet application logic
    response = process_user_message(message.content, user_id, application_id)
    
    # Add the user message to the database
    user_message_id = add_message(user_id, application_id, message.content, "user")
    
    # If there's a response message, add it to the database
    if response.response:
        add_message(user_id, application_id, response.response, "assistant")
    
    # Add any food items to the database
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
    
    # Send the response back to the user
    if response.food_items:
        # Create a markdown table for the food items
        table = "| Food Item | Quantity | Unit | Calories | Meal Type |\n"
        table += "|-----------|----------|------|----------|------------|\n"
        for item in response.food_items:
            table += f"| {item.name} | {item.quantity} | {item.unit} | {item.total_calories} | {item.meal_type} |\n"
        
        await cl.Message(content=table).send()
    
    if response.response:
        await cl.Message(content=response.response).send()
    else:
        await cl.Message(content="I've recorded your meal information.").send() 