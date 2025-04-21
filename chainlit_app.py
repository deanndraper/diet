import chainlit as cl
import json
from diet import process_user_message, FullResponse, format_response, get_recent_chat, build_chat_history_block

@cl.on_chat_start
async def start():
    cl.user_session.set("user_id", "1")  # Default user ID
    cl.user_session.set("application_id", "1")  # Default application ID
    
    # Send a welcome message
    await cl.Message(
        content="""Welcome to the Diet Assistant! 🥗

I can help you track your meals and calories. Here's what you can do:
- Tell me what you ate (e.g., "I had 2 eggs and toast for breakfast")
- Ask about your daily calorie intake
- Get meal suggestions based on your dietary restrictions

Try telling me what you ate today!""",
    ).send()
    recent = get_recent_chat(1,1)
    chat_history_json = build_chat_history_block(recent)
    chat_history = json.loads(chat_history_json)
    for msg in chat_history: 
        if msg["role"] == "user":
            await cl.Message(content=msg["message"]).send()  
        else:
            await cl.Message(content=msg["message"]).send()  
    

@cl.on_message
async def main(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    application_id = cl.user_session.get("application_id")
    
    # Process the message using the existing diet application logic
    response = process_user_message(message.content, user_id, application_id)
    
    # Handle the response
    if response.error_message:
        # This is an error message
        await cl.Message(content=response.error_message).send()
    else:
        # This is a FullResponse object
        formatted_response = format_response(response)
        await cl.Message(content=formatted_response).send()