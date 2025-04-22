import gradio as gr
from diet import FullResponse, process_image, get_meal_info, get_calorie_summary,process_user_message, format_response, LLMResponseText, FoodItem
import json
from datetime import datetime
from memory import DietMemory
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError

# Default user and application IDs
DEFAULT_USER_ID = "1"
DEFAULT_APPLICATION_ID = "1"

def get_chat_history():
    """Get the recent chat history, parse AI messages, and format for display."""
    try:
        memory_manager = DietMemory(DEFAULT_USER_ID, DEFAULT_APPLICATION_ID)
        history_messages = memory_manager.get_message_history().messages
        
        gradio_history = []
        for i in range(0, len(history_messages), 2): 
            user_msg_content = None
            ai_msg_content = None # Raw content from AIMessage
            formatted_ai_msg = None # Formatted string for display

            # Get User Message
            if i < len(history_messages) and isinstance(history_messages[i], HumanMessage):
                user_msg_content = history_messages[i].content
            
            # Get AI Message
            if (i + 1) < len(history_messages) and isinstance(history_messages[i+1], AIMessage):
                ai_msg_content = history_messages[i+1].content
            elif i < len(history_messages) and isinstance(history_messages[i], AIMessage):
                 # Handle case where history starts with AI message or is uneven
                 ai_msg_content = history_messages[i].content
                 # Parse and format this standalone AI message
                 try:
                    parsed_data = json.loads(ai_msg_content)
                    # Re-validate with Pydantic model before formatting
                    validated_response = LLMResponseText(**parsed_data) 
                    formatted_ai_msg = format_response(validated_response)
                 except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    print(f"Error parsing/formatting standalone AI message: {e}\nRaw: {ai_msg_content}")
                    formatted_ai_msg = ai_msg_content # Fallback to raw content
                 gradio_history.append((None, formatted_ai_msg))
                 continue # Skip pair processing
            
            # Parse and Format AI Message Content if available
            if ai_msg_content:
                try:
                    # The content should be the JSON string from the LLM
                    parsed_data = json.loads(ai_msg_content)
                    # Re-validate with Pydantic model before formatting
                    # This ensures we handle the structure format_response expects
                    validated_response = LLMResponseText(**parsed_data) 
                    formatted_ai_msg = format_response(validated_response)
                except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    print(f"Error parsing/formatting AI message content: {e}\nRaw: {ai_msg_content}")
                    formatted_ai_msg = ai_msg_content # Fallback to raw content
            
            # Add the pair (User message, Formatted AI message) to the history
            if user_msg_content is not None or formatted_ai_msg is not None:
                 gradio_history.append((user_msg_content, formatted_ai_msg))
                 
        return gradio_history
    except Exception as e:
        print(f"Error getting chat history: {e}") 
        return [] 

def respond(message, chat_history_display):
    """Process the user's message, update history, and return the response for display."""
    text = message.get("text", "")
    files = message.get("files", [])
    
    # 1. Handle Image Input (Optional)
    image_summary = ""
    if files:
        for file_path in files:
            # Display the uploaded image immediately in the chat
            chat_history_display.append((None, gr.Image(value=file_path))) 
            
            image_analysis: FullResponse = process_image(file_path)
            
            if image_analysis.error_message:
                 # Display error processing image
                 chat_history_display.append((None, f"Error processing image: {image_analysis.error_message}"))
            else:
                # Format the food items into a table for display
                formatted_pic_response = format_response(image_analysis) # Use existing formatter
                if formatted_pic_response:
                    chat_history_display.append((None, f"From the image:\n{formatted_pic_response}"))
                # Prepare image analysis summary for the text prompt
                if image_analysis.food_items:
                     image_summary += f"\n\n[Image Analysis: User provided an image showing: {json.dumps([item.model_dump(exclude={'meal_type'}) for item in image_analysis.food_items])}]"

    # Append image summary to text message if exists
    full_message_text = text + image_summary if image_summary else text

    # 2. Process Text Input (including image summary if applicable)
    assistant_response_content = "" # Default empty response
    if full_message_text.strip(): # Only process if there's text content
        # --- Call the updated process_user_message --- 
        # Note: process_user_message needs to be updated in diet.py 
        #       to use RunnableWithMessageHistory and return a formatted string or FullResponse
        # For now, assuming it returns a string response directly.
        try:
            # This function now implicitly uses the history via DietMemory/RunnableWithMessageHistory
            assistant_response_content = process_user_message(
                user_id=DEFAULT_USER_ID, 
                application_id=DEFAULT_APPLICATION_ID, 
                message=full_message_text
            ) 
        except Exception as e:
             print(f"Error calling process_user_message: {e}")
             assistant_response_content = "Sorry, I encountered an error processing your message." 
    
    # 3. Update & Return Display
    # Since process_user_message updated the persistent history,
    # fetch the latest history for display.
    updated_history_display = get_chat_history() 
    
    # If there was only an image and no text, the last AI message might be the image analysis.
    # If there was text processing, the last AI message should be assistant_response_content.
    # get_chat_history() *should* now contain the latest turn.

    dashboard_text = update_dashboard()
    # Return empty string for the input box, the updated history, and the dashboard
    return "", updated_history_display, dashboard_text

def update_dashboard():
    # Replace with actual logic
    time = datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    summary = get_calorie_summary(DEFAULT_USER_ID, DEFAULT_APPLICATION_ID)
    total_calories = summary['total_calories_today']
    remaining_calories = summary['remaining_calories'] 

    meals = get_meal_info(DEFAULT_USER_ID, DEFAULT_APPLICATION_ID)

    r =  f"""**{time}**
    **Total Calories:** {total_calories} kcal
    **Remaining Calories:** {remaining_calories} kcal\n\n"""
    r += "|Meal Type|Food Name|Calories|Quantity|Unit|\n|---|---|---|---|---|\n"
    for meal in meals:
        r += f"|{meal['meal_type']}|{meal['food_name']}|{meal['calories']}|{meal['quantity']}|{meal['unit']}|\n"
    return r

custom_css = """
footer {
  display: none !important;
}
a[href*="gradio.app"] {
  display: none !important;
}

#centered-markdownxxx {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}
#centered-columnx { 
  justify-content: center;
  align-items: center;
  height: 1000;
  vertical-align: middle;
}
#flex-column {
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 100%;
}
"""
with gr.Blocks(css=custom_css, title="Diet Assistant 🥗", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=2):  # Left: Chat
            gr.Markdown("## Diet Assistant 🥗")
            chatbot = gr.Chatbot(
                value=get_chat_history(),
                height=500,
                show_copy_button=False
            )
            with gr.Row():
                msg = gr.MultimodalTextbox(
                    placeholder="What did you eat today? Upload food pictures!",
                    show_label=False,
                    container=False,
                    scale=9,
                    file_types=["image"],
                    file_count="multiple"
                )
                submit = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Column(scale=2, elem_id = "flex-column"):  
            dashboard_text = gr.Markdown(update_dashboard(), label="Calorie Summary", elem_id = "centered-markdownx")

    msg.submit(respond, [msg, chatbot], [msg, chatbot, dashboard_text])
    submit.click(respond, [msg, chatbot], [msg, chatbot, dashboard_text]) 

if __name__ == "__main__":
    demo.launch(show_api=False) 