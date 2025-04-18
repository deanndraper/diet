import gradio as gr
from diet import get_meal_info, get_calorie_summary, process_user_message, format_response, get_recent_chat, build_chat_history_block
import json
from datetime import datetime
# Default user and application IDs
DEFAULT_USER_ID = "1"
DEFAULT_APPLICATION_ID = "1"

def get_chat_history():
    """Get the recent chat history and format it for display"""
    recent = get_recent_chat(DEFAULT_USER_ID, DEFAULT_APPLICATION_ID)
    chat_history_json = build_chat_history_block(recent)
    chat_history = json.loads(chat_history_json)
    return [(msg["message"], None) if msg["role"] == "user" else (None, msg["message"]) for msg in chat_history]

def respond(message, chat_history):
    """Process the user's message and return the response"""
    response = process_user_message(message, DEFAULT_USER_ID, DEFAULT_APPLICATION_ID)
    formatted_response = format_response(response)
    chat_history.append((message, formatted_response))
    dashboard_text = update_dashboard()
    return "", chat_history, dashboard_text

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
                show_copy_button=True
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What did you eat today?",
                    show_label=False,
                    container=False,
                    scale=9
                )
                submit = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Column(scale=2, elem_id = "flex-column"):  
            dashboard_text = gr.Markdown(update_dashboard(), label="Calorie Summary", elem_id = "centered-markdownx")

    msg.submit(respond, [msg, chatbot], [msg, chatbot, dashboard_text])
    submit.click(respond, [msg, chatbot], [msg, chatbot, dashboard_text]) 

if __name__ == "__main__":
    demo.launch(show_api=False) 