import gradio as gr
from diet import process_user_message, format_response, get_recent_chat, build_chat_history_block
import json

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
    return "", chat_history

with gr.Blocks(title="Diet Assistant 🥗", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Diet Assistant 🥗
    
    I can help you track your meals and calories. Here's what you can do:
    - Tell me what you ate (e.g., "I had 2 eggs and toast for breakfast")
    - Ask about your daily calorie intake
    - Get meal suggestions based on your dietary restrictions
    
    Try telling me what you ate today!
    """)
    
    chatbot = gr.Chatbot(
        value=get_chat_history(),
        height=500,
        show_copy_button=True,
        bubble_full_width=False
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="What did you eat today?",
            show_label=False,
            container=False,
            scale=9
        )
        submit = gr.Button("Send", variant="primary", scale=1)
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch() 