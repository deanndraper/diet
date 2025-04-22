from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv() # Ensure environment variables are loaded

class DietMemory:
    """Manages the persistence setup for chat history using SQLChatMessageHistory."""
    
    def __init__(self, user_id: str, application_id: str):
        """Initialize the memory system persistence.
        
        Args:
            user_id: The ID of the user
            application_id: The ID of the application
        """
        self.user_id = user_id
        self.application_id = application_id
        self.session_id = f"{user_id}_{application_id}"
        
    def get_message_history(self) -> SQLChatMessageHistory:
        """Get the configured SQLChatMessageHistory instance for the current session."""
        return SQLChatMessageHistory(
            session_id=self.session_id,
            connection="sqlite:///memory.db" 
        )

    def clear(self) -> None:
        """Clear the history for the current session."""
        # Get a history instance and clear it
        history = self.get_message_history()
        history.clear() 

if __name__ == "__main__":
    # 1. Setup persistence manager
    memory_manager = DietMemory("user123", "diet_app_v1")
    session_id = memory_manager.session_id

    # 2. Define the LLM and Prompt
    llm = ChatOpenAI(model="gpt-4o-mini") # Or your preferred model
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful diet assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Create the base chain
    chain = prompt | llm

    # 4. Create the runnable with history management
    #    The lambda function tells RunnableWithMessageHistory how to get 
    #    the history object for a given session_id.
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory_manager.get_message_history(), # Use the manager method
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # 5. Invoke the chain with history configuration
    print(f"--- Running chain for session: {session_id} ---")
    
    # First interaction
    config = {"configurable": {"session_id": session_id}}
    response = chain_with_history.invoke({"input": "Hi! I want to track my calories today."}, config=config)
    print(f"AI: {response.content}")

    # Second interaction (history is automatically loaded and saved)
    response = chain_with_history.invoke({"input": "I had a salad for lunch."}, config=config)
    print(f"AI: {response.content}")

    # Optional: Verify history (for demonstration)
    # history_instance = memory_manager.get_message_history()
    # print("\n--- Stored History ---")
    # print(history_instance.messages)

    # Optional: Clear history
    # memory_manager.clear()
    # print("\n--- History Cleared ---")
    # history_instance_after_clear = memory_manager.get_message_history()
    # print(history_instance_after_clear.messages)