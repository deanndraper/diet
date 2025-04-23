from langchain.tools import tool
import diet 
# Assume you have a way to get your database connection/session
# from .database import get_db_session # Example import

@tool
def update_daily_calorie_limit(user_id: str, new_calorie_limit: int) -> str:
    """
    Updates the daily calorie limit for a specific user.

    Args:
        user_id: The identifier of the user whose limit needs updating.
        new_calorie_limit: The new daily calorie limit (as an integer).

    Returns:
        A confirmation message string.
    """
    print(f"Attempting to update calorie limit for user '{user_id}' to {new_calorie_limit}...")
    try:
        # --- Database Interaction --- 
        conn = diet.db.get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET calorie_goal_per_day = ? WHERE id = ?",
                (new_calorie_limit, user_id)
            )
            # Check if any row was actually updated
            if cursor.rowcount > 0:
                print(f"Successfully updated calorie limit for user '{user_id}'.")
                return f"Successfully updated daily calorie limit for user {user_id} to {new_calorie_limit} calories."
            else:
                print(f"User '{user_id}' not found or limit already set to this value.")
                return f"Error: User with ID '{user_id}' not found, or their limit was already {new_calorie_limit}."
        # --- End Database Interaction ---

    except Exception as e:
        print(f"Error updating calorie limit for user '{user_id}': {e}")
        # Log the exception details for debugging
        import traceback
        traceback.print_exc()
        return f"An error occurred while trying to update the calorie limit: {e}"

# Example of how you might get the user_id - this depends heavily on your app structure
# You might get it from session state, authentication context, etc.
# This function is just illustrative and NOT part of the tool itself.
def get_current_user_id() -> str:
    # Replace with your actual logic to retrieve the current user's ID
    return "default_user" 