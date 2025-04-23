import sqlite3
import os

def run_migrations():
    """Run all SQL migrations in the migrations directory."""
    # Get the directory of this script
    migrations_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Connect to the database
    conn = sqlite3.connect('diet.db')
    cursor = conn.cursor()
    
    try:
        # Read and execute each SQL file
        for filename in os.listdir(migrations_dir):
            if filename.endswith('.sql'):
                with open(os.path.join(migrations_dir, filename), 'r') as f:
                    sql = f.read()
                    cursor.executescript(sql)
        
        # Commit the changes
        conn.commit()
        print("Migrations completed successfully!")
        
    except Exception as e:
        print(f"Error running migrations: {str(e)}")
        conn.rollback()
        
    finally:
        conn.close()

if __name__ == "__main__":
    run_migrations() 