-- Create the message_store table for LangChain chat history
CREATE TABLE IF NOT EXISTS message_store (
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    role TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (session_id, timestamp)
);

-- Create an index for faster lookups by session_id
CREATE INDEX IF NOT EXISTS idx_message_store_session_id ON message_store(session_id); 