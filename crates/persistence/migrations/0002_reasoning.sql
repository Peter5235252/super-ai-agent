-- v2: keep the model's private reasoning next to its message so the UI
-- can show a "Thinking" section for past turns. Never sent back to providers.
ALTER TABLE messages ADD COLUMN reasoning TEXT;
