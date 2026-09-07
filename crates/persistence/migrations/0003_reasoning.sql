-- v3: per-session reasoning effort (off/low/medium/high/max), NULL = Default.
ALTER TABLE sessions ADD COLUMN reasoning_effort TEXT;
