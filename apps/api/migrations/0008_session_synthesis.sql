-- 0008_session_synthesis.sql
-- Add columns for deferred synthesis recovery.
-- accumulator_json stores the serialized SessionAccumulator when the DO
-- cannot synthesize (disconnect, alarm timeout).
-- needs_synthesis flags sessions that need deferred synthesis on next load.

ALTER TABLE sessions ADD COLUMN accumulator_json TEXT;
ALTER TABLE sessions ADD COLUMN needs_synthesis INTEGER DEFAULT 0;
