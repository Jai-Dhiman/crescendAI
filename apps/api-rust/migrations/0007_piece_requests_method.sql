-- Add match_method column to piece_requests
-- Values: "text" (user named the piece), "ngram+dtw" (automatic MIDI matching), NULL (legacy)
ALTER TABLE piece_requests ADD COLUMN match_method TEXT;
