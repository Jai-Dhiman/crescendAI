-- Prevent duplicate auth_identities rows from TOCTOU race in find_or_create_student.
CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_identities_provider_user
ON auth_identities(provider, provider_user_id);
