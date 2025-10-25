-- Initialize Piano Platform Database
-- This script runs on container startup

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types for user roles
CREATE TYPE user_role AS ENUM ('teacher', 'student', 'admin');

-- Create enum types for project access levels
CREATE TYPE access_level AS ENUM ('view', 'edit', 'admin');

-- Create enum types for annotation types
CREATE TYPE annotation_type AS ENUM ('highlight', 'note', 'drawing');

-- Set up basic configuration for pgvector performance
-- These will be optimized later based on load testing
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET work_mem = '32MB';

-- Note: PostgreSQL config changes require restart, which will happen automatically
-- in the Docker container on first boot

-- Success message
SELECT 'Piano Platform Database initialized successfully' AS status;
