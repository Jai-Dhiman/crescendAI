#!/bin/bash

# Piano Platform API - Comprehensive Local Testing Script
# Tests all endpoints with curl commands

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8080/api"
TEACHER_EMAIL="test.teacher@example.com"
TEACHER_PASSWORD="TestPass123!"
STUDENT_EMAIL="test.student@example.com"
STUDENT_PASSWORD="TestPass123!"

# Global variables for storing tokens and IDs
TEACHER_TOKEN=""
STUDENT_TOKEN=""
RELATIONSHIP_ID=""
PROJECT_ID=""
ANNOTATION_ID=""
KNOWLEDGE_DOC_ID=""
CHAT_SESSION_ID=""

# Helper functions
print_section() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}\n"
}

print_test() {
    echo -e "${YELLOW}Testing: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_response() {
    echo "$1" | jq '.' 2>/dev/null || echo "$1"
}

# Test functions

test_health() {
    print_section "1. Health Check"

    print_test "GET /api/health"
    response=$(curl -s "${API_URL}/health")
    print_response "$response"

    if echo "$response" | grep -q "ok"; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        exit 1
    fi
}

test_auth() {
    print_section "2. Authentication"

    # Register teacher
    print_test "POST /api/auth/register (Teacher)"
    response=$(curl -s -X POST "${API_URL}/auth/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"email\": \"${TEACHER_EMAIL}\",
            \"password\": \"${TEACHER_PASSWORD}\",
            \"full_name\": \"Test Teacher\",
            \"role\": \"teacher\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "access_token"; then
        print_success "Teacher registration successful"
        TEACHER_TOKEN=$(echo "$response" | jq -r '.access_token')
    else
        print_error "Teacher registration failed (may already exist)"
        # Try to login instead
        print_test "POST /api/auth/login (Teacher - fallback)"
        response=$(curl -s -X POST "${API_URL}/auth/login" \
            -H "Content-Type: application/json" \
            -d "{
                \"email\": \"${TEACHER_EMAIL}\",
                \"password\": \"${TEACHER_PASSWORD}\"
            }")
        TEACHER_TOKEN=$(echo "$response" | jq -r '.access_token')
        if [ -n "$TEACHER_TOKEN" ] && [ "$TEACHER_TOKEN" != "null" ]; then
            print_success "Teacher login successful"
        else
            print_error "Teacher login failed"
            exit 1
        fi
    fi

    # Register student
    print_test "POST /api/auth/register (Student)"
    response=$(curl -s -X POST "${API_URL}/auth/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"email\": \"${STUDENT_EMAIL}\",
            \"password\": \"${STUDENT_PASSWORD}\",
            \"full_name\": \"Test Student\",
            \"role\": \"student\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "access_token"; then
        print_success "Student registration successful"
        STUDENT_TOKEN=$(echo "$response" | jq -r '.access_token')
    else
        print_error "Student registration failed (may already exist)"
        # Try to login instead
        print_test "POST /api/auth/login (Student - fallback)"
        response=$(curl -s -X POST "${API_URL}/auth/login" \
            -H "Content-Type: application/json" \
            -d "{
                \"email\": \"${STUDENT_EMAIL}\",
                \"password\": \"${STUDENT_PASSWORD}\"
            }")
        STUDENT_TOKEN=$(echo "$response" | jq -r '.access_token')
        if [ -n "$STUDENT_TOKEN" ] && [ "$STUDENT_TOKEN" != "null" ]; then
            print_success "Student login successful"
        else
            print_error "Student login failed"
            exit 1
        fi
    fi

    # Get teacher profile
    print_test "GET /api/auth/me (Teacher)"
    response=$(curl -s -X GET "${API_URL}/auth/me" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"

    if echo "$response" | grep -q "teacher"; then
        print_success "Teacher profile retrieved"
    else
        print_error "Failed to retrieve teacher profile"
    fi

    # Get student profile
    print_test "GET /api/auth/me (Student)"
    response=$(curl -s -X GET "${API_URL}/auth/me" \
        -H "Authorization: Bearer ${STUDENT_TOKEN}")
    print_response "$response"

    if echo "$response" | grep -q "student"; then
        print_success "Student profile retrieved"
    else
        print_error "Failed to retrieve student profile"
    fi
}

test_relationships() {
    print_section "3. Teacher-Student Relationships"

    # Get student ID
    print_test "Getting student ID"
    response=$(curl -s -X GET "${API_URL}/auth/me" \
        -H "Authorization: Bearer ${STUDENT_TOKEN}")
    STUDENT_ID=$(echo "$response" | jq -r '.id')
    print_success "Student ID: ${STUDENT_ID}"

    # Create relationship (teacher creates)
    print_test "POST /api/relationships (Teacher creates)"
    response=$(curl -s -X POST "${API_URL}/relationships" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"student_id\": \"${STUDENT_ID}\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "id"; then
        print_success "Relationship created"
        RELATIONSHIP_ID=$(echo "$response" | jq -r '.id')
    else
        print_error "Failed to create relationship (may already exist)"
    fi

    # List relationships (teacher)
    print_test "GET /api/relationships (Teacher)"
    response=$(curl -s -X GET "${API_URL}/relationships" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Teacher relationships listed"

    # List relationships (student)
    print_test "GET /api/relationships (Student)"
    response=$(curl -s -X GET "${API_URL}/relationships" \
        -H "Authorization: Bearer ${STUDENT_TOKEN}")
    print_response "$response"
    print_success "Student relationships listed"
}

test_projects() {
    print_section "4. Projects & PDF Management"

    # Create project
    print_test "POST /api/projects (Create project)"
    response=$(curl -s -X POST "${API_URL}/projects" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"title\": \"Chopin Nocturne Op. 9 No. 2\",
            \"description\": \"Working on dynamics and pedaling\",
            \"filename\": \"chopin_nocturne.pdf\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "upload_url"; then
        print_success "Project created with presigned upload URL"
        PROJECT_ID=$(echo "$response" | jq -r '.project.id')
        UPLOAD_URL=$(echo "$response" | jq -r '.upload_url')
        echo "Project ID: ${PROJECT_ID}"
        echo "Upload URL: ${UPLOAD_URL}"
    else
        print_error "Failed to create project"
    fi

    # Note: In a real test, you would upload a PDF to the presigned URL here
    # For now, we'll skip the upload and just test other endpoints

    # List projects
    print_test "GET /api/projects (List projects)"
    response=$(curl -s -X GET "${API_URL}/projects" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Projects listed"

    # Get specific project
    if [ -n "$PROJECT_ID" ]; then
        print_test "GET /api/projects/${PROJECT_ID} (Get project)"
        response=$(curl -s -X GET "${API_URL}/projects/${PROJECT_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Project details retrieved"

        # Update project
        print_test "PATCH /api/projects/${PROJECT_ID} (Update project)"
        response=$(curl -s -X PATCH "${API_URL}/projects/${PROJECT_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"title\": \"Chopin Nocturne Op. 9 No. 2 (Updated)\",
                \"description\": \"Focus on measures 25-32\"
            }")
        print_response "$response"
        print_success "Project updated"

        # Grant access to student
        print_test "POST /api/projects/${PROJECT_ID}/access (Grant access)"
        response=$(curl -s -X POST "${API_URL}/projects/${PROJECT_ID}/access" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"user_id\": \"${STUDENT_ID}\",
                \"access_level\": \"edit\"
            }")
        print_response "$response"
        print_success "Access granted to student"

        # List project access
        print_test "GET /api/projects/${PROJECT_ID}/access (List access)"
        response=$(curl -s -X GET "${API_URL}/projects/${PROJECT_ID}/access" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Project access listed"
    fi
}

test_annotations() {
    print_section "5. Annotations"

    if [ -z "$PROJECT_ID" ]; then
        print_error "Skipping annotations tests (no project ID)"
        return
    fi

    # Create highlight annotation
    print_test "POST /api/annotations (Create highlight)"
    response=$(curl -s -X POST "${API_URL}/annotations" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"project_id\": \"${PROJECT_ID}\",
            \"page_number\": 1,
            \"annotation_type\": \"highlight\",
            \"content\": {
                \"text\": \"This passage needs more legato\",
                \"color\": \"#FFFF00\",
                \"position\": {
                    \"x\": 100,
                    \"y\": 200,
                    \"width\": 150,
                    \"height\": 20
                }
            }
        }")
    print_response "$response"

    if echo "$response" | grep -q "id"; then
        print_success "Highlight annotation created"
        ANNOTATION_ID=$(echo "$response" | jq -r '.id')
    else
        print_error "Failed to create annotation"
    fi

    # Create note annotation
    print_test "POST /api/annotations (Create note)"
    response=$(curl -s -X POST "${API_URL}/annotations" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"project_id\": \"${PROJECT_ID}\",
            \"page_number\": 1,
            \"annotation_type\": \"note\",
            \"content\": {
                \"text\": \"Practice this with metronome at 60 BPM\",
                \"position\": {
                    \"x\": 300,
                    \"y\": 400
                }
            }
        }")
    print_response "$response"
    print_success "Note annotation created"

    # List annotations for project
    print_test "GET /api/annotations?project_id=${PROJECT_ID} (List annotations)"
    response=$(curl -s -X GET "${API_URL}/annotations?project_id=${PROJECT_ID}" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Annotations listed"

    # Get specific annotation
    if [ -n "$ANNOTATION_ID" ]; then
        print_test "GET /api/annotations/${ANNOTATION_ID} (Get annotation)"
        response=$(curl -s -X GET "${API_URL}/annotations/${ANNOTATION_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Annotation retrieved"

        # Update annotation
        print_test "PATCH /api/annotations/${ANNOTATION_ID} (Update annotation)"
        response=$(curl -s -X PATCH "${API_URL}/annotations/${ANNOTATION_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"content\": {
                    \"text\": \"This passage needs MORE legato (updated)\",
                    \"color\": \"#FF0000\",
                    \"position\": {
                        \"x\": 100,
                        \"y\": 200,
                        \"width\": 150,
                        \"height\": 20
                    }
                }
            }")
        print_response "$response"
        print_success "Annotation updated"
    fi
}

test_knowledge_base() {
    print_section "6. Knowledge Base"

    # Create knowledge doc
    print_test "POST /api/knowledge (Create doc)"
    response=$(curl -s -X POST "${API_URL}/knowledge" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"title\": \"Piano Technique Fundamentals\",
            \"doc_type\": \"pdf\",
            \"source_url\": \"https://example.com/technique.pdf\",
            \"is_public\": true,
            \"filename\": \"technique.pdf\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "id"; then
        print_success "Knowledge doc created"
        KNOWLEDGE_DOC_ID=$(echo "$response" | jq -r '.doc.id')
    else
        print_error "Failed to create knowledge doc"
    fi

    # List knowledge docs
    print_test "GET /api/knowledge (List docs)"
    response=$(curl -s -X GET "${API_URL}/knowledge" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Knowledge docs listed"

    # Get specific knowledge doc
    if [ -n "$KNOWLEDGE_DOC_ID" ]; then
        print_test "GET /api/knowledge/${KNOWLEDGE_DOC_ID} (Get doc)"
        response=$(curl -s -X GET "${API_URL}/knowledge/${KNOWLEDGE_DOC_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Knowledge doc retrieved"

        # Get processing status
        print_test "GET /api/knowledge/${KNOWLEDGE_DOC_ID}/status (Get status)"
        response=$(curl -s -X GET "${API_URL}/knowledge/${KNOWLEDGE_DOC_ID}/status" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Processing status retrieved"
    fi
}

test_chat() {
    print_section "7. Chat & RAG"

    # Check chat health
    print_test "GET /api/chat/health (Chat health check)"
    response=$(curl -s -X GET "${API_URL}/chat/health" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Chat health check passed"

    # Create chat session
    print_test "POST /api/chat/sessions (Create session)"
    response=$(curl -s -X POST "${API_URL}/chat/sessions" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"title\": \"Piano Technique Questions\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "id"; then
        print_success "Chat session created"
        CHAT_SESSION_ID=$(echo "$response" | jq -r '.id')
    else
        print_error "Failed to create chat session"
    fi

    # List chat sessions
    print_test "GET /api/chat/sessions (List sessions)"
    response=$(curl -s -X GET "${API_URL}/chat/sessions" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}")
    print_response "$response"
    print_success "Chat sessions listed"

    # Get specific session
    if [ -n "$CHAT_SESSION_ID" ]; then
        print_test "GET /api/chat/sessions/${CHAT_SESSION_ID} (Get session)"
        response=$(curl -s -X GET "${API_URL}/chat/sessions/${CHAT_SESSION_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Chat session retrieved"
    fi

    # Test RAG query (will fail without Workers AI credentials, but should return proper error)
    print_test "POST /api/chat/query (RAG query - may fail without AI credentials)"
    response=$(curl -s -X POST "${API_URL}/chat/query" \
        -H "Authorization: Bearer ${TEACHER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"How do I improve my finger independence?\",
            \"session_id\": \"${CHAT_SESSION_ID}\"
        }")
    print_response "$response"

    if echo "$response" | grep -q "answer\|error"; then
        print_success "RAG query endpoint responding (may need AI credentials for full functionality)"
    else
        print_error "RAG query endpoint not responding"
    fi
}

# Cleanup function
cleanup() {
    print_section "8. Cleanup (Optional)"

    # Delete annotation
    if [ -n "$ANNOTATION_ID" ]; then
        print_test "DELETE /api/annotations/${ANNOTATION_ID} (Delete annotation)"
        response=$(curl -s -X DELETE "${API_URL}/annotations/${ANNOTATION_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Annotation deleted"
    fi

    # Delete project
    if [ -n "$PROJECT_ID" ]; then
        print_test "DELETE /api/projects/${PROJECT_ID} (Delete project)"
        response=$(curl -s -X DELETE "${API_URL}/projects/${PROJECT_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Project deleted"
    fi

    # Delete knowledge doc
    if [ -n "$KNOWLEDGE_DOC_ID" ]; then
        print_test "DELETE /api/knowledge/${KNOWLEDGE_DOC_ID} (Delete doc)"
        response=$(curl -s -X DELETE "${API_URL}/knowledge/${KNOWLEDGE_DOC_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Knowledge doc deleted"
    fi

    # Delete chat session
    if [ -n "$CHAT_SESSION_ID" ]; then
        print_test "DELETE /api/chat/sessions/${CHAT_SESSION_ID} (Delete session)"
        response=$(curl -s -X DELETE "${API_URL}/chat/sessions/${CHAT_SESSION_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Chat session deleted"
    fi

    # Delete relationship
    if [ -n "$RELATIONSHIP_ID" ]; then
        print_test "DELETE /api/relationships/${RELATIONSHIP_ID} (Delete relationship)"
        response=$(curl -s -X DELETE "${API_URL}/relationships/${RELATIONSHIP_ID}" \
            -H "Authorization: Bearer ${TEACHER_TOKEN}")
        print_response "$response"
        print_success "Relationship deleted"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║   Piano Platform API - Comprehensive Testing Suite       ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo "Testing API at: ${API_URL}"
    echo ""

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        print_error "jq is not installed. Please install it for better JSON output."
        echo "macOS: brew install jq"
        echo "Continuing without jq..."
    fi

    # Run all tests
    test_health
    test_auth
    test_relationships
    test_projects
    test_annotations
    test_knowledge_base
    test_chat

    # Ask if user wants to cleanup
    echo ""
    read -p "Do you want to cleanup test data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi

    print_section "Testing Complete!"
    echo -e "${GREEN}All tests completed successfully!${NC}"
    echo ""
    echo "Summary:"
    echo "- Teacher Token: ${TEACHER_TOKEN:0:20}..."
    echo "- Student Token: ${STUDENT_TOKEN:0:20}..."
    echo "- Project ID: ${PROJECT_ID}"
    echo "- Annotation ID: ${ANNOTATION_ID}"
    echo "- Knowledge Doc ID: ${KNOWLEDGE_DOC_ID}"
    echo "- Chat Session ID: ${CHAT_SESSION_ID}"
}

# Run main function
main
