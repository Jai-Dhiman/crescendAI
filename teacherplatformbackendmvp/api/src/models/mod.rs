pub mod access;
pub mod user;
pub mod knowledge;
pub mod chat;

pub use access::AccessLevel;
pub use user::{User, UserRole, CreateUserRequest, UserResponse};
pub use knowledge::{
    KnowledgeDoc, DocumentChunk, ProcessingStatus,
    CreateKnowledgeRequest, CreateKnowledgeResponse,
    ProcessingStatusResponse, SearchResult, ChunkMetadata,
};
pub use chat::{
    ChatSession, ChatMessage, CreateSessionRequest,
    SessionResponse, SessionListResponse, SessionMessagesResponse,
};
