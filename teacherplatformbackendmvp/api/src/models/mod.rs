pub mod access;
pub mod annotation;
pub mod chat;
pub mod knowledge;
pub mod project;
pub mod user;

pub use access::AccessLevel;
pub use annotation::{
    Annotation, AnnotationType, CreateAnnotationRequest, DrawingContent, HighlightContent,
    NoteContent, Point, UpdateAnnotationRequest,
};
pub use chat::{
    ChatMessage, ChatSession, CreateSessionRequest, SessionListResponse, SessionMessagesResponse,
    SessionResponse,
};
pub use knowledge::{
    ChunkMetadata, CreateKnowledgeRequest, CreateKnowledgeResponse, DocumentChunk, KnowledgeDoc,
    ProcessingStatus, ProcessingStatusResponse, SearchResult,
};
pub use project::{
    AccessResponse, CreateProjectRequest, CreateProjectResponse,
    GrantAccessRequest, PdfMetadata, Project, ProjectAccess, ProjectAccessListResponse,
    ProjectAccessWithUser, ProjectWithAccess, UpdateProjectRequest,
};
pub use user::{CreateUserRequest, User, UserResponse, UserRole};
