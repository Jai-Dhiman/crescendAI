// Chat-related types for the CrescendAI API

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, any>;
  result?: any;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatSessionWithMessages extends ChatSession {
  messages: Message[];
}

export interface CreateChatSessionRequest {
  title: string;
}

export interface SendMessageRequest {
  session_id: string;
  content: string;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
  total_pages: number;
}

export interface ListChatSessionsResponse {
  sessions: ChatSession[];
  pagination: Pagination;
}
