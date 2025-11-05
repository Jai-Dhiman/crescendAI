<script lang="ts">
import { onMount } from 'svelte';
import { crescendApi } from '$lib/services/crescendApi';
import type { Message } from '$lib/types/chat';
import ChatMessage from './ChatMessage.svelte';
import ChatInput from './ChatInput.svelte';
import { MessageCircle } from 'lucide-svelte';

let messages = $state<Message[]>([]);
let sessionId = $state<string | null>(null);
let isLoading = $state(false);
let error = $state<string | null>(null);
let streamingContent = $state('');
let isStreaming = $state(false);

onMount(async () => {
	try {
		// Create a new chat session for beginners
		const session = await crescendApi.createChatSession('Beginner Questions');
		sessionId = session.id;
	} catch (err) {
		console.error('Failed to create chat session:', err);
		error = 'Failed to start chat. Please refresh the page.';
	}
});

async function handleSendMessage(content: string) {
	if (!sessionId || isLoading) return;

	// Add user message immediately
	const userMessage: Message = {
		id: `temp-${Date.now()}`,
		role: 'user',
		content,
		created_at: new Date().toISOString(),
	};
	messages = [...messages, userMessage];

	isLoading = true;
	isStreaming = true;
	streamingContent = '';
	error = null;

	try {
		const assistantMessage = await crescendApi.sendChatMessage(
			sessionId,
			content,
			(token) => {
				// Update streaming content as tokens arrive
				streamingContent += token;
			}
		);

		// Replace streaming content with final message
		messages = [...messages, assistantMessage];
		streamingContent = '';
		isStreaming = false;
	} catch (err) {
		console.error('Failed to send message:', err);
		error = 'Failed to send message. Please try again.';
		// Remove the user message on error
		messages = messages.filter(m => m.id !== userMessage.id);
	} finally {
		isLoading = false;
		isStreaming = false;
		streamingContent = '';
	}
}

// Scroll to bottom when new messages arrive
$effect(() => {
	if (messages.length > 0 || streamingContent) {
		const messagesContainer = document.querySelector('.chat-messages');
		if (messagesContainer) {
			messagesContainer.scrollTop = messagesContainer.scrollHeight;
		}
	}
});
</script>

<div class="beginner-chat-container">
	<div class="chat-header">
		<MessageCircle size={24} />
		<div class="header-text">
			<h3>Ask a Question</h3>
			<p>Learn about piano technique, practice methods, and more</p>
		</div>
	</div>

	<div class="chat-messages">
		{#if messages.length === 0 && !streamingContent}
			<div class="empty-state">
				<MessageCircle size={48} class="empty-icon" />
				<h4>Welcome to CrescendAI</h4>
				<p>Ask me anything about piano practice, technique, or music theory!</p>
				<div class="starter-questions">
					<button
						class="starter-button"
						onclick={() => handleSendMessage('How should I practice scales effectively?')}
						disabled={!sessionId || isLoading}
					>
						How should I practice scales?
					</button>
					<button
						class="starter-button"
						onclick={() => handleSendMessage('What are the best warm-up exercises for beginners?')}
						disabled={!sessionId || isLoading}
					>
						Best warm-up exercises?
					</button>
					<button
						class="starter-button"
						onclick={() => handleSendMessage('How can I improve my sight-reading?')}
						disabled={!sessionId || isLoading}
					>
						Improve sight-reading?
					</button>
				</div>
			</div>
		{:else}
			{#each messages as message (message.id)}
				<ChatMessage {message} />
			{/each}
			{#if isStreaming && streamingContent}
				<ChatMessage
					message={{
						id: 'streaming',
						role: 'assistant',
						content: streamingContent,
						created_at: new Date().toISOString(),
					}}
					isStreaming={true}
				/>
			{/if}
		{/if}

		{#if error}
			<div class="error-message">
				{error}
			</div>
		{/if}
	</div>

	<ChatInput
		onSend={handleSendMessage}
		disabled={!sessionId || isLoading}
		placeholder={isLoading ? 'Waiting for response...' : 'Ask a question about piano...'}
	/>
</div>

<style>
.beginner-chat-container {
	display: flex;
	flex-direction: column;
	background: white;
	border-radius: 1rem;
	box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
	overflow: hidden;
	max-width: 900px;
	margin: 0 auto;
	height: 600px;
}

.chat-header {
	display: flex;
	align-items: center;
	gap: 1rem;
	padding: 1.25rem 1.5rem;
	background: linear-gradient(135deg, #1a1a2e 0%, #2c2c3e 100%);
	color: white;
	border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header-text h3 {
	margin: 0;
	font-family: var(--font-sans, sans-serif);
	font-size: 1.25rem;
	font-weight: 600;
}

.header-text p {
	margin: 0.25rem 0 0 0;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 0.875rem;
	opacity: 0.8;
}

.chat-messages {
	flex: 1;
	overflow-y: auto;
	padding: 1.5rem;
	display: flex;
	flex-direction: column;
	background: #fafafa;
}

.empty-state {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	text-align: center;
	height: 100%;
	padding: 2rem;
	color: #666;
}

.empty-state :global(.empty-icon) {
	opacity: 0.3;
	margin-bottom: 1rem;
}

.empty-state h4 {
	margin: 0 0 0.5rem 0;
	font-family: var(--font-sans, sans-serif);
	font-size: 1.5rem;
	color: #1a1a2e;
}

.empty-state p {
	margin: 0 0 2rem 0;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 1rem;
	color: #666;
}

.starter-questions {
	display: flex;
	flex-direction: column;
	gap: 0.75rem;
	width: 100%;
	max-width: 400px;
}

.starter-button {
	padding: 0.875rem 1.25rem;
	background: white;
	border: 2px solid #e5e5e5;
	border-radius: 0.75rem;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 0.95rem;
	color: #1a1a2e;
	cursor: pointer;
	transition: all 0.2s ease;
	text-align: left;
}

.starter-button:hover:not(:disabled) {
	border-color: #c4a777;
	background: #fffbf5;
	transform: translateY(-2px);
	box-shadow: 0 4px 12px rgba(196, 167, 119, 0.2);
}

.starter-button:disabled {
	opacity: 0.5;
	cursor: not-allowed;
}

.error-message {
	padding: 1rem;
	background: #fee2e2;
	border: 1px solid #fecaca;
	border-radius: 0.5rem;
	color: #dc3545;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 0.9rem;
	margin-top: 1rem;
}

@media (max-width: 768px) {
	.beginner-chat-container {
		height: 500px;
		border-radius: 0.75rem;
	}

	.chat-header {
		padding: 1rem;
	}

	.header-text h3 {
		font-size: 1.1rem;
	}

	.header-text p {
		font-size: 0.8rem;
	}

	.chat-messages {
		padding: 1rem;
	}

	.starter-questions {
		max-width: 100%;
	}
}
</style>
