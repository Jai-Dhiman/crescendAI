<script lang="ts">
import type { Message } from '$lib/types/chat';

interface Props {
	message: Message;
	isStreaming?: boolean;
}

let { message, isStreaming = false }: Props = $props();

function formatTime(timestamp: string): string {
	const date = new Date(timestamp);
	return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}
</script>

<div class="message {message.role === 'user' ? 'user-message' : 'assistant-message'}">
	<div class="message-content">
		<div class="message-text">
			{message.content}
			{#if isStreaming}
				<span class="cursor">|</span>
			{/if}
		</div>
		{#if message.tool_calls && message.tool_calls.length > 0}
			<div class="tool-calls">
				{#each message.tool_calls as toolCall}
					<div class="tool-call">
						<span class="tool-icon">🔍</span>
						Searching: {toolCall.tool}
					</div>
				{/each}
			</div>
		{/if}
	</div>
	{#if message.created_at}
		<div class="message-time">{formatTime(message.created_at)}</div>
	{/if}
</div>

<style>
.message {
	margin-bottom: 1rem;
	display: flex;
	flex-direction: column;
	max-width: 85%;
}

.user-message {
	align-self: flex-end;
	align-items: flex-end;
}

.assistant-message {
	align-self: flex-start;
	align-items: flex-start;
}

.message-content {
	background: var(--color-primary, #1a1a2e);
	color: white;
	padding: 0.75rem 1rem;
	border-radius: 1rem;
	box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.user-message .message-content {
	background: #c4a777;
	color: #1a1a2e;
	border-bottom-right-radius: 0.25rem;
}

.assistant-message .message-content {
	background: #2c2c3e;
	color: white;
	border-bottom-left-radius: 0.25rem;
}

.message-text {
	line-height: 1.5;
	white-space: pre-wrap;
	word-wrap: break-word;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 0.95rem;
}

.cursor {
	display: inline-block;
	animation: blink 1s infinite;
	margin-left: 2px;
}

@keyframes blink {
	0%, 50% { opacity: 1; }
	51%, 100% { opacity: 0; }
}

.message-time {
	font-size: 0.75rem;
	color: #666;
	margin-top: 0.25rem;
	padding: 0 0.5rem;
	font-family: var(--font-sans, sans-serif);
}

.tool-calls {
	margin-top: 0.5rem;
	padding-top: 0.5rem;
	border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.tool-call {
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.85rem;
	color: rgba(255, 255, 255, 0.7);
	font-style: italic;
}

.tool-icon {
	font-size: 1rem;
}
</style>
