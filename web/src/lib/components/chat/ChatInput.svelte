<script lang="ts">
import { Send } from 'lucide-svelte';

interface Props {
	onSend: (message: string) => void;
	disabled?: boolean;
	placeholder?: string;
}

let { onSend, disabled = false, placeholder = 'Ask a question about piano technique...' }: Props = $props();

let message = $state('');
let textarea: HTMLTextAreaElement | undefined = $state();

function handleSend() {
	const trimmed = message.trim();
	if (trimmed && !disabled) {
		onSend(trimmed);
		message = '';
		if (textarea) {
			textarea.style.height = 'auto';
		}
	}
}

function handleKeydown(e: KeyboardEvent) {
	if (e.key === 'Enter' && !e.shiftKey) {
		e.preventDefault();
		handleSend();
	}
}

function autoResize() {
	if (textarea) {
		textarea.style.height = 'auto';
		textarea.style.height = textarea.scrollHeight + 'px';
	}
}
</script>

<div class="chat-input-wrapper">
	<textarea
		bind:this={textarea}
		bind:value={message}
		onkeydown={handleKeydown}
		oninput={autoResize}
		{placeholder}
		{disabled}
		class="chat-input"
		rows="1"
	></textarea>
	<button
		onclick={handleSend}
		disabled={disabled || !message.trim()}
		class="send-button"
		aria-label="Send message"
	>
		<Send size={20} />
	</button>
</div>

<style>
.chat-input-wrapper {
	display: flex;
	align-items: flex-end;
	gap: 0.75rem;
	padding: 1rem;
	background: white;
	border-top: 1px solid #e5e5e5;
	border-radius: 0 0 1rem 1rem;
}

.chat-input {
	flex: 1;
	resize: none;
	border: 2px solid #e5e5e5;
	border-radius: 1rem;
	padding: 0.75rem 1rem;
	font-family: var(--font-serif, 'Georgia', serif);
	font-size: 0.95rem;
	line-height: 1.5;
	max-height: 150px;
	overflow-y: auto;
	transition: border-color 0.2s ease;
}

.chat-input:focus {
	outline: none;
	border-color: #c4a777;
}

.chat-input:disabled {
	background: #f5f5f5;
	cursor: not-allowed;
}

.send-button {
	flex-shrink: 0;
	width: 44px;
	height: 44px;
	display: flex;
	align-items: center;
	justify-content: center;
	background: #c4a777;
	color: white;
	border: none;
	border-radius: 50%;
	cursor: pointer;
	transition: all 0.2s ease;
	box-shadow: 0 2px 8px rgba(196, 167, 119, 0.3);
}

.send-button:hover:not(:disabled) {
	background: #b89968;
	transform: scale(1.05);
	box-shadow: 0 4px 12px rgba(196, 167, 119, 0.4);
}

.send-button:disabled {
	background: #d5d5d5;
	cursor: not-allowed;
	box-shadow: none;
}

.send-button:active:not(:disabled) {
	transform: scale(0.95);
}
</style>
