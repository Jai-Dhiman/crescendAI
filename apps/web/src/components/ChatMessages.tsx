import { useThrottledCallback } from "@tanstack/react-pacer";
import { useEffect, useRef } from "react";
import type { RichMessage } from "../lib/types";
import { InlineCard } from "./InlineCard";
import { MessageContent } from "./MessageContent";

interface ChatMessagesProps {
	messages: RichMessage[];
	streamingContent: string | null;
}

export function ChatMessages({
	messages,
	streamingContent,
}: ChatMessagesProps) {
	const bottomRef = useRef<HTMLDivElement>(null);
	const contentLengthRef = useRef(0);

	const scrollToBottom = useThrottledCallback(
		() => {
			bottomRef.current?.scrollIntoView({ behavior: "smooth" });
		},
		{ wait: 16 },
	);

	// Scroll when content changes
	const currentLength = messages.length + (streamingContent?.length ?? 0);
	if (currentLength !== contentLengthRef.current) {
		contentLengthRef.current = currentLength;
		scrollToBottom();
	}

	// Also scroll on mount
	useEffect(() => {
		scrollToBottom();
	}, [scrollToBottom]);

	if (messages.length === 0 && !streamingContent) {
		return null;
	}

	return (
		<div className="flex-1 overflow-y-auto px-6 py-8">
			<div className="max-w-2xl mx-auto space-y-6">
				{messages.map((msg) => (
					<MessageBubble key={msg.id} message={msg} />
				))}
				{streamingContent !== null && (
					<div className="flex justify-start">
						<div className="max-w-[80%]">
							<MessageContent content={streamingContent} />
						</div>
					</div>
				)}
				<div ref={bottomRef} />
			</div>
		</div>
	);
}

function MessageBubble({ message }: { message: RichMessage }) {
	if (message.role === "user") {
		return (
			<div className="flex justify-end">
				<div className="bg-surface border border-border rounded-2xl px-5 py-3 max-w-[80%]">
					<p className="text-body-md text-cream whitespace-pre-wrap">
						{message.content}
					</p>
				</div>
			</div>
		);
	}

	return (
		<div className="flex justify-start">
			<div className="max-w-[80%]">
				<MessageContent content={message.content} />
				{message.components?.map((component, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: components have no stable id
					<InlineCard key={`${message.id}-card-${i}`} component={component} />
				))}
			</div>
		</div>
	);
}
