import { memo, useCallback, useEffect, useRef } from "react";
import type { RichMessage } from "../lib/types";
import { InlineCard } from "./InlineCard";
import { MessageContent } from "./MessageContent";

interface ChatMessagesProps {
	messages: RichMessage[];
	children?: React.ReactNode;
}

export function ChatMessages({ messages, children }: ChatMessagesProps) {
	const scrollContainerRef = useRef<HTMLDivElement>(null);
	const isNearBottomRef = useRef(true);
	const prevMessageCountRef = useRef(0);

	const scrollToBottom = useCallback(
		(behavior: ScrollBehavior = "instant") => {
			const container = scrollContainerRef.current;
			if (!container) return;
			if (behavior === "smooth") {
				container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
			} else {
				container.scrollTop = container.scrollHeight;
			}
		},
		[],
	);

	// Track whether user is near the bottom
	useEffect(() => {
		const container = scrollContainerRef.current;
		if (!container) return;

		function handleScroll() {
			if (!container) return;
			const threshold = 150;
			const distanceFromBottom =
				container.scrollHeight - container.scrollTop - container.clientHeight;
			isNearBottomRef.current = distanceFromBottom <= threshold;
		}

		container.addEventListener("scroll", handleScroll, { passive: true });
		return () => container.removeEventListener("scroll", handleScroll);
	}, []);

	// Auto-scroll on content changes
	useEffect(() => {
		if (!isNearBottomRef.current) return;

		const isNewMessage = messages.length > prevMessageCountRef.current;
		prevMessageCountRef.current = messages.length;

		// Instant scroll for streaming-related changes (avoids jerk);
		// smooth only for new non-streaming message additions
		const lastMsg = messages[messages.length - 1];
		const behavior =
			lastMsg?.streaming
				? "instant"
				: isNewMessage
					? "smooth"
					: "instant";
		scrollToBottom(behavior);
	}, [messages, scrollToBottom]);

	// Scroll on mount
	useEffect(() => {
		scrollToBottom("instant");
	}, [scrollToBottom]);

	if (messages.length === 0) {
		return null;
	}

	return (
		<div
			ref={scrollContainerRef}
			className="flex-1 overflow-y-auto px-6 pt-8 flex flex-col"
			style={{ scrollBehavior: "auto" }}
		>
			<div className="flex-1 max-w-3xl mx-auto space-y-6 w-full">
				{messages.map((msg) => (
					<MessageBubble key={msg.id} message={msg} />
				))}
			</div>
			{children}
		</div>
	);
}

const MessageBubble = memo(function MessageBubble({
	message,
}: { message: RichMessage }) {
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
		<div className="flex justify-start animate-fade-in">
			<div className="max-w-[80%]">
				<MessageContent content={message.content} />
				{message.components?.map((component, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: components have no stable id
					<InlineCard key={`${message.id}-card-${i}`} component={component} />
				))}
			</div>
		</div>
	);
});
