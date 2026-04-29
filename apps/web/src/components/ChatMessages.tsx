import { memo, useCallback, useEffect, useRef, useState } from "react";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";
import { useMountEffect } from "../hooks/useFoundation";
import type { RichMessage, ToolCallStatus } from "../lib/types";
import { Artifact } from "./Artifact";
import { MessageContent } from "./MessageContent";
import { ToolCallBar } from "./ToolCallBar";

interface ChatMessagesProps {
	messages: RichMessage[];
	children?: React.ReactNode;
	onTryExercises?: (dimension: string) => Promise<void>;
}

export function ChatMessages({
	messages,
	children,
	onTryExercises,
}: ChatMessagesProps) {
	const scrollContainerRef = useRef<HTMLDivElement>(null);
	const isNearBottomRef = useRef(true);
	const prevMessageCountRef = useRef(0);

	const scrollToBottom = useCallback((behavior: ScrollBehavior = "instant") => {
		const container = scrollContainerRef.current;
		if (!container) return;
		if (behavior === "smooth") {
			container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
		} else {
			container.scrollTop = container.scrollHeight;
		}
	}, []);

	// Track whether user is near the bottom of the scroll container.
	useMountEffect(() => {
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
	});

	// Auto-scroll on content changes
	useEffect(() => {
		if (!isNearBottomRef.current) return;

		const isNewMessage = messages.length > prevMessageCountRef.current;
		prevMessageCountRef.current = messages.length;

		// Instant scroll for streaming-related changes (avoids jerk);
		// smooth only for new non-streaming message additions
		const lastMsg = messages[messages.length - 1];
		const behavior = lastMsg?.streaming
			? "instant"
			: isNewMessage
				? "smooth"
				: "instant";
		scrollToBottom(behavior);
	}, [messages, scrollToBottom]);

	// Scroll on mount
	useMountEffect(() => {
		scrollToBottom("instant");
	});

	if (messages.length === 0) {
		return null;
	}

	return (
		<ArtifactScrollContext.Provider value={scrollContainerRef}>
			<div
				ref={scrollContainerRef}
				className="flex-1 overflow-y-auto px-6 pt-8 flex flex-col"
				style={{ scrollBehavior: "auto" }}
			>
				<div className="flex-1 max-w-3xl mx-auto space-y-6 w-full">
					{messages.map((msg) => (
						<MessageBubble
							key={msg.id}
							message={msg}
							onTryExercises={onTryExercises}
						/>
					))}
				</div>
				{children}
			</div>
		</ArtifactScrollContext.Provider>
	);
}

const MessageBubble = memo(function MessageBubble({
	message,
	onTryExercises,
}: {
	message: RichMessage;
	onTryExercises?: (dimension: string) => Promise<void>;
}) {
	const [tryState, setTryState] = useState<"idle" | "loading" | "error">(
		"idle",
	);

	// Session lifecycle dividers
	if (
		message.messageType === "session_start" ||
		message.messageType === "session_end"
	) {
		return (
			<div className="flex items-center gap-3 py-3">
				<div className="flex-1 border-t border-border" />
				<span className="text-xs text-text-tertiary whitespace-nowrap">
					{message.messageType === "session_start"
						? "Recording started"
						: "Recording ended"}
				</span>
				<div className="flex-1 border-t border-border" />
			</div>
		);
	}

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

	async function handleTryExercises() {
		if (!message.dimension || !onTryExercises) return;
		setTryState("loading");
		try {
			await onTryExercises(message.dimension);
			setTryState("idle");
		} catch (_err) {
			setTryState("error");
		}
	}

	const showTryButton =
		message.dimension && !message.streaming && onTryExercises;

	// For persisted messages, derive bar states from stored search_catalog_result components.
	// Streaming messages use message.toolCalls directly (set by the tool_start/tool_result handlers).
	const persistedBars: ToolCallStatus[] =
		!message.toolCalls && !message.streaming
			? (message.components ?? [])
					.filter(
						(c) => (c as { type: string }).type === "search_catalog_result",
					)
					.map((c) => {
						const matches =
							(
								c as {
									type: string;
									config: { matches?: Array<{ title: string }> };
								}
							).config.matches ?? [];
						return matches.length > 0
							? ({
									name: "search_catalog",
									status: "found",
									label: `Found: ${matches[0].title}`,
								} as ToolCallStatus)
							: ({
									name: "search_catalog",
									status: "not_found",
								} as ToolCallStatus);
					})
			: [];

	const displayToolCalls =
		message.toolCalls ?? (persistedBars.length > 0 ? persistedBars : undefined);

	// Don't pass search_catalog_result through to Artifact — it has no renderer
	const renderableComponents = (message.components ?? []).filter(
		(c) => (c as { type: string }).type !== "search_catalog_result",
	);

	return (
		<div className="flex justify-start animate-fade-in">
			<div className="max-w-[80%]">
				{message.messageType === "observation" && message.dimension && (
					<span className="inline-block text-xs px-2 py-0.5 rounded-full bg-surface-2 text-text-secondary mb-1">
						{message.dimension}
					</span>
				)}
				{displayToolCalls && displayToolCalls.length > 0 && (
					<div className="mb-2 space-y-1">
						{displayToolCalls.map((tc, i) => (
							// biome-ignore lint/suspicious/noArrayIndexKey: tool calls are positionally stable within a message
							<ToolCallBar key={i} toolCall={tc} />
						))}
					</div>
				)}
				<MessageContent content={message.content} />
				{!message.streaming &&
					renderableComponents.map((component, i) => (
						<Artifact
							// biome-ignore lint/suspicious/noArrayIndexKey: components have no stable id
							key={`${message.id}-artifact-${i}`}
							artifactId={`${message.id}-artifact-${i}`}
							component={component}
						/>
					))}
				{showTryButton && (
					<button
						type="button"
						onClick={handleTryExercises}
						disabled={tryState === "loading"}
						className={`mt-2 text-body-xs px-3 py-1.5 rounded-lg border transition ${
							tryState === "error"
								? "border-red-500 text-red-400 hover:bg-red-500/10"
								: "border-border text-text-tertiary hover:text-cream hover:border-accent hover:bg-surface disabled:opacity-50"
						}`}
					>
						{tryState === "loading"
							? "Loading exercises..."
							: tryState === "error"
								? "Try again"
								: "Try exercises for this"}
					</button>
				)}
			</div>
		</div>
	);
});
