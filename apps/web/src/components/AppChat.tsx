import {
	ChatCircle,
	MagnifyingGlass,
	PlusCircle,
	SidebarSimple,
	Trash,
	X,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import {
	useConversation,
	useConversations,
	useDeleteConversation,
} from "../hooks/useConversations";
import { usePracticeSession } from "../hooks/usePracticeSession";
import type { ChatStreamEvent } from "../lib/api";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import type { RichMessage } from "../lib/types";
import { useToastStore } from "../stores/toast";
import { useUIStore } from "../stores/ui";
import { ChatInput } from "./ChatInput";
import { ChatMessages } from "./ChatMessages";
import { RecordingBar } from "./RecordingBar";
import {
	ChatSkeleton,
	ConversationSkeleton,
	FullPageSkeleton,
} from "./Skeleton";

interface AppChatProps {
	initialConversationId?: string;
}

export default function AppChat({ initialConversationId }: AppChatProps) {
	const { user, isLoading, isAuthenticated, signOut } = useAuth();
	const navigate = useNavigate();
	const [showProfile, setShowProfile] = useState(false);
	const { sidebarOpen, setSidebarOpen } = useUIStore();
	const profileRef = useRef<HTMLDivElement>(null);
	const addToast = useToastStore((s) => s.addToast);

	// Chat state
	const [activeConversationId, setActiveConversationId] = useState<
		string | null
	>(initialConversationId ?? null);
	const [messages, setMessages] = useState<RichMessage[]>([]);
	const [streamingContent, setStreamingContent] = useState<string | null>(null);
	const [isStreaming, setIsStreaming] = useState(false);

	// Ref to track streaming content without nesting setState calls
	const streamingContentRef = useRef<string | null>(null);

	// Keep ref in sync with state
	useEffect(() => {
		streamingContentRef.current = streamingContent;
	}, [streamingContent]);

	// TanStack Query
	const queryClient = useQueryClient();
	const { data: conversations = [], isPending: isConversationsPending } =
		useConversations(isAuthenticated);
	const { data: conversationData, isPending: isConversationLoading } =
		useConversation(initialConversationId ?? null);
	const deleteConversation = useDeleteConversation();

	// Sync conversation data from query into local messages state
	useEffect(() => {
		if (conversationData) {
			setActiveConversationId(conversationData.conversation.id);
			setMessages(conversationData.messages);
		}
	}, [conversationData]);

	// Practice recording
	const practice = usePracticeSession();

	function handleRecord() {
		practice.start();
	}

	// When practice summary arrives, post it to chat
	useEffect(() => {
		if (practice.summary) {
			const summaryMsg: RichMessage = {
				id: `practice-${Date.now()}`,
				role: "assistant",
				content: practice.summary,
				created_at: new Date().toISOString(),
			};
			setMessages((prev) => [...prev, summaryMsg]);
		}
	}, [practice.summary]);

	// Redirect if not authenticated
	useEffect(() => {
		if (!isLoading && !isAuthenticated) {
			navigate({ to: "/signin" });
		}
	}, [isLoading, isAuthenticated, navigate]);

	// Click outside to close profile dropdown
	useEffect(() => {
		if (!showProfile) return;
		function handleClick(e: MouseEvent) {
			if (
				profileRef.current &&
				!profileRef.current.contains(e.target as Node)
			) {
				setShowProfile(false);
			}
		}
		document.addEventListener("mousedown", handleClick);
		return () => document.removeEventListener("mousedown", handleClick);
	}, [showProfile]);

	async function handleSignOut() {
		await signOut();
		navigate({ to: "/" });
	}

	const loadConversation = useCallback(
		(id: string) => {
			setSidebarOpen(false);
			navigate({
				to: "/app/c/$conversationId",
				params: { conversationId: id },
			});
		},
		[navigate, setSidebarOpen],
	);

	function handleNewChat() {
		setActiveConversationId(null);
		setMessages([]);
		setStreamingContent(null);
		setSidebarOpen(false);
		navigate({ to: "/app", replace: true });
	}

	function handleDeleteConversation(id: string) {
		deleteConversation.mutate(id, {
			onSuccess: () => {
				addToast({ type: "success", message: "Conversation deleted" });
			},
		});
		if (activeConversationId === id) {
			handleNewChat();
		}
	}

	async function handleSend(message: string) {
		if (isStreaming) return;

		const tempUserMsg: RichMessage = {
			id: `temp-${Date.now()}`,
			role: "user",
			content: message,
			created_at: new Date().toISOString(),
		};
		setMessages((prev) => [...prev, tempUserMsg]);
		setStreamingContent("");
		setIsStreaming(true);

		let newConversationId: string | null = null;

		try {
			await api.chat.send(
				message,
				activeConversationId,
				(event: ChatStreamEvent) => {
					switch (event.type) {
						case "start":
							if (event.conversation_id && !activeConversationId) {
								newConversationId = event.conversation_id;
								setActiveConversationId(event.conversation_id);
							}
							break;
						case "delta":
							if (event.text) {
								setStreamingContent((prev) => (prev ?? "") + event.text);
							}
							break;
						case "done": {
							const finalContent = streamingContentRef.current;
							if (finalContent !== null) {
								const assistantMsg: RichMessage = {
									id: event.message_id ?? `msg-${Date.now()}`,
									role: "assistant",
									content: finalContent,
									created_at: new Date().toISOString(),
								};
								setMessages((prev) => [...prev, assistantMsg]);
							}
							setStreamingContent(null);
							setIsStreaming(false);
							break;
						}
					}
				},
			);

			if (newConversationId) {
				navigate({
					to: "/app/c/$conversationId",
					params: { conversationId: newConversationId },
					replace: true,
				});
			}

			queryClient.invalidateQueries({ queryKey: ["conversations"] });
		} catch (e) {
			const errorMessage =
				e instanceof Error ? e.message : "Failed to send message";
			addToast({ type: "error", message: errorMessage });
			setStreamingContent(null);
			setIsStreaming(false);
		}
	}

	if (isLoading) {
		return <FullPageSkeleton />;
	}

	const hour = new Date().getHours();
	let greeting = "Good morning";
	if (hour >= 12 && hour < 17) greeting = "Good afternoon";
	else if (hour >= 17) greeting = "Good evening";

	const hasMessages = messages.length > 0 || streamingContent !== null;
	const showConversationSkeleton =
		initialConversationId && isConversationLoading && messages.length === 0;
	const userInitial =
		user?.display_name?.charAt(0).toUpperCase() ??
		user?.email?.charAt(0).toUpperCase() ??
		"?";

	return (
		<div className="h-dvh flex overflow-hidden">
			{/* Mobile sidebar backdrop */}
			{sidebarOpen && (
				<button
					type="button"
					className="fixed inset-0 bg-black/50 z-30 md:hidden"
					onClick={() => setSidebarOpen(false)}
					aria-label="Close sidebar"
				/>
			)}

			{/* Sidebar */}
			<aside
				className={`shrink-0 border-r border-border flex flex-col py-4 transition-all duration-200 overflow-hidden bg-espresso ${
					sidebarOpen
						? "fixed inset-y-0 left-0 z-40 w-64 md:relative md:w-56"
						: "w-0 md:w-12"
				}`}
			>
				<div className="flex items-center h-10 px-2 mb-2">
					{sidebarOpen ? (
						<>
							<div className="flex items-center gap-2 flex-1 min-w-0">
								<img
									src="/icon_nobackground.png"
									alt="crescend"
									className="w-7 h-7 shrink-0"
								/>
								<span className="font-display text-body-md text-cream truncate">
									crescend
								</span>
							</div>
							<button
								type="button"
								onClick={() => setSidebarOpen(false)}
								className="shrink-0 w-10 h-10 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
								aria-label="Collapse sidebar"
							>
								<X size={18} className="md:hidden" />
								<SidebarSimple size={18} className="hidden md:block" />
							</button>
						</>
					) : (
						<button
							type="button"
							onClick={() => setSidebarOpen(true)}
							className="w-10 h-10 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition mx-auto"
							aria-label="Expand sidebar"
						>
							<SidebarSimple size={20} />
						</button>
					)}
				</div>

				<div className="flex flex-col items-center">
					<div style={{ width: "100%" }}>
						<SidebarButton
							icon={
								<PlusCircle size={24} weight="fill" className="text-accent" />
							}
							label="New Chat"
							expanded={sidebarOpen}
							onClick={handleNewChat}
						/>
					</div>
					<div className="w-full">
						<SidebarButton
							icon={<MagnifyingGlass size={20} />}
							label="Search"
							expanded={sidebarOpen}
							onClick={() => {}}
						/>
					</div>
				</div>

				{/* Conversation list */}
				{sidebarOpen && (
					<div className="mt-4 flex-1 overflow-y-auto px-2">
						{isConversationsPending ? (
							<ConversationSkeleton />
						) : (
							conversations.map((conv) => (
								<button
									type="button"
									key={conv.id}
									className={`group flex w-full items-center gap-2 rounded-lg px-3 py-2 cursor-pointer text-body-sm transition min-h-[44px] text-left ${
										conv.id === activeConversationId
											? "bg-surface text-cream"
											: "text-text-secondary hover:text-cream hover:bg-surface"
									}`}
									onClick={() => loadConversation(conv.id)}
								>
									<ChatCircle size={16} className="shrink-0" />
									<span className="flex-1 truncate">
										{conv.title ?? "New conversation"}
									</span>
									<button
										type="button"
										onClick={(e) => {
											e.stopPropagation();
											handleDeleteConversation(conv.id);
										}}
										className="opacity-0 group-hover:opacity-100 shrink-0 w-8 h-8 flex items-center justify-center text-text-tertiary hover:text-cream transition"
										aria-label="Delete conversation"
									>
										<Trash size={14} />
									</button>
								</button>
							))
						)}
					</div>
				)}

				{/* Profile at sidebar bottom */}
				<div ref={profileRef} className="mt-auto pt-2 px-2 relative">
					<button
						type="button"
						onClick={() => setShowProfile(!showProfile)}
						className={`flex items-center gap-3 min-h-[44px] rounded-lg transition hover:bg-surface ${
							sidebarOpen ? "w-full px-2" : "justify-center mx-auto"
						}`}
					>
						<span className="shrink-0 w-8 h-8 bg-surface border border-border rounded-full flex items-center justify-center text-body-sm text-cream font-medium">
							{userInitial}
						</span>
						{sidebarOpen && (
							<div className="flex flex-col items-start min-w-0">
								<span className="text-body-sm text-cream truncate">
									{user?.display_name ?? user?.email ?? "User"}
								</span>
								<span className="text-body-xs text-text-tertiary">Pianist</span>
							</div>
						)}
					</button>

					{showProfile && (
						<div className="absolute left-2 bottom-full mb-2 bg-surface border border-border rounded-lg py-1 min-w-[140px] z-20">
							<button
								type="button"
								onClick={handleSignOut}
								className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition rounded-lg"
							>
								Sign Out
							</button>
						</div>
					)}
				</div>
			</aside>

			{/* Mobile sidebar toggle (visible when sidebar is collapsed on mobile) */}
			{!sidebarOpen && (
				<button
					type="button"
					onClick={() => setSidebarOpen(true)}
					className="fixed top-3 left-3 z-20 w-10 h-10 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition md:hidden"
					aria-label="Open sidebar"
				>
					<SidebarSimple size={20} />
				</button>
			)}

			{/* Main content */}
			<div className="flex-1 relative flex flex-col min-w-0">
				{practice.state !== "idle" && (
					<RecordingBar
						state={practice.state}
						elapsedSeconds={practice.elapsedSeconds}
						observations={practice.observations}
						analyserNode={practice.analyserNode}
						error={practice.error}
						chunksProcessed={practice.chunksProcessed}
						onStop={practice.stop}
					/>
				)}
				{showConversationSkeleton ? (
					<ChatSkeleton />
				) : !hasMessages ? (
					<div className="flex-1 flex flex-col items-center justify-center px-6 pb-[22vh]">
						<img
							src="/icon_nobackground.png"
							alt=""
							className="w-20 h-20 opacity-50 mb-6"
						/>
						<h1 className="font-display text-display-md text-cream mb-8">
							{greeting}.
						</h1>
						<ChatInput
							onSend={handleSend}
							onRecord={handleRecord}
							disabled={isStreaming || practice.state === "recording"}
							placeholder="What are you practicing today?"
							centered={true}
						/>
					</div>
				) : (
					<>
						<ChatMessages
							messages={messages}
							streamingContent={streamingContent}
						/>
						<div className="sticky bottom-0 bg-espresso">
							<ChatInput
								onSend={handleSend}
								onRecord={handleRecord}
								disabled={isStreaming || practice.state === "recording"}
								placeholder="Message your teacher..."
								centered={false}
							/>
						</div>
					</>
				)}
			</div>
		</div>
	);
}

function SidebarButton({
	icon,
	label,
	expanded = false,
	onClick,
}: {
	icon: React.ReactNode;
	label: string;
	expanded?: boolean;
	onClick?: () => void;
}) {
	return (
		<button
			type="button"
			onClick={onClick}
			className={`flex items-center text-text-secondary hover:text-cream hover:bg-surface transition group relative rounded-lg ${
				expanded
					? "w-[calc(100%-16px)] mx-2 px-3 min-h-[44px] gap-3"
					: "w-10 min-h-[44px] justify-center mx-auto"
			}`}
			aria-label={label}
		>
			<span className="shrink-0 w-6 flex items-center justify-center">
				{icon}
			</span>
			{expanded && (
				<span className="text-body-sm whitespace-nowrap">{label}</span>
			)}
			{!expanded && (
				<span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 rounded text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
					{label}
				</span>
			)}
		</button>
	);
}
