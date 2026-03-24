import {
	ChatCircle,
	MagnifyingGlass,
	Moon,
	MusicNote,
	PlusCircle,
	SidebarSimple,
	SignOut,
	Sun,
	Trash,
	X,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	useConversation,
	useConversations,
	useDeleteConversation,
} from "../hooks/useConversations";
import { useClickOutside } from "../hooks/useDom";
import { usePracticeSession } from "../hooks/usePracticeSession";
import type { ChatStreamEvent } from "../lib/api";
import { api, checkNeedsSynthesis, triggerDeferredSynthesis } from "../lib/api";
import { useAuth } from "../lib/auth";
import type { RichMessage } from "../lib/types";
import { useScorePanelStore } from "../stores/score-panel";
import { useThemeStore } from "../stores/theme";
import { useToastStore } from "../stores/toast";
import { useUIStore } from "../stores/ui";
import { ArtifactOverlay } from "./ArtifactOverlay";
import { ChatInput } from "./ChatInput";
import { ChatMessages } from "./ChatMessages";
import { ListeningMode } from "./ListeningMode";
import { ScorePanel } from "./ScorePanel";
import {
	ChatSkeleton,
	ConversationSkeleton,
	FullPageSkeleton,
} from "./Skeleton";

interface AppChatProps {
	initialConversationId?: string;
}

const GREETINGS = [
	// Warm / encouraging
	"Let's make some music.",
	"Your piano misses you.",
	"Ready when you are.",
	"Every session counts.",
	"Time to play.",
	"The keys are waiting.",
	"Pick up where you left off.",
	"Another day, another phrase.",
	"Let's hear what you've got.",
	"Music starts here.",
	"Let's turn practice into progress.",
	"Sound check: you're going to be great.",
	"Your fingers remember more than you think.",
	"One phrase at a time.",
	"Make it sing.",
	"Trust the process.",
	"Play something beautiful.",
	"You showed up. That's half the battle.",
	"Small steps, big music.",
	"Listen closely. You're getting better.",
	"The best practice is the one you do.",
	"Start slow. Finish strong.",
	"Today's the day it clicks.",
	"Your sound is yours alone.",
	"There's music in you.",
	"Just you and the keys.",
	"Let the music breathe.",
	"Feel it before you play it.",
	"Sit down. Breathe. Begin.",
	"Progress hides in repetition.",
	"You're closer than you think.",
	"The piano doesn't judge. Neither do we.",
	"This is your time.",
	"Play like nobody's listening.",
	"Welcome back.",
	// Witty / playful
	"Chopin never had an app this good.",
	"Scales won't practice themselves.",
	"Your future self will thank you.",
	"Beethoven practiced. So should you.",
	"Plot twist: practice is fun.",
	"Your neighbors called. They want an encore.",
	"No metronome judgment here.",
	"Rubato is not an excuse for wrong notes.",
	"One more rep. Of that tricky passage.",
	"Fingers warmed up? Didn't think so.",
	"Tempo: whatever you can handle.",
	"The sustain pedal forgives, but it doesn't forget.",
	"Debussy would be impressed. Probably.",
	"Somewhere, a metronome is ticking for you.",
	"Practice makes permanent.",
];

export default function AppChat({ initialConversationId }: AppChatProps) {
	const { user, isLoading, isAuthenticated, signOut } = useAuth();
	const navigate = useNavigate();
	const [showProfile, setShowProfile] = useState(false);
	const [dropdownPos, setDropdownPos] = useState<{
		bottom: number;
		left: number;
	} | null>(null);
	const sidebarOpen = useUIStore((s) => s.sidebarOpen);
	const setSidebarOpen = useUIStore((s) => s.setSidebarOpen);
	const theme = useThemeStore((s) => s.theme);
	const toggleTheme = useThemeStore((s) => s.toggleTheme);
	const profileRef = useRef<HTMLDivElement>(null);
	const addToast = useToastStore((s) => s.addToast);
	const scorePanelClear = useScorePanelStore((s) => s.clear);
	const scorePanelToggle = useScorePanelStore((s) => s.toggle);
	const scorePanelIsOpen = useScorePanelStore((s) => s.isOpen);
	const scorePanelSessionData = useScorePanelStore((s) => s.sessionData);

	const recordButtonRef = useRef<HTMLButtonElement>(null);
	const [showListeningMode, setShowListeningMode] = useState(false);
	const [recordButtonRect, setRecordButtonRect] = useState<DOMRect | null>(null);
	const [sessionNotes, setSessionNotes] = useState("");
	const [pieceContext, setPieceContext] = useState<{
		piece: string;
		section?: string;
	} | null>(null);

	// Chat state
	const [activeConversationId, setActiveConversationId] = useState<
		string | null
	>(initialConversationId ?? null);
	const [transientMessages, setTransientMessages] = useState<RichMessage[]>([]);
	const [isStreaming, setIsStreaming] = useState(false);

	// RAF-batched streaming refs
	const streamingIndexRef = useRef(-1);
	const deltaBufferRef = useRef("");
	const rafIdRef = useRef(0);

	const flushDeltas = useCallback(() => {
		rafIdRef.current = 0;
		const idx = streamingIndexRef.current;
		if (idx < 0) {
			// Streaming message not yet committed by React; retry next frame
			if (deltaBufferRef.current) {
				rafIdRef.current = requestAnimationFrame(flushDeltas);
			}
			return;
		}
		const buffered = deltaBufferRef.current;
		if (!buffered) return;
		deltaBufferRef.current = "";
		setTransientMessages((prev) => {
			const updated = [...prev];
			const msg = updated[idx];
			if (msg) {
				updated[idx] = { ...msg, content: msg.content + buffered };
			}
			return updated;
		});
	}, []);

	const appendDelta = useCallback(
		(text: string) => {
			deltaBufferRef.current += text;
			if (!rafIdRef.current) {
				rafIdRef.current = requestAnimationFrame(flushDeltas);
			}
		},
		[flushDeltas],
	);

	// TanStack Query
	const queryClient = useQueryClient();
	const { data: conversations = [], isPending: isConversationsPending } =
		useConversations(isAuthenticated);
	const { data: conversationData, isPending: isConversationLoading } =
		useConversation(activeConversationId);
	const deleteConversation = useDeleteConversation();

	// Sync active conversation ID when query data arrives
	const persistedConvId = conversationData?.conversation.id ?? null;
	if (persistedConvId && persistedConvId !== activeConversationId) {
		setActiveConversationId(persistedConvId);
	}

	// Check for deferred synthesis when a conversation loads
	useEffect(() => {
		if (!activeConversationId) return;

		checkNeedsSynthesis(activeConversationId).then(async (sessionIds) => {
			if (sessionIds.length === 0) return;

			console.log(`[Deferred] Found ${sessionIds.length} sessions needing synthesis`);
			for (const sid of sessionIds) {
				const result = await triggerDeferredSynthesis(sid);
				if (result?.status === "synthesized") {
					console.log(`[Deferred] Synthesis completed for session ${sid}`);
					// Refresh conversation messages to show the new synthesis
					queryClient.invalidateQueries({ queryKey: ["conversation", activeConversationId] });
				}
			}
		});
	}, [activeConversationId]);

	// Derive messages: persisted (from query) + transient (streaming/placeholders)
	const persistedMessages: RichMessage[] = conversationData?.messages ?? [];
	const messages = useMemo(
		() => [...persistedMessages, ...transientMessages],
		[persistedMessages, transientMessages],
	);

	// Practice recording — event-driven callbacks, no useEffect
	const practice = usePracticeSession({
		onSummarizing: () => {
			setTransientMessages((prev) => {
				if (prev.some((m) => m.id === "summarizing-placeholder")) return prev;
				return [
					...prev,
					{
						id: "summarizing-placeholder",
						role: "assistant" as const,
						content: "Reviewing your practice session...",
						created_at: new Date().toISOString(),
						streaming: true,
					},
				];
			});
		},
		onSummary: (_summary, conversationId) => {
			setSessionNotes("");
			setShowListeningMode(false);
			setRecordButtonRect(null);

			const convId = conversationId ?? activeConversationId;
			if (convId) {
				if (convId !== activeConversationId) {
					setActiveConversationId(convId);
					window.history.replaceState(
						window.history.state,
						"",
						`/app/c/${convId}`,
					);
				}
				queryClient
					.invalidateQueries({ queryKey: ["conversation", convId] })
					.then(() => setTransientMessages([]));
				queryClient.invalidateQueries({ queryKey: ["conversations"] });
			} else {
				setTransientMessages([]);
			}
		},
	});

	async function extractPieceContext(msgs: RichMessage[]) {
		if (msgs.length === 0) return;
		try {
			const apiBase = import.meta.env.PROD
				? "https://api.crescend.ai"
				: "http://localhost:8787";
			const conversationText = msgs
				.slice(-10)
				.map((m) => `${m.role}: ${m.content}`)
				.join("\n");

			const res = await fetch(`${apiBase}/api/extract-goals`, {
				method: "POST",
				credentials: "include",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					message: `Extract the piece name, composer, and section/bars being discussed from this conversation. Return JSON: {"piece": "Composer - Title", "section": "bars X-Y"} or null if no piece is mentioned.\n\n${conversationText}`,
				}),
			});

			if (res.ok) {
				const data = (await res.json()) as {
					piece?: string;
					section?: string;
				} | null;
				if (data?.piece) {
					setPieceContext({ piece: data.piece, section: data.section });
				}
			}
		} catch (e) {
			// Non-critical -- user can edit piece name manually
			console.error("Piece context extraction failed:", e);
		}
	}

	function handleRecord() {
		const rect = recordButtonRef.current?.getBoundingClientRect() ?? null;
		setRecordButtonRect(rect);
		setPieceContext(null);
		setShowListeningMode(true);
		practice.start(activeConversationId ?? undefined);
		extractPieceContext(messages);
	}

	function handleExitListeningMode() {
		setShowListeningMode(false);
		setRecordButtonRect(null);

		// If the practice session created a new conversation, navigate to it
		// so the chat view loads persisted observations from D1.
		const practiceConvId = practice.conversationId;
		if (practiceConvId && practiceConvId !== activeConversationId) {
			setActiveConversationId(practiceConvId);
			window.history.replaceState(
				window.history.state,
				"",
				`/app/c/${practiceConvId}`,
			);
			queryClient.invalidateQueries({
				queryKey: ["conversation", practiceConvId],
			});
			queryClient.invalidateQueries({ queryKey: ["conversations"] });
		}
	}

	// Merge practice observation messages into the chat thread during recording
	const displayMessages = useMemo(() => {
		if (practice.state === "idle" || practice.observationMessages.length === 0) {
			return messages;
		}
		// Deduplicate: don't show observations that are already in messages (from D1 reload)
		const existingObsIds = new Set(
			messages
				.filter((m) => m.message_type === "observation")
				.map((m) => m.content),
		);
		const newObs = practice.observationMessages.filter(
			(m) => !existingObsIds.has(m.content),
		);
		return [...messages, ...newObs];
	}, [messages, practice.observationMessages, practice.state]);


	// Click outside to close profile dropdown
	useClickOutside(profileRef, () => setShowProfile(false), showProfile);

	async function handleSignOut() {
		await signOut();
		navigate({ to: "/" });
	}

	const loadConversation = useCallback(
		(id: string) => {
			scorePanelClear();
			setTransientMessages([]);
			setSidebarOpen(false);
			navigate({
				to: "/app/c/$conversationId",
				params: { conversationId: id },
			});
		},
		[navigate, setSidebarOpen, scorePanelClear],
	);

	function handleNewChat() {
		setActiveConversationId(null);
		setTransientMessages([]);
		scorePanelClear();
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
		setTransientMessages((prev) => [...prev, tempUserMsg]);
		setIsStreaming(true);

		let newConversationId: string | null = null;

		try {
			await api.chat.send(
				message,
				activeConversationId,
				(event: ChatStreamEvent) => {
					switch (event.type) {
						case "start": {
							if (event.conversation_id && !activeConversationId) {
								newConversationId = event.conversation_id;
								setActiveConversationId(event.conversation_id);
							}
							// Append streaming placeholder to the transient array
							setTransientMessages((prev) => {
								streamingIndexRef.current = prev.length;
								return [
									...prev,
									{
										id: `streaming-${Date.now()}`,
										role: "assistant" as const,
										content: "",
										created_at: new Date().toISOString(),
										streaming: true,
									},
								];
							});
							break;
						}
						case "delta":
							if (event.text) {
								appendDelta(event.text);
							}
							break;
						case "done": {
							// Cancel pending RAF and flush remaining buffer
							if (rafIdRef.current) {
								cancelAnimationFrame(rafIdRef.current);
								rafIdRef.current = 0;
							}
							const remaining = deltaBufferRef.current;
							deltaBufferRef.current = "";
							const idx = streamingIndexRef.current;
							streamingIndexRef.current = -1;

							setTransientMessages((prev) => {
								const updated = [...prev];
								const msg = updated[idx];
								if (msg) {
									updated[idx] = {
										...msg,
										content: msg.content + remaining,
										streaming: false,
									};
								}
								return updated;
							});
							setIsStreaming(false);
							break;
						}
					}
				},
			);

			// Defer post-stream side effects so they don't interfere with
			// the "done" render commit (URL update, cache sync, refetch).
			const convId = newConversationId ?? activeConversationId;
			setTimeout(async () => {
				if (newConversationId) {
					window.history.replaceState(
						window.history.state,
						"",
						`/app/c/${newConversationId}`,
					);
				}
				if (convId) {
					// Wait for the conversation query to refetch from D1 before
					// clearing transient messages, to avoid a flash of missing content.
					await queryClient.invalidateQueries({
						queryKey: ["conversation", convId],
					});
					setTransientMessages([]);
				}
				queryClient.invalidateQueries({ queryKey: ["conversations"] });
			}, 0);
		} catch (e) {
			// Cancel pending RAF and clean up streaming message
			if (rafIdRef.current) {
				cancelAnimationFrame(rafIdRef.current);
				rafIdRef.current = 0;
			}
			deltaBufferRef.current = "";
			const idx = streamingIndexRef.current;
			streamingIndexRef.current = -1;
			if (idx >= 0) {
				setTransientMessages((prev) => prev.filter((_, i) => i !== idx));
			}

			const errorMessage =
				e instanceof Error ? e.message : "Failed to send message";
			addToast({ type: "error", message: errorMessage });
			setIsStreaming(false);
		}
	}

	const handleTryExercises = useCallback(
		async (dimension: string) => {
			const { exercises } = await api.exercises.fetch({ dimension });
			if (exercises.length === 0) return;

			const exerciseMsg: RichMessage = {
				id: `exercises-${Date.now()}`,
				role: "assistant",
				content: `Here are some exercises to work on your ${dimension}:`,
				created_at: new Date().toISOString(),
				components: [
					{
						type: "exercise_set" as const,
						config: {
							source_passage: "Based on your recent practice",
							target_skill: `${dimension} improvement`,
							exercises: exercises.map((e) => ({
								title: e.title,
								instruction: e.instructions,
								focus_dimension: e.dimensions[0] ?? dimension,
								exercise_id: e.id,
							})),
						},
					},
				],
			};

			setTransientMessages((prev) => [...prev, exerciseMsg]);
		},
		[],
	);

	const greeting = useMemo(
		() => GREETINGS[Math.floor(Math.random() * GREETINGS.length)],
		[],
	);

	if (isLoading) {
		return <FullPageSkeleton />;
	}

	const hasMessages = messages.length > 0;
	const showConversationSkeleton =
		activeConversationId && isConversationLoading && messages.length === 0;
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
						<span className="px-3 text-body-xs text-text-tertiary uppercase tracking-wider">
							Recent
						</span>
						{isConversationsPending ? (
							<ConversationSkeleton />
						) : (
							<>
								{conversations.slice(0, 8).map((conv) => (
									<div
										role="button"
										tabIndex={0}
										key={conv.id}
										className={`group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 cursor-pointer text-body-sm transition min-h-[36px] text-left ${
											conv.id === activeConversationId
												? "bg-surface text-cream"
												: "text-text-secondary hover:text-cream hover:bg-surface"
										}`}
										onClick={() => loadConversation(conv.id)}
										onKeyDown={(e) => {
											if (e.key === "Enter" || e.key === " ") {
												e.preventDefault();
												loadConversation(conv.id);
											}
										}}
									>
										<ChatCircle size={14} className="shrink-0" />
										<span className="flex-1 truncate">
											{conv.title ?? "New conversation"}
										</span>
										<button
											type="button"
											onClick={(e) => {
												e.stopPropagation();
												handleDeleteConversation(conv.id);
											}}
											className="opacity-0 group-hover:opacity-100 shrink-0 w-7 h-7 flex items-center justify-center text-text-tertiary hover:text-cream transition"
											aria-label="Delete conversation"
										>
											<Trash size={14} />
										</button>
									</div>
								))}
								{conversations.length > 8 && (
									<button
										type="button"
										className="w-full mt-1 px-3 py-2 text-body-xs text-text-tertiary hover:text-cream transition text-left"
										onClick={() => navigate({ to: "/app/chats" })}
									>
										See All Chats
									</button>
								)}
							</>
						)}
					</div>
				)}

				{/* Profile at sidebar bottom */}
				<div ref={profileRef} className="mt-auto pt-2 px-2 relative">
					<button
						type="button"
						onClick={() => {
							if (!showProfile && profileRef.current) {
								const rect = profileRef.current.getBoundingClientRect();
								setDropdownPos({
									bottom: window.innerHeight - rect.top + 8,
									left: rect.left,
								});
							}
							setShowProfile(!showProfile);
						}}
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

					{showProfile && dropdownPos && (
						<div
							className="fixed bg-surface border border-border rounded-lg py-1 min-w-[160px] z-50"
							style={{
								bottom: dropdownPos.bottom,
								left: dropdownPos.left + 8,
							}}
						>
							<button
								type="button"
								onClick={toggleTheme}
								className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition rounded-lg flex items-center gap-2"
							>
								{theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
								<span>{theme === "light" ? "Dark Mode" : "Light Mode"}</span>
							</button>
							<button
								type="button"
								onClick={handleSignOut}
								className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition rounded-lg flex items-center gap-2"
							>
								<SignOut size={16} />
								<span>Sign Out</span>
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
				{/* Score panel toggle button */}
				{scorePanelSessionData && !scorePanelIsOpen && (
					<button
						type="button"
						onClick={scorePanelToggle}
						className="absolute top-3 right-3 z-20 flex items-center gap-2 px-3 py-2 rounded-lg bg-surface border border-border text-text-secondary hover:text-cream hover:bg-surface-2 transition text-body-sm"
						aria-label="Open score panel"
					>
						<MusicNote size={16} className="text-accent" />
						<span className="hidden sm:inline">View Score</span>
					</button>
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
							{greeting}
						</h1>
						<ChatInput
							onSend={handleSend}
							onRecord={handleRecord}
							disabled={isStreaming || practice.state === "recording"}
							placeholder="What are you practicing today?"
							centered={true}
							recordButtonRef={recordButtonRef}
						/>
					</div>
				) : (
					<ChatMessages messages={displayMessages} onTryExercises={handleTryExercises}>
						<div className="sticky bottom-0">
							<ChatInput
								onSend={handleSend}
								onRecord={handleRecord}
								disabled={isStreaming || practice.state === "recording"}
								placeholder="Message your teacher..."
								centered={false}
								recordButtonRef={recordButtonRef}
							/>
						</div>
					</ChatMessages>
				)}
			</div>

			{/* Listening mode overlay */}
			{showListeningMode && (
				<ListeningMode
					state={practice.state}
					energy={practice.energy}
					isPlaying={practice.isPlaying}
					latestScores={practice.latestScores}
					error={practice.error}
					wsStatus={practice.wsStatus}
					onStop={practice.stop}
					originRect={recordButtonRect}
					onExit={handleExitListeningMode}
					pieceContext={pieceContext}
					sessionNotes={sessionNotes}
					onNotesChange={setSessionNotes}
				/>
			)}

			{/* Artifact expanded overlay */}
			<ArtifactOverlay />

			{/* Score panel (artifacts-style right sidebar) */}
			<ScorePanel />
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
