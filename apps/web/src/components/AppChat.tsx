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
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	useConversation,
	useConversations,
	useDeleteConversation,
} from "../hooks/useConversations";
import { useClickOutside } from "../hooks/useDom";
import { useMountEffect, useSyncRef } from "../hooks/useFoundation";
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
import { PracticeMode } from "./PracticeMode";
import { ScorePanel } from "./ScorePanel";
import {
	ChatSkeleton,
	ConversationSkeleton,
	FullPageSkeleton,
} from "./Skeleton";

export default function AppChat() {
	const { user, isLoading, isAuthenticated, signOut } = useAuth();
	const navigate = useNavigate();

	// Read conversationId reactively from URL params (survives route transitions)
	const conversationIdFromUrl = useRouterState({
		select: (s) => {
			const match = s.matches.find(
				(m) => m.routeId === "/app/c/$conversationId",
			);
			return (match?.params as Record<string, string>)?.conversationId ?? null;
		},
	});
	const [showProfile, setShowProfile] = useState(false);
	const [dropdownPos, setDropdownPos] = useState<{
		bottom: number;
		left: number;
	} | null>(null);
	const sidebarOpen = useUIStore((s) => s.sidebarOpen);
	const setSidebarOpen = useUIStore((s) => s.setSidebarOpen);
	const sidebarOpenRef = useSyncRef(sidebarOpen);
	const [searchOpen, setSearchOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");
	const searchInputRef = useRef<HTMLInputElement>(null);
	const theme = useThemeStore((s) => s.theme);
	const toggleTheme = useThemeStore((s) => s.toggleTheme);
	const profileRef = useRef<HTMLDivElement>(null);
	const addToast = useToastStore((s) => s.addToast);
	const scorePanelClear = useScorePanelStore((s) => s.clear);
	const scorePanelToggle = useScorePanelStore((s) => s.toggle);
	const scorePanelIsOpen = useScorePanelStore((s) => s.isOpen);
	const scorePanelSessionData = useScorePanelStore((s) => s.sessionData);

	useMountEffect(() => {
		const handler = (e: KeyboardEvent) => {
			if ((e.metaKey || e.ctrlKey) && e.key === "k") {
				e.preventDefault();
				if (!sidebarOpenRef.current) setSidebarOpen(true);
				setSearchOpen(true);
				requestAnimationFrame(() => {
					searchInputRef.current?.focus();
				});
			}
		};
		document.addEventListener("keydown", handler);
		return () => document.removeEventListener("keydown", handler);
	});

	const recordButtonRef = useRef<HTMLButtonElement>(null);

	// Chat state — URL is source of truth for conversation ID.
	// Override allows immediate ID switch before navigate completes (e.g., chat
	// stream returns a new conversation_id, we need useConversation to fetch it
	// before the deferred navigate updates the URL).
	const [conversationOverride, setConversationOverride] = useState<
		string | null
	>(null);
	const activeConversationId = conversationOverride ?? conversationIdFromUrl;
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

	// queryClient is stable for the life of the provider, so this callback is too.
	// That is what lets the deferred-synthesis effect below depend on it without
	// re-running on every render.
	const invalidateConversation = useCallback(
		(conversationId: string) => {
			queryClient.invalidateQueries({
				queryKey: ["conversation", conversationId],
			});
			queryClient.invalidateQueries({ queryKey: ["conversations"] });
		},
		[queryClient],
	);

	// Clear override once URL catches up (navigate completed)
	if (conversationOverride && conversationOverride === conversationIdFromUrl) {
		setConversationOverride(null);
	}

	// Check for deferred synthesis when a conversation loads.
	useEffect(() => {
		if (!activeConversationId) return;

		checkNeedsSynthesis(activeConversationId).then(async (sessionIds) => {
			if (sessionIds.length === 0) return;

			console.log(
				`[Deferred] Found ${sessionIds.length} sessions needing synthesis`,
			);
			for (const sid of sessionIds) {
				const result = await triggerDeferredSynthesis(sid);
				if (result?.status === "synthesized") {
					console.log(`[Deferred] Synthesis completed for session ${sid}`);
					// Refresh conversation messages to show the new synthesis
					invalidateConversation(activeConversationId);
				}
			}
		});
	}, [activeConversationId, invalidateConversation]);

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
						createdAt: new Date().toISOString(),
						streaming: true,
					},
				];
			});
		},
		onSummary: (_summary, conversationId) => {
			setShowListeningMode(false);

			const convId = conversationId ?? activeConversationId;
			if (convId) {
				if (convId !== activeConversationId) {
					navigate({
						to: "/app/c/$conversationId",
						params: { conversationId: convId },
						replace: true,
					});
				}
				queryClient
					.invalidateQueries({ queryKey: ["conversation", convId] })
					.then(() => setTransientMessages([]));
				invalidateConversation(convId);
			} else {
				setTransientMessages([]);
			}
		},
	});
	// Lazy-initialized from practice.state at mount: a fresh session always
	// mounts idle (false), but a session resumed mid-recording (e.g. this
	// component remounting while the hook's state is already non-idle) should
	// show the surface without waiting for a click.
	const [showListeningMode, setShowListeningMode] = useState(
		() => practice.state !== "idle",
	);

	function handleRecord() {
		setShowListeningMode(true);
		practice.start(activeConversationId ?? undefined);
	}

	function handleExitListeningMode() {
		setShowListeningMode(false);

		// If the practice session created a new conversation, navigate to it
		// so the chat view loads persisted observations from D1.
		const practiceConvId = practice.conversationId;
		if (practiceConvId && practiceConvId !== activeConversationId) {
			navigate({
				to: "/app/c/$conversationId",
				params: { conversationId: practiceConvId },
				replace: true,
			});
			invalidateConversation(practiceConvId);
		}
	}

	function handleStopPracticeMode() {
		practice.stop();
		handleExitListeningMode();
	}

	// Merge practice observation messages into the chat thread during recording
	const displayMessages = useMemo(() => {
		if (
			practice.state === "idle" ||
			practice.observationMessages.length === 0
		) {
			return messages;
		}
		// Deduplicate: don't show observations that are already in messages (from D1 reload)
		const existingObsIds = new Set(
			messages
				.filter((m) => m.messageType === "observation")
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
			setConversationOverride(null);
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
		setConversationOverride(null);
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
			createdAt: new Date().toISOString(),
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
							if (event.conversationId && !activeConversationId) {
								newConversationId = event.conversationId;
								// Don't setConversationOverride here — it would trigger
								// useConversation to fetch persisted messages while transient
								// messages are still showing, causing duplicates. The navigate
								// in the post-stream setTimeout handles the switch cleanly.
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
										createdAt: new Date().toISOString(),
										streaming: true,
									},
								];
							});
							break;
						}
						case "delta":
							appendDelta(event.text);
							break;
						case "tool_start": {
							// Discard any buffered pre-tool narration ("Let me search...")
							deltaBufferRef.current = "";
							setTransientMessages((prev) => {
								const updated = [...prev];
								const last = updated[updated.length - 1];
								if (last && last.role === "assistant") {
									updated[updated.length - 1] = {
										...last,
										content: "",
										toolCalls: [
											...(last.toolCalls ?? []),
											{ name: event.name, status: "pending" as const },
										],
									};
								}
								return updated;
							});
							break;
						}
						case "tool_result": {
							const components = event.componentsJson;
							const searchResult = components.find(
								(c) => c.type === "search_catalog_result",
							);
							const renderableComponents = components.filter(
								(c) => c.type !== "search_catalog_result",
							) as unknown as import("../lib/types").InlineComponent[];

							setTransientMessages((prev) => {
								const updated = [...prev];
								const last = updated[updated.length - 1];
								if (last && last.role === "assistant") {
									const toolCalls = [...(last.toolCalls ?? [])];
									let pendingIdx = -1;
									for (let i = toolCalls.length - 1; i >= 0; i--) {
										if (toolCalls[i].status === "pending") {
											pendingIdx = i;
											break;
										}
									}
									if (pendingIdx >= 0) {
										const pending = toolCalls[pendingIdx];
										if (searchResult) {
											const matches = searchResult.config.matches as Array<{
												title: string;
											}>;
											toolCalls[pendingIdx] =
												matches.length > 0
													? {
															name: pending.name,
															status: "found",
															label: `Found: ${matches[0].title}`,
														}
													: { name: pending.name, status: "not_found" };
										} else {
											toolCalls[pendingIdx] = {
												name: pending.name,
												status: "done",
											};
										}
									}
									updated[updated.length - 1] = {
										...last,
										toolCalls,
										components: [
											...(last.components ?? []),
											...renderableComponents,
										],
									};
								}
								return updated;
							});
							break;
						}
						case "tool_error": {
							setTransientMessages((prev) => {
								const updated = [...prev];
								const last = updated[updated.length - 1];
								if (last && last.role === "assistant") {
									const toolCalls = [...(last.toolCalls ?? [])];
									let pendingIdx = -1;
									for (let i = toolCalls.length - 1; i >= 0; i--) {
										if (toolCalls[i].status === "pending") {
											pendingIdx = i;
											break;
										}
									}
									if (pendingIdx >= 0) {
										toolCalls[pendingIdx] = {
											name: event.name,
											status: "error",
											message: event.message,
										};
									} else {
										toolCalls.push({
											name: event.name,
											status: "error",
											message: event.message,
										});
									}
									updated[updated.length - 1] = { ...last, toolCalls };
								}
								return updated;
							});
							break;
						}
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
						case "error": {
							if (rafIdRef.current) {
								cancelAnimationFrame(rafIdRef.current);
								rafIdRef.current = 0;
							}
							deltaBufferRef.current = "";
							const idx = streamingIndexRef.current;
							streamingIndexRef.current = -1;
							if (idx >= 0) {
								setTransientMessages((prev) =>
									prev.filter((_, i) => i !== idx),
								);
							}
							addToast({ type: "error", message: event.message });
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
				if (convId) {
					// Prime the query cache BEFORE navigating so the new route
					// has data immediately (avoids flash of empty content).
					await queryClient.invalidateQueries({
						queryKey: ["conversation", convId],
					});
				}
				if (newConversationId) {
					navigate({
						to: "/app/c/$conversationId",
						params: { conversationId: newConversationId },
						replace: true,
					});
				}
				// Clear transient messages after both query and navigation are done
				setTransientMessages([]);
				if (convId) {
					invalidateConversation(convId);
				} else {
					queryClient.invalidateQueries({ queryKey: ["conversations"] });
				}
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

	const handleSendRef = useRef(handleSend);
	handleSendRef.current = handleSend;
	const handleDecline = useCallback((focusDimension: string) => {
		void handleSendRef.current(
			`Not right now -- something else? (focus: ${focusDimension})`,
		);
	}, []);

	const handleTryExercises = useCallback(async (dimension: string) => {
		const { exercises } = await api.exercises.fetch({ dimension });
		if (exercises.length === 0) return;

		const exerciseMsg: RichMessage = {
			id: `exercises-${Date.now()}`,
			role: "assistant",
			content: `Here are some exercises to work on your ${dimension}:`,
			createdAt: new Date().toISOString(),
			components: [
				{
					type: "exercise_set" as const,
					config: {
						sourcePassage: "Based on your recent practice",
						targetSkill: `${dimension} improvement`,
						exercises: exercises.map((e) => ({
							title: e.title,
							instruction: e.instructions,
							focusDimension: e.dimensions[0] ?? dimension,
							exerciseId: e.id,
						})),
					},
				},
			],
		};

		setTransientMessages((prev) => [...prev, exerciseMsg]);
	}, []);

	const filteredConversations = useMemo(() => {
		if (!searchOpen || !searchQuery.trim()) return null;
		const q = searchQuery.toLowerCase();
		return conversations.filter((c) =>
			(c.title ?? "New conversation").toLowerCase().includes(q),
		);
	}, [searchOpen, searchQuery, conversations]);

	if (isLoading) {
		return <FullPageSkeleton />;
	}

	const hasMessages = messages.length > 0;
	const showConversationSkeleton =
		activeConversationId && isConversationLoading && messages.length === 0;
	const userInitial =
		user?.displayName?.charAt(0).toUpperCase() ??
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
				className={`shrink-0 border-r border-border-subtle flex flex-col py-4 transition-all duration-200 overflow-hidden bg-surface-page ${
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
								<span className="font-display text-body-md text-ink-primary truncate">
									crescend
								</span>
							</div>
							<button
								type="button"
								onClick={() => setSidebarOpen(false)}
								className="shrink-0 w-10 h-10 flex items-center justify-center rounded-lg text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition"
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
							className="w-10 h-10 flex items-center justify-center rounded-lg text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition mx-auto"
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
						{searchOpen && sidebarOpen ? (
							<div className="flex items-center gap-1 px-2 py-1">
								<MagnifyingGlass
									size={16}
									className="shrink-0 text-ink-tertiary"
								/>
								<input
									ref={searchInputRef}
									type="text"
									value={searchQuery}
									onChange={(e) => setSearchQuery(e.target.value)}
									onKeyDown={(e) => {
										if (e.key === "Escape") {
											setSearchOpen(false);
											setSearchQuery("");
										}
									}}
									placeholder="Search conversations..."
									className="flex-1 bg-transparent text-body-sm text-ink-primary placeholder:text-ink-tertiary outline-none min-w-0"
									// biome-ignore lint/a11y/noAutofocus: intentional UX for search activation
									autoFocus
								/>
								<button
									type="button"
									onClick={() => {
										setSearchOpen(false);
										setSearchQuery("");
									}}
									className="shrink-0 w-6 h-6 flex items-center justify-center text-ink-tertiary hover:text-ink-primary transition"
									aria-label="Close search"
								>
									<X size={14} />
								</button>
							</div>
						) : (
							<SidebarButton
								icon={<MagnifyingGlass size={20} />}
								label="Search"
								expanded={sidebarOpen}
								onClick={() => {
									if (!sidebarOpen) setSidebarOpen(true);
									setSearchOpen(true);
									requestAnimationFrame(() => {
										searchInputRef.current?.focus();
									});
								}}
							/>
						)}
					</div>
				</div>

				{/* Conversation list */}
				{sidebarOpen && (
					<div className="mt-4 flex-1 overflow-y-auto px-2">
						<span className="px-3 text-body-xs text-ink-tertiary uppercase tracking-wider">
							Recent
						</span>
						{isConversationsPending ? (
							<ConversationSkeleton />
						) : (
							<>
								{(filteredConversations ?? conversations.slice(0, 8)).map(
									(conv) => (
										// biome-ignore lint/a11y/useSemanticElements: the row contains its own delete <button>; a <button> wrapper would nest interactive controls.
										<div
											role="button"
											tabIndex={0}
											key={conv.id}
											className={`group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 cursor-pointer text-body-sm transition min-h-[36px] text-left ${
												conv.id === activeConversationId
													? "bg-surface-raised text-ink-primary"
													: "text-ink-secondary hover:text-ink-primary hover:bg-surface-raised"
											}`}
											onClick={() => {
												loadConversation(conv.id);
												setSearchOpen(false);
												setSearchQuery("");
											}}
											onKeyDown={(e) => {
												if (e.key === "Enter" || e.key === " ") {
													e.preventDefault();
													loadConversation(conv.id);
													setSearchOpen(false);
													setSearchQuery("");
												}
											}}
										>
											<ChatCircle size={14} className="shrink-0" />
											<span className="flex-1 truncate">
												{conv.title ?? "New conversation"}
											</span>
											{!searchOpen && (
												<button
													type="button"
													onClick={(e) => {
														e.stopPropagation();
														handleDeleteConversation(conv.id);
													}}
													className="opacity-0 group-hover:opacity-100 shrink-0 w-7 h-7 flex items-center justify-center text-ink-tertiary hover:text-ink-primary transition"
													aria-label="Delete conversation"
												>
													<Trash size={14} />
												</button>
											)}
										</div>
									),
								)}
								{filteredConversations !== null &&
									filteredConversations.length === 0 && (
										<div className="px-3 py-6 text-center">
											<span className="text-body-xs text-ink-tertiary">
												No conversations matching &lsquo;{searchQuery}&rsquo;
											</span>
										</div>
									)}
								{!searchOpen && conversations.length > 8 && (
									<button
										type="button"
										className="w-full mt-1 px-3 py-2 text-body-xs text-ink-tertiary hover:text-ink-primary transition text-left"
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
				<div ref={profileRef} className="mt-auto pt-2 relative">
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
						className={`flex items-center gap-3 min-h-[44px] transition hover:bg-surface-raised ${
							sidebarOpen
								? "w-[calc(100%-16px)] mx-2 px-3 rounded-lg"
								: "w-full justify-center rounded-none"
						}`}
					>
						<span className="shrink-0 w-8 h-8 bg-surface-raised border border-border-subtle rounded-full flex items-center justify-center text-body-sm text-ink-primary font-medium">
							{userInitial}
						</span>
						{sidebarOpen && (
							<div className="flex flex-col items-start min-w-0">
								<span className="text-body-sm text-ink-primary truncate">
									{user?.displayName ?? user?.email ?? "User"}
								</span>
							</div>
						)}
					</button>

					{showProfile && dropdownPos && (
						<div
							className="fixed bg-surface-raised border border-border-subtle rounded-lg py-1 min-w-[160px] z-50"
							style={{
								bottom: dropdownPos.bottom,
								left: dropdownPos.left + 8,
							}}
						>
							<button
								type="button"
								onClick={toggleTheme}
								className="w-full text-left px-4 py-2 text-body-sm text-ink-secondary hover:text-ink-primary hover:bg-surface-sunken transition rounded-lg flex items-center gap-2"
							>
								{theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
								<span>{theme === "light" ? "Dark Mode" : "Light Mode"}</span>
							</button>
							<button
								type="button"
								onClick={() => {
									setShowProfile(false);
									navigate({ to: "/app/sandbox" });
								}}
								className="w-full text-left px-4 py-2 text-body-sm text-ink-secondary hover:text-ink-primary hover:bg-surface-sunken transition rounded-lg flex items-center gap-2"
							>
								<span>Artifact Sandbox</span>
							</button>
							<button
								type="button"
								onClick={handleSignOut}
								className="w-full text-left px-4 py-2 text-body-sm text-ink-secondary hover:text-ink-primary hover:bg-surface-sunken transition rounded-lg flex items-center gap-2"
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
					className="fixed top-3 left-3 z-20 w-10 h-10 flex items-center justify-center rounded-lg text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition md:hidden"
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
						className="absolute top-3 right-3 z-20 flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-raised border border-border-subtle text-ink-secondary hover:text-ink-primary hover:bg-surface-sunken transition text-body-sm"
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
					<ChatMessages
						messages={displayMessages}
						onTryExercises={handleTryExercises}
						onDecline={handleDecline}
					>
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
				<div className="fixed inset-0 z-50 bg-surface-page">
					<PracticeMode
						userPickedPieceId={null}
						confidentGuess={null}
						marks={practice.marks}
						elapsedSeconds={practice.elapsedSeconds}
						isPlaying={practice.isPlaying}
						isRecording={practice.state === "recording"}
						onStop={handleStopPracticeMode}
					/>
				</div>
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
			className={`flex items-center text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition group relative rounded-lg ${
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
				<span className="absolute left-full ml-2 px-2 py-1 bg-surface-sunken rounded text-body-xs text-ink-primary whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
					{label}
				</span>
			)}
		</button>
	);
}
