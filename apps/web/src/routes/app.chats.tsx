import {
	ArrowLeft,
	ChatCircle,
	CheckSquare,
	MagnifyingGlass,
	PlusCircle,
	Square,
	Trash,
} from "@phosphor-icons/react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
	useConversations,
	useDeleteConversations,
} from "../hooks/useConversations";

const PAGE_SIZE = 20;

function AllChats() {
	const navigate = useNavigate();
	const { data: conversations = [], isPending } = useConversations();
	const deleteMutation = useDeleteConversations();

	const [search, setSearch] = useState("");
	const [selected, setSelected] = useState<Set<string>>(new Set());
	const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

	const filtered = useMemo(() => {
		if (!search.trim()) return conversations;
		const q = search.toLowerCase();
		return conversations.filter((c) =>
			(c.title ?? "New conversation").toLowerCase().includes(q),
		);
	}, [conversations, search]);

	const visible = filtered.slice(0, visibleCount);
	const hasMore = visibleCount < filtered.length;

	function toggleSelect(id: string) {
		setSelected((prev) => {
			const next = new Set(prev);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			return next;
		});
	}

	function toggleSelectAll() {
		if (selected.size === filtered.length) {
			setSelected(new Set());
		} else {
			setSelected(new Set(filtered.map((c) => c.id)));
		}
	}

	function handleDeleteSelected() {
		const ids = Array.from(selected);
		if (ids.length === 0) return;
		deleteMutation.mutate(ids, {
			onSuccess: () => setSelected(new Set()),
		});
	}

	function formatDate(dateStr: string) {
		const date = new Date(dateStr);
		const now = new Date();
		const diffMs = now.getTime() - date.getTime();
		const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

		if (diffDays === 0) return "Today";
		if (diffDays === 1) return "Yesterday";
		if (diffDays < 7) return `${diffDays}d ago`;
		return date.toLocaleDateString(undefined, {
			month: "short",
			day: "numeric",
		});
	}

	return (
		<div className="min-h-screen bg-espresso text-cream">
			<div className="mx-auto max-w-2xl px-4 py-8">
				{/* Header */}
				<div className="flex items-center justify-between mb-6">
					<div className="flex items-center gap-3">
						<button
							type="button"
							onClick={() => navigate({ to: "/app" })}
							className="w-9 h-9 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
							aria-label="Back"
						>
							<ArrowLeft size={20} />
						</button>
						<h1 className="font-display text-display-sm">All Chats</h1>
					</div>
					<button
						type="button"
						onClick={() => navigate({ to: "/app" })}
						className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-on-accent text-body-sm font-medium transition hover:opacity-90"
					>
						<PlusCircle size={18} weight="fill" />
						New Chat
					</button>
				</div>

				{/* Search */}
				<div className="relative mb-4">
					<MagnifyingGlass
						size={18}
						className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none"
					/>
					<input
						type="text"
						placeholder="Search chats..."
						value={search}
						onChange={(e) => {
							setSearch(e.target.value);
							setVisibleCount(PAGE_SIZE);
						}}
						className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-surface border border-border text-body-sm text-cream placeholder:text-text-tertiary focus:outline-none focus:border-accent transition"
					/>
				</div>

				{/* Selection toolbar */}
				<div className={`flex items-center justify-between mb-3 px-3 py-2 rounded-lg border transition ${selected.size > 0 ? "bg-surface border-border" : "bg-surface/40 border-transparent"}`}>
					<div className="flex items-center gap-3">
						<button
							type="button"
							onClick={toggleSelectAll}
							disabled={filtered.length === 0}
							className={`text-body-xs transition ${selected.size > 0 ? "text-text-secondary hover:text-cream" : "text-text-tertiary/50 cursor-default"}`}
						>
							{selected.size > 0 && selected.size === filtered.length
								? "Deselect All"
								: "Select All"}
						</button>
						<span className={`text-body-xs transition ${selected.size > 0 ? "text-text-tertiary" : "text-text-tertiary/30"}`}>
							{selected.size > 0
								? `${selected.size} selected`
								: "None selected"}
						</span>
					</div>
					<button
						type="button"
						onClick={handleDeleteSelected}
						disabled={selected.size === 0 || deleteMutation.isPending}
						className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-body-xs transition ${selected.size > 0 ? "text-red-400 hover:bg-red-400/10" : "text-text-tertiary/30 cursor-default"}`}
					>
						<Trash size={14} />
						Delete
					</button>
				</div>

				{/* Conversation list */}
				{isPending ? (
					<div className="space-y-2">
						{Array.from({ length: 6 }).map((_, i) => (
							<div
								key={`skeleton-${i}`}
								className="h-12 rounded-lg bg-surface animate-pulse"
							/>
						))}
					</div>
				) : filtered.length === 0 ? (
					<div className="text-center py-16">
						<p className="text-text-secondary text-body-md mb-4">
							{search
								? "No chats match your search"
								: "No conversations yet"}
						</p>
						{!search && (
							<button
								type="button"
								onClick={() => navigate({ to: "/app" })}
								className="px-4 py-2 rounded-lg bg-accent text-on-accent text-body-sm font-medium transition hover:opacity-90"
							>
								Start a new chat
							</button>
						)}
					</div>
				) : (
					<div className="space-y-1">
						{visible.map((conv) => (
							<div
								role="button"
								tabIndex={0}
								key={conv.id}
								className="group flex items-center gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition hover:bg-surface"
								onClick={() =>
									navigate({
										to: "/app/c/$conversationId",
										params: { conversationId: conv.id },
									})
								}
								onKeyDown={(e) => {
									if (e.key === "Enter" || e.key === " ") {
										e.preventDefault();
										navigate({
											to: "/app/c/$conversationId",
											params: { conversationId: conv.id },
										});
									}
								}}
							>
								{/* Checkbox */}
								<button
									type="button"
									onClick={(e) => {
										e.stopPropagation();
										toggleSelect(conv.id);
									}}
									className="shrink-0 w-7 h-7 flex items-center justify-center text-text-tertiary hover:text-cream transition"
									aria-label={
										selected.has(conv.id) ? "Deselect" : "Select"
									}
								>
									{selected.has(conv.id) ? (
										<CheckSquare
											size={18}
											weight="fill"
											className="text-accent"
										/>
									) : (
										<Square
											size={18}
											className="opacity-0 group-hover:opacity-100 transition"
										/>
									)}
								</button>

								<ChatCircle
									size={16}
									className="shrink-0 text-text-tertiary"
								/>
								<span className="flex-1 truncate text-body-sm">
									{conv.title ?? "New conversation"}
								</span>
								<span className="shrink-0 text-body-xs text-text-tertiary">
									{formatDate(conv.updatedAt)}
								</span>
							</div>
						))}

						{/* Show More */}
						{hasMore && (
							<button
								type="button"
								onClick={() =>
									setVisibleCount((prev) => prev + PAGE_SIZE)
								}
								className="w-full mt-2 py-2.5 rounded-lg text-body-sm text-text-secondary hover:text-cream hover:bg-surface transition"
							>
								Show More
							</button>
						)}
					</div>
				)}
			</div>
		</div>
	);
}

export const Route = createFileRoute("/app/chats")({
	component: AllChats,
});
