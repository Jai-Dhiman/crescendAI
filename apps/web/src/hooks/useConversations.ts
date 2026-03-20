import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ConversationSummary, MessageRow } from "../lib/api";
import { api } from "../lib/api";
import type { InlineComponent } from "../lib/types";
import type { RichMessage } from "../lib/types";

function mapMessageRow(row: MessageRow): RichMessage {
	let components: InlineComponent[] | undefined;
	if (row.components_json) {
		try {
			components = JSON.parse(row.components_json);
		} catch {
			/* ignore malformed JSON */
		}
	}
	return {
		id: row.id,
		role: row.role,
		content: row.content,
		created_at: row.created_at,
		message_type: row.message_type as RichMessage["message_type"],
		dimension: row.dimension,
		framing: row.framing,
		session_id: row.session_id,
		components,
	};
}

export function useConversations(enabled = true) {
	return useQuery({
		queryKey: ["conversations"],
		queryFn: () => api.chat.list().then((r) => r.conversations),
		enabled,
	});
}

export function useConversation(id: string | null) {
	return useQuery({
		queryKey: ["conversation", id],
		queryFn: async () => {
			const data = await api.chat.get(id as string);
			return {
				...data,
				messages: data.messages.map(mapMessageRow),
			};
		},
		enabled: !!id,
	});
}

export function useDeleteConversation() {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: (id: string) => api.chat.delete(id),
		onMutate: async (id) => {
			await queryClient.cancelQueries({ queryKey: ["conversations"] });
			const previous = queryClient.getQueryData<ConversationSummary[]>([
				"conversations",
			]);
			queryClient.setQueryData<ConversationSummary[]>(
				["conversations"],
				(old) => old?.filter((c) => c.id !== id),
			);
			return { previous };
		},
		onError: (_err, _id, context) => {
			queryClient.setQueryData(["conversations"], context?.previous);
		},
		onSettled: () => {
			queryClient.invalidateQueries({ queryKey: ["conversations"] });
		},
	});
}

export function useDeleteConversations() {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: async (ids: string[]) => {
			await Promise.all(ids.map((id) => api.chat.delete(id)));
		},
		onMutate: async (ids) => {
			await queryClient.cancelQueries({ queryKey: ["conversations"] });
			const previous = queryClient.getQueryData<ConversationSummary[]>([
				"conversations",
			]);
			const idSet = new Set(ids);
			queryClient.setQueryData<ConversationSummary[]>(
				["conversations"],
				(old) => old?.filter((c) => !idSet.has(c.id)),
			);
			return { previous };
		},
		onError: (_err, _ids, context) => {
			queryClient.setQueryData(["conversations"], context?.previous);
		},
		onSettled: () => {
			queryClient.invalidateQueries({ queryKey: ["conversations"] });
		},
	});
}

export function useInvalidateConversations() {
	const queryClient = useQueryClient();
	return () => queryClient.invalidateQueries({ queryKey: ["conversations"] });
}
