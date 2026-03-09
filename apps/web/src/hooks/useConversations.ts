import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ConversationSummary } from "../lib/api";
import { api } from "../lib/api";

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
		queryFn: () => api.chat.get(id as string),
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

export function useInvalidateConversations() {
	const queryClient = useQueryClient();
	return () => queryClient.invalidateQueries({ queryKey: ["conversations"] });
}
