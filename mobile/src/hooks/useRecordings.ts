import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../services/api";
import { queryKeys } from "../services/queryClient";
import { useRecordingsStore } from "../stores";
import type { Recording } from "../types";
import { useAuth } from "./useAuth";

export function useRecordings() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const {
    recordings,
    addRecording,
    updateRecording,
    removeRecording,
    setRecordings,
  } = useRecordingsStore();

  // Get user recordings query
  const recordingsQuery = useQuery({
    queryKey: queryKeys.recordingsList(user?.id || ""),
    queryFn: async () => {
      if (!user?.id) throw new Error("No user ID");
      const response = await apiClient.getUserRecordings(user.id);
      setRecordings(response.data);
      return response;
    },
    enabled: !!user?.id,
  });

  // Create recording mutation
  const createRecordingMutation = useMutation({
    mutationFn: (
      recordingData: Omit<Recording, "id" | "createdAt" | "updatedAt">
    ) => apiClient.createRecording(recordingData),
    onSuccess: (response) => {
      addRecording(response.data);
      queryClient.invalidateQueries({
        queryKey: queryKeys.recordingsList(user?.id || ""),
      });
    },
  });

  // Update recording mutation
  const updateRecordingMutation = useMutation({
    mutationFn: ({
      id,
      updates,
    }: {
      id: string;
      updates: Partial<Recording>;
    }) => apiClient.updateRecording(id, updates),
    onSuccess: (response, variables) => {
      updateRecording(variables.id, response.data);
      queryClient.setQueryData(
        queryKeys.recording(variables.id),
        response.data
      );
      queryClient.invalidateQueries({
        queryKey: queryKeys.recordingsList(user?.id || ""),
      });
    },
  });

  // Delete recording mutation
  const deleteRecordingMutation = useMutation({
    mutationFn: (recordingId: string) => apiClient.deleteRecording(recordingId),
    onSuccess: (_, recordingId) => {
      removeRecording(recordingId);
      queryClient.removeQueries({ queryKey: queryKeys.recording(recordingId) });
      queryClient.invalidateQueries({
        queryKey: queryKeys.recordingsList(user?.id || ""),
      });
    },
  });

  // Upload recording mutation
  const uploadRecordingMutation = useMutation({
    mutationFn: ({
      recordingId,
      audioFile,
    }: {
      recordingId: string;
      audioFile: File | Blob;
    }) => apiClient.uploadRecording(recordingId, audioFile),
    onSuccess: (response, variables) => {
      updateRecording(variables.recordingId, {
        status: "uploading",
        audioUrl: response.data.uploadUrl,
      });
    },
  });

  const createRecording = (
    recordingData: Omit<Recording, "id" | "createdAt" | "updatedAt">
  ) => {
    return createRecordingMutation.mutateAsync(recordingData);
  };

  const updateRecordingById = (id: string, updates: Partial<Recording>) => {
    return updateRecordingMutation.mutateAsync({ id, updates });
  };

  const deleteRecording = (recordingId: string) => {
    return deleteRecordingMutation.mutateAsync(recordingId);
  };

  const uploadRecording = (recordingId: string, audioFile: File | Blob) => {
    return uploadRecordingMutation.mutateAsync({ recordingId, audioFile });
  };

  const refreshRecordings = () => {
    return recordingsQuery.refetch();
  };

  return {
    // Data
    recordings,

    // Query state
    isLoading: recordingsQuery.isLoading,
    isError: recordingsQuery.isError,
    error: recordingsQuery.error,

    // Actions
    createRecording,
    updateRecording: updateRecordingById,
    deleteRecording,
    uploadRecording,
    refreshRecordings,

    // Mutation states
    isCreating: createRecordingMutation.isPending,
    isUpdating: updateRecordingMutation.isPending,
    isDeleting: deleteRecordingMutation.isPending,
    isUploading: uploadRecordingMutation.isPending,

    // Errors
    createError: createRecordingMutation.error,
    updateError: updateRecordingMutation.error,
    deleteError: deleteRecordingMutation.error,
    uploadError: uploadRecordingMutation.error,
  };
}
