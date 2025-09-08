import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Recording, UploadProgress } from "../types";
import { createMMKVStorage } from "./mmkv";

interface RecordingsStore {
  // State
  recordings: Recording[];
  currentRecording: Recording | null;
  uploadProgress: Record<string, UploadProgress>;
  isRecording: boolean;

  // Actions
  addRecording: (recording: Recording) => void;
  updateRecording: (id: string, updates: Partial<Recording>) => void;
  removeRecording: (id: string) => void;
  setRecordings: (recordings: Recording[]) => void;
  setCurrentRecording: (recording: Recording | null) => void;
  setUploadProgress: (recordingId: string, progress: UploadProgress) => void;
  removeUploadProgress: (recordingId: string) => void;
  setIsRecording: (isRecording: boolean) => void;
  getRecordingById: (id: string) => Recording | undefined;
  getRecordingsByStatus: (status: Recording["status"]) => Recording[];
}

export const useRecordingsStore = create<RecordingsStore>()(
  persist(
    (set, get) => ({
      recordings: [],
      currentRecording: null,
      uploadProgress: {},
      isRecording: false,

      addRecording: (recording: Recording) =>
        set((state) => ({
          ...state,
          recordings: [recording, ...state.recordings],
        })),

      updateRecording: (id: string, updates: Partial<Recording>) =>
        set((state) => ({
          ...state,
          recordings: state.recordings.map((recording) =>
            recording.id === id ? { ...recording, ...updates } : recording
          ),
          currentRecording:
            state.currentRecording?.id === id
              ? { ...state.currentRecording, ...updates }
              : state.currentRecording,
        })),

      removeRecording: (id: string) =>
        set((state) => ({
          ...state,
          recordings: state.recordings.filter(
            (recording) => recording.id !== id
          ),
          currentRecording:
            state.currentRecording?.id === id ? null : state.currentRecording,
        })),

      setRecordings: (recordings: Recording[]) =>
        set((state) => ({ ...state, recordings })),

      setCurrentRecording: (recording: Recording | null) =>
        set((state) => ({ ...state, currentRecording: recording })),

      setUploadProgress: (recordingId: string, progress: UploadProgress) =>
        set((state) => ({
          ...state,
          uploadProgress: {
            ...state.uploadProgress,
            [recordingId]: progress,
          },
        })),

      removeUploadProgress: (recordingId: string) =>
        set((state) => {
          const { [recordingId]: removed, ...rest } = state.uploadProgress;
          return { ...state, uploadProgress: rest };
        }),

      setIsRecording: (isRecording: boolean) =>
        set((state) => ({ ...state, isRecording })),

      getRecordingById: (id: string) => {
        const { recordings } = get();
        return recordings.find((recording) => recording.id === id);
      },

      getRecordingsByStatus: (status: Recording["status"]) => {
        const { recordings } = get();
        return recordings.filter((recording) => recording.status === status);
      },
    }),
    {
      name: "recordings-store",
      storage: createMMKVStorage(),
      partialize: (state) => ({
        recordings: state.recordings,
        currentRecording: state.currentRecording,
      }),
    }
  )
);
