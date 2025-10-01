import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Recording, UploadProgress } from '../types';
import { storage } from './storage';

interface RecordingsStore {
  recordings: Recording[];
  uploadProgress: Record<string, UploadProgress>;
  isLoading: boolean;

  // Recording actions
  addRecording: (recording: Recording) => void;
  updateRecording: (id: string, updates: Partial<Recording>) => void;
  removeRecording: (id: string) => void;
  setRecordings: (recordings: Recording[]) => void;

  // Upload progress actions
  setUploadProgress: (recordingId: string, progress: UploadProgress) => void;
  clearUploadProgress: (recordingId: string) => void;

  // Loading state
  setLoading: (loading: boolean) => void;

  // Utility getters
  getRecording: (id: string) => Recording | undefined;
  getRecentRecordings: (limit?: number) => Recording[];
}

const createRecordingsStore = () => create<RecordingsStore>()(
  persist(
    (set, get) => ({
      recordings: [],
      uploadProgress: {},
      isLoading: false,

      addRecording: (recording: Recording) => {
        set((state) => ({
          recordings: [recording, ...state.recordings],
        }));
      },

      updateRecording: (id: string, updates: Partial<Recording>) => {
        set((state) => ({
          recordings: state.recordings.map((recording) =>
            recording.id === id ? { ...recording, ...updates } : recording
          ),
        }));
      },

      removeRecording: (id: string) => {
        set((state) => ({
          recordings: state.recordings.filter((recording) => recording.id !== id),
        }));
      },

      setRecordings: (recordings: Recording[]) => {
        set({ recordings });
      },

      setUploadProgress: (recordingId: string, progress: UploadProgress) => {
        set((state) => ({
          uploadProgress: {
            ...state.uploadProgress,
            [recordingId]: progress,
          },
        }));
      },

      clearUploadProgress: (recordingId: string) => {
        set((state) => {
          const { [recordingId]: removed, ...rest } = state.uploadProgress;
          return { uploadProgress: rest };
        });
      },

      setLoading: (isLoading: boolean) => {
        set({ isLoading });
      },

      getRecording: (id: string) => {
        return get().recordings.find((recording) => recording.id === id);
      },

      getRecentRecordings: (limit = 10) => {
        return get()
          .recordings.slice()
          .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
          .slice(0, limit);
      },
    }),
    {
      name: 'recordings-storage',
      storage: {
        getItem: (name) => {
          const str = storage.getString(name);
          return str ? JSON.parse(str) : null;
        },
        setItem: (name, value) => {
          storage.setString(name, JSON.stringify(value));
        },
        removeItem: (name) => {
          storage.delete(name);
        },
      },
      // Only persist recordings
      partialize: (state: RecordingsStore): Partial<RecordingsStore> => ({ recordings: state.recordings }),
    }
  )
);

export const useRecordingsStore = createRecordingsStore();