import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { createMMKVStorage } from './mmkv';
import type { UserSettings, NotificationSettings, AudioSettings, PrivacySettings } from '../types';

interface SettingsStore {
  // State
  settings: UserSettings;
  
  // Actions
  updateNotificationSettings: (settings: Partial<NotificationSettings>) => void;
  updateAudioSettings: (settings: Partial<AudioSettings>) => void;
  updatePrivacySettings: (settings: Partial<PrivacySettings>) => void;
  resetSettings: () => void;
}

const defaultSettings: UserSettings = {
  notifications: {
    practiceReminders: true,
    analysisComplete: true,
    weeklyReports: true,
  },
  audio: {
    sampleRate: 44100,
    bitRate: 128,
    format: 'm4a',
  },
  privacy: {
    shareProgress: false,
    allowAnalytics: true,
  },
};

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      settings: defaultSettings,

      updateNotificationSettings: (notificationUpdates: Partial<NotificationSettings>) =>
        set((state) => ({
          ...state,
          settings: {
            ...state.settings,
            notifications: {
              ...state.settings.notifications,
              ...notificationUpdates,
            },
          },
        })),

      updateAudioSettings: (audioUpdates: Partial<AudioSettings>) =>
        set((state) => ({
          ...state,
          settings: {
            ...state.settings,
            audio: {
              ...state.settings.audio,
              ...audioUpdates,
            },
          },
        })),

      updatePrivacySettings: (privacyUpdates: Partial<PrivacySettings>) =>
        set((state) => ({
          ...state,
          settings: {
            ...state.settings,
            privacy: {
              ...state.settings.privacy,
              ...privacyUpdates,
            },
          },
        })),

      resetSettings: () =>
        set(() => ({
          settings: defaultSettings,
        })),
    }),
    {
      name: 'settings-store',
      storage: createMMKVStorage(),
    }
  )
);