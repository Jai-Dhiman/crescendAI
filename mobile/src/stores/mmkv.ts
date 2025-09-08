import { MMKV } from "react-native-mmkv";
import type { PersistStorage } from "zustand/middleware";

// Initialize MMKV instance
export const mmkv = new MMKV();

// Create Zustand storage adapter for MMKV
export const createMMKVStorage = <T>(): PersistStorage<T> => ({
  setItem: (name, value) => {
    return mmkv.set(name, JSON.stringify(value));
  },
  getItem: (name) => {
    const value = mmkv.getString(name);
    return value ? JSON.parse(value) : null;
  },
  removeItem: (name) => {
    return mmkv.delete(name);
  },
});
