import { create } from "zustand";

export interface Toast {
	id: string;
	type: "info" | "success" | "error";
	message: string;
	duration: number;
}

interface ToastState {
	toasts: Toast[];
	addToast: (toast: Omit<Toast, "id" | "duration"> & { duration?: number }) => void;
	removeToast: (id: string) => void;
}

export const useToastStore = create<ToastState>((set) => ({
	toasts: [],
	addToast: (toast) => {
		const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
		const newToast: Toast = {
			id,
			type: toast.type,
			message: toast.message,
			duration: toast.duration ?? 5000,
		};
		set((state) => ({
			toasts: [...state.toasts.slice(-2), newToast],
		}));
	},
	removeToast: (id) =>
		set((state) => ({
			toasts: state.toasts.filter((t) => t.id !== id),
		})),
}));
