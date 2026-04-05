import { useMountEffect } from "../hooks/useFoundation";
import { type Toast, useToastStore } from "../stores/toast";

function ToastItem({ toast }: { toast: Toast }) {
	const removeToast = useToastStore((s) => s.removeToast);

	// Auto-dismiss after duration -- runs once on mount
	useMountEffect(() => {
		const timer = setTimeout(() => removeToast(toast.id), toast.duration);
		return () => clearTimeout(timer);
	});

	const bgColor =
		toast.type === "error"
			? "border-red-500/40"
			: toast.type === "success"
				? "border-accent/40"
				: "border-border";

	return (
		<div
			className={`bg-surface-card border ${bgColor} rounded-lg px-4 py-3 shadow-card animate-slide-in-right min-w-[260px] max-w-sm`}
		>
			<p className="text-body-sm text-cream">{toast.message}</p>
		</div>
	);
}

export function ToastContainer() {
	const toasts = useToastStore((s) => s.toasts);

	if (toasts.length === 0) return null;

	return (
		<div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3">
			{toasts.map((toast) => (
				<ToastItem key={toast.id} toast={toast} />
			))}
		</div>
	);
}
