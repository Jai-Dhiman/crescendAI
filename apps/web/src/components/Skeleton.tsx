interface SkeletonProps {
	className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
	return (
		<div
			className={`bg-surface rounded animate-shimmer bg-[length:200%_100%] bg-gradient-to-r from-surface via-surface-2 to-surface ${className}`}
		/>
	);
}

export function ConversationSkeleton() {
	return (
		<div className="px-2 space-y-1">
			{Array.from({ length: 6 }, (_, i) => (
				// biome-ignore lint/suspicious/noArrayIndexKey: static skeleton list
				<div key={i} className="flex items-center gap-2 px-3 py-2">
					<Skeleton className="w-4 h-4 rounded-full shrink-0" />
					<Skeleton
						className={`h-4 rounded ${i % 3 === 0 ? "w-28" : i % 3 === 1 ? "w-36" : "w-24"}`}
					/>
				</div>
			))}
		</div>
	);
}

export function MessageSkeleton({ align }: { align: "left" | "right" }) {
	if (align === "right") {
		return (
			<div className="flex justify-end">
				<div className="max-w-[80%] space-y-2">
					<Skeleton className="h-4 w-48 rounded" />
					<Skeleton className="h-4 w-32 rounded" />
				</div>
			</div>
		);
	}

	return (
		<div className="flex justify-start">
			<div className="max-w-[80%] space-y-2">
				<Skeleton className="h-4 w-56 rounded" />
				<Skeleton className="h-4 w-44 rounded" />
				<Skeleton className="h-4 w-36 rounded" />
			</div>
		</div>
	);
}

export function ChatSkeleton() {
	return (
		<div className="flex-1 overflow-hidden px-6 py-8">
			<div className="max-w-2xl mx-auto space-y-6">
				<MessageSkeleton align="right" />
				<MessageSkeleton align="left" />
				<MessageSkeleton align="right" />
			</div>
		</div>
	);
}

export function FullPageSkeleton() {
	return (
		<div className="h-dvh flex items-center justify-center">
			<div className="flex flex-col items-center gap-4">
				<Skeleton className="w-16 h-16 rounded-full" />
				<Skeleton className="h-4 w-32 rounded" />
			</div>
		</div>
	);
}
