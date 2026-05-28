import type { ReactNode } from "react";

// Borderless, chrome-light device frames (untitled.stream style): the app
// content is the focus, with just enough chrome to read as "web" vs "phone".

export function BrowserFrame({
	url = "crescend.ai",
	children,
	className = "",
}: {
	url?: string;
	children: ReactNode;
	className?: string;
}) {
	return (
		<div
			className={`rounded-2xl bg-[#141110] shadow-2xl ring-1 ring-white/5 overflow-hidden ${className}`}
		>
			<div className="flex items-center gap-3 h-9 px-4 border-b border-white/5">
				<div className="flex items-center gap-1.5">
					<span className="w-3 h-3 rounded-full bg-white/15" />
					<span className="w-3 h-3 rounded-full bg-white/15" />
					<span className="w-3 h-3 rounded-full bg-white/15" />
				</div>
				<div className="flex-1 flex justify-center">
					<div className="px-3 py-1 rounded-md bg-white/5 text-text-tertiary text-[11px] leading-none max-w-[60%] truncate">
						{url}
					</div>
				</div>
				<div className="w-12" />
			</div>
			<div className="bg-espresso">{children}</div>
		</div>
	);
}

export function PhoneFrame({
	children,
	className = "",
}: {
	children: ReactNode;
	className?: string;
}) {
	return (
		<div
			className={`rounded-[2.4rem] bg-[#141110] shadow-2xl ring-1 ring-white/5 p-2 ${className}`}
		>
			<div className="relative rounded-[2rem] overflow-hidden bg-espresso aspect-[9/19.5]">
				{/* Dynamic Island */}
				<div className="absolute top-2.5 left-1/2 -translate-x-1/2 w-20 h-5 rounded-full bg-black z-10" />
				{children}
				{/* Home indicator */}
				<div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-28 h-1 rounded-full bg-white/40 z-10" />
			</div>
		</div>
	);
}
