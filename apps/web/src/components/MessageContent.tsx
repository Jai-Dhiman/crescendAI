import { memo } from "react";
import type { Components } from "react-markdown";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const components: Components = {
	p({ children }) {
		return (
			<p className="text-body-md text-ink-primary mb-3 last:mb-0">{children}</p>
		);
	},
	strong({ children }) {
		return (
			<strong className="font-semibold text-ink-primary">{children}</strong>
		);
	},
	em({ children }) {
		return <em className="italic text-ink-secondary">{children}</em>;
	},
	code({ children, className }) {
		const isBlock = className?.startsWith("language-");
		if (isBlock) {
			return <code className="text-body-sm text-ink-primary">{children}</code>;
		}
		return (
			<code className="bg-accent/15 px-1.5 py-0.5 rounded text-body-sm text-accent/70">
				{children}
			</code>
		);
	},
	pre({ children }) {
		return (
			<pre className="bg-surface-raised border border-border-subtle rounded-lg p-4 overflow-x-auto mb-3 last:mb-0">
				{children}
			</pre>
		);
	},
	ul({ children }) {
		return (
			<ul className="list-disc list-outside ml-5 mb-3 last:mb-0 space-y-1 text-body-md text-ink-primary marker:text-accent">
				{children}
			</ul>
		);
	},
	ol({ children }) {
		return (
			<ol className="list-decimal list-outside ml-5 mb-3 last:mb-0 space-y-1 text-body-md text-ink-primary marker:text-accent">
				{children}
			</ol>
		);
	},
	li({ children }) {
		return <li className="text-body-md text-ink-primary">{children}</li>;
	},
	a({ children, href }) {
		return (
			<a
				href={href}
				target="_blank"
				rel="noopener noreferrer"
				className="text-accent hover:text-accent/70 underline underline-offset-2 transition-colors"
			>
				{children}
			</a>
		);
	},
	blockquote({ children }) {
		return (
			<blockquote className="border-l-2 border-accent pl-4 mb-3 last:mb-0 text-ink-secondary italic">
				{children}
			</blockquote>
		);
	},
	h1({ children }) {
		return (
			<h1 className="font-display text-display-sm text-ink-primary mb-3">
				{children}
			</h1>
		);
	},
	h2({ children }) {
		return (
			<h2 className="font-display text-body-lg font-semibold text-ink-primary mb-2">
				{children}
			</h2>
		);
	},
	h3({ children }) {
		return (
			<h3 className="font-display text-body-md font-semibold text-ink-primary mb-2">
				{children}
			</h3>
		);
	},
	hr() {
		return <hr className="border-accent/30 my-4" />;
	},
};

interface MessageContentProps {
	content: string;
}

export const MessageContent = memo(function MessageContent({
	content,
}: MessageContentProps) {
	const processed = content.replace(/(?<!-)--(?!-)/g, "\u2014");
	return (
		<Markdown remarkPlugins={[remarkGfm]} components={components}>
			{processed}
		</Markdown>
	);
});
