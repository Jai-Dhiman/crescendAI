interface Logger {
	log: (...args: unknown[]) => void;
	warn: (...args: unknown[]) => void;
	error: (...args: unknown[]) => void;
}

export function createLogger(tag: string): Logger {
	const prefix = `[${tag}]`;

	if (!import.meta.env.DEV) {
		const noop = () => {};
		return { log: noop, warn: noop, error: noop };
	}

	return {
		log: (...args: unknown[]) => console.log("%s", prefix, ...args),
		warn: (...args: unknown[]) => console.warn("%s", prefix, ...args),
		error: (...args: unknown[]) => console.error("%s", prefix, ...args),
	};
}
