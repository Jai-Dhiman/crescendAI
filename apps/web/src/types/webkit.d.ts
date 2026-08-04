// The iOS WKWebView injects `window.webkit.messageHandlers.<name>` into the
// scorehost page. It is absent in every other browser, hence optional.

interface WebKitMessageHandler {
	postMessage(message: unknown): void;
}

interface Window {
	webkit?: {
		messageHandlers?: Record<string, WebKitMessageHandler | undefined>;
	};
}
