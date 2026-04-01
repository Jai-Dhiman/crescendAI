export const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

export const WS_BASE = import.meta.env.PROD
	? "wss://api.crescend.ai"
	: "ws://localhost:8787";
