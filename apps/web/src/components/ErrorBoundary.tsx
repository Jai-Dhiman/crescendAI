import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
	children: ReactNode;
	pathname?: string;
}

interface ErrorBoundaryState {
	hasError: boolean;
	error: Error | null;
}

export class ErrorBoundary extends Component<
	ErrorBoundaryProps,
	ErrorBoundaryState
> {
	constructor(props: ErrorBoundaryProps) {
		super(props);
		this.state = { hasError: false, error: null };
	}

	static getDerivedStateFromError(error: Error): ErrorBoundaryState {
		return { hasError: true, error };
	}

	componentDidCatch(error: Error, errorInfo: ErrorInfo) {
		console.error("ErrorBoundary caught:", error, errorInfo);
	}

	componentDidUpdate(prevProps: ErrorBoundaryProps) {
		if (prevProps.pathname !== this.props.pathname && this.state.hasError) {
			this.setState({ hasError: false, error: null });
		}
	}

	render() {
		if (this.state.hasError) {
			return (
				<div className="h-dvh flex items-center justify-center px-6">
					<div className="text-center max-w-md">
						<h1 className="font-display text-display-sm text-cream mb-4">
							Something went wrong
						</h1>
						<p className="text-body-md text-text-secondary mb-6">
							{this.state.error?.message ?? "An unexpected error occurred."}
						</p>
						<button
							type="button"
							onClick={() => window.location.reload()}
							className="px-6 py-2.5 bg-accent hover:bg-accent-lighter text-espresso font-medium rounded-lg transition-colors"
						>
							Reload
						</button>
					</div>
				</div>
			);
		}

		return this.props.children;
	}
}
