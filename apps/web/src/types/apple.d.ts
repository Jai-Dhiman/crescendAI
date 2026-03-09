interface AppleSignInI {
	auth: {
		init(config: {
			clientId: string;
			scope: string;
			redirectURI: string;
			usePopup: boolean;
		}): void;
		signIn(): Promise<{
			authorization: {
				id_token: string;
				code: string;
			};
			user?: {
				email?: string;
				name?: {
					firstName?: string;
					lastName?: string;
				};
			};
		}>;
	};
}

interface Window {
	AppleID?: AppleSignInI;
}
