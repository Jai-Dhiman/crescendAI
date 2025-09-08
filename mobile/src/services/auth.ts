import * as AuthSession from "expo-auth-session";
import * as Crypto from "expo-crypto";
import * as WebBrowser from "expo-web-browser";
import { Platform } from "react-native";

// Complete the auth session for web
WebBrowser.maybeCompleteAuthSession();

// OAuth Configuration
const GOOGLE_CLIENT_ID = process.env.EXPO_PUBLIC_GOOGLE_CLIENT_ID || "";
const GOOGLE_CLIENT_ID_WEB = process.env.EXPO_PUBLIC_GOOGLE_CLIENT_ID_WEB || "";

// Get the appropriate client ID based on platform
const getGoogleClientId = () => {
  if (Platform.OS === "web") {
    return GOOGLE_CLIENT_ID_WEB;
  }
  return GOOGLE_CLIENT_ID;
};

// OAuth scopes
const SCOPES = ["openid", "profile", "email"];

export class AuthService {
  private discovery = AuthSession.makeRedirectUri({
    scheme: "piano-analyzer",
    path: "auth",
  });

  async signInWithGoogle(): Promise<string> {
    try {
      // For now, use implicit flow which is simpler but less secure
      // In production, implement proper PKCE flow
      const request = new AuthSession.AuthRequest({
        clientId: getGoogleClientId(),
        scopes: SCOPES,
        responseType: AuthSession.ResponseType.Token,
        redirectUri: this.discovery,
        extraParams: {
          access_type: "offline",
          prompt: "select_account",
        },
      });

      // Make the auth request
      const result = await request.promptAsync({
        authorizationEndpoint: "https://accounts.google.com/o/oauth2/v2/auth",
      });

      if (result.type === "success") {
        if (result.params.access_token) {
          return result.params.access_token;
        } else {
          throw new Error("Failed to get access token");
        }
      } else if (result.type === "cancel") {
        throw new Error("User cancelled authentication");
      } else {
        throw new Error("Authentication failed");
      }
    } catch (error) {
      console.error("Google OAuth error:", error);
      throw error instanceof Error
        ? error
        : new Error("Unknown authentication error");
    }
  }

  // Validate if OAuth is properly configured
  isConfigured(): boolean {
    const clientId = getGoogleClientId();
    return clientId !== "";
  }

  // Get configuration status for debugging
  getConfigStatus() {
    return {
      hasClientId: !!getGoogleClientId(),
      hasWebClientId: !!GOOGLE_CLIENT_ID_WEB,
      platform: Platform.OS,
      redirectUri: this.discovery,
    };
  }
}

export const authService = new AuthService();
