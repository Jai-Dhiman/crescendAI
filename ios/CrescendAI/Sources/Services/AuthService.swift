import Foundation
import Combine

// MARK: - Auth Service Protocol

protocol AuthServiceProtocol: ObservableObject {
    var authState: AuthState { get }
    var isAuthenticated: Bool { get }
    var currentUser: User? { get }
    
    func signIn(with token: String) async throws
    func signOut() async throws
    func refreshTokenIfNeeded() async throws
    func updateUser(_ user: User)
}

// MARK: - Auth Service Implementation

@MainActor
class AuthService: AuthServiceProtocol {
    
    // MARK: - Published Properties
    
    @Published private(set) var authState = AuthState()
    
    var isAuthenticated: Bool {
        authState.isAuthenticated
    }
    
    var currentUser: User? {
        authState.user
    }
    
    // MARK: - Private Properties
    
    private let apiClient: APIClientProtocol
    private let storage: AuthStorageProtocol
    
    // MARK: - Initialization
    
    init(apiClient: APIClientProtocol = APIClient(), storage: AuthStorageProtocol = KeychainAuthStorage()) {
        self.apiClient = apiClient
        self.storage = storage
        
        // Load saved auth state on initialization
        loadSavedAuthState()
    }
    
    // MARK: - Public Methods
    
    func signIn(with token: String) async throws {
        authState.isLoading = true
        
        do {
            let response = try await apiClient.authenticateWithGoogle(token: token)
            
            // Save tokens to secure storage
            try storage.saveTokens(response.data.tokens)
            
            // Update auth state
            authState.user = response.data.user
            authState.tokens = response.data.tokens
            authState.isAuthenticated = true
            authState.isLoading = false
            
        } catch {
            authState.isLoading = false
            throw error
        }
    }
    
    func signOut() async throws {
        authState.isLoading = true
        
        do {
            // Call API to invalidate session
            _ = try await apiClient.signOut()
            
            // Clear stored tokens
            try storage.deleteTokens()
            
            // Reset auth state
            authState = AuthState()
            
        } catch {
            // Even if API call fails, clear local state
            try storage.deleteTokens()
            authState = AuthState()
            
            // Re-throw error for UI handling
            throw error
        }
    }
    
    func refreshTokenIfNeeded() async throws {
        guard let tokens = authState.tokens else {
            throw AppError(code: "NO_TOKENS", message: "No tokens available for refresh")
        }
        
        // Check if token is expired or will expire soon (within 5 minutes)
        let expirationDate = ISO8601DateFormatter().date(from: tokens.expiresAt)
        let fiveMinutesFromNow = Date().addingTimeInterval(5 * 60)
        
        guard let expiration = expirationDate, expiration <= fiveMinutesFromNow else {
            return // Token is still valid
        }
        
        do {
            let response = try await apiClient.refreshToken(refreshToken: tokens.refreshToken)
            
            // Save new tokens
            let newTokens = response.data
            try storage.saveTokens(newTokens)
            
            // Update auth state
            authState.tokens = newTokens
            
        } catch {
            // If refresh fails, sign out the user
            try await signOut()
            throw AppError(code: "REFRESH_FAILED", message: "Failed to refresh authentication token")
        }
    }
    
    func updateUser(_ user: User) {
        authState.user = user
    }
    
    // MARK: - Private Methods
    
    private func loadSavedAuthState() {
        do {
            let tokens = try storage.loadTokens()
            
            // Check if tokens are still valid
            let expirationDate = ISO8601DateFormatter().date(from: tokens.expiresAt)
            
            if let expiration = expirationDate, expiration > Date() {
                authState.tokens = tokens
                authState.isAuthenticated = true
                
                // Load user data if available
                // This could be enhanced to cache user data locally
                Task {
                    await loadUserData()
                }
            } else {
                // Tokens are expired, clear them
                try storage.deleteTokens()
            }
            
        } catch {
            // No saved tokens or error loading them
            print("No saved authentication tokens found")
        }
    }
    
    private func loadUserData() async {
        guard let tokens = authState.tokens else { return }
        
        // This would typically involve decoding user data from the JWT token
        // or making an API call to get user profile
        // For now, we'll leave this as a placeholder
    }
}

// MARK: - Auth Storage Protocol

protocol AuthStorageProtocol {
    func saveTokens(_ tokens: AuthTokens) throws
    func loadTokens() throws -> AuthTokens
    func deleteTokens() throws
}

// MARK: - Keychain Auth Storage

class KeychainAuthStorage: AuthStorageProtocol {
    
    private let service = "com.crescendai.client.auth"
    private let tokenKey = "auth_tokens"
    
    func saveTokens(_ tokens: AuthTokens) throws {
        let data = try JSONEncoder().encode(tokens)
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: tokenKey,
            kSecValueData as String: data
        ]
        
        // Delete existing item first
        SecItemDelete(query as CFDictionary)
        
        // Add new item
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status != errSecSuccess {
            throw AppError(code: "KEYCHAIN_SAVE_ERROR", message: "Failed to save tokens to keychain")
        }
    }
    
    func loadTokens() throws -> AuthTokens {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: tokenKey,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecItemNotFound {
            throw AppError(code: "NO_TOKENS", message: "No tokens found in keychain")
        }
        
        if status != errSecSuccess {
            throw AppError(code: "KEYCHAIN_LOAD_ERROR", message: "Failed to load tokens from keychain")
        }
        
        guard let data = dataTypeRef as? Data else {
            throw AppError(code: "INVALID_TOKEN_DATA", message: "Invalid token data in keychain")
        }
        
        do {
            return try JSONDecoder().decode(AuthTokens.self, from: data)
        } catch {
            throw AppError(code: "TOKEN_DECODE_ERROR", message: "Failed to decode tokens")
        }
    }
    
    func deleteTokens() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: tokenKey
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        if status != errSecSuccess && status != errSecItemNotFound {
            throw AppError(code: "KEYCHAIN_DELETE_ERROR", message: "Failed to delete tokens from keychain")
        }
    }
}
