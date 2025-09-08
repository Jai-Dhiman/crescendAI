import { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '../src/hooks';
import { authService } from '../src/services/auth';

export default function AuthScreen() {
  const { signInWithGoogle, isSigningIn, signInError } = useAuth();
  const [isConfigured, setIsConfigured] = useState(authService.isConfigured());

  const handleGoogleSignIn = async () => {
    try {
      // Check if OAuth is configured
      if (!isConfigured) {
        Alert.alert(
          'Configuration Required',
          'Google OAuth is not configured. Please set up your Google Client ID in the environment variables.',
          [{ text: 'OK' }]
        );
        return;
      }

      // Get Google access token
      const accessToken = await authService.signInWithGoogle();
      
      // Sign in with our backend
      await signInWithGoogle(accessToken);
      
      // Navigate to main app
      router.replace('/(tabs)');
    } catch (error) {
      console.error('Sign in error:', error);
      Alert.alert(
        'Sign In Failed',
        error instanceof Error ? error.message : 'An unexpected error occurred',
        [{ text: 'OK' }]
      );
    }
  };

  const showConfigStatus = () => {
    const config = authService.getConfigStatus();
    Alert.alert(
      'OAuth Configuration',
      `Platform: ${config.platform}
Has Client ID: ${config.hasClientId}
Has Web Client ID: ${config.hasWebClientId}
Redirect URI: ${config.redirectUri}`,
      [{ text: 'OK' }]
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Piano Performance Analyzer</Text>
        <Text style={styles.subtitle}>
          Analyze your piano performance with AI-powered feedback
        </Text>

        <TouchableOpacity
          style={[styles.button, isSigningIn && styles.buttonDisabled]}
          onPress={handleGoogleSignIn}
          disabled={isSigningIn}
        >
          <Text style={styles.buttonText}>
            {isSigningIn ? 'Signing in...' : 'Sign in with Google'}
          </Text>
        </TouchableOpacity>

        {signInError && (
          <Text style={styles.errorText}>
            {signInError.message}
          </Text>
        )}

        {!isConfigured && (
          <TouchableOpacity 
            style={styles.configButton}
            onPress={showConfigStatus}
          >
            <Text style={styles.configButtonText}>
              Configuration Status
            </Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 16,
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 48,
    color: '#666',
    lineHeight: 22,
  },
  button: {
    backgroundColor: '#4285F4',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 8,
    marginBottom: 16,
    minWidth: 200,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  errorText: {
    color: '#ff4444',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 16,
  },
  configButton: {
    marginTop: 32,
    padding: 8,
  },
  configButtonText: {
    color: '#666',
    fontSize: 12,
    textDecorationLine: 'underline',
  },
});