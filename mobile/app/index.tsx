import { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Redirect } from 'expo-router';
import { useAuth } from '../src/hooks';

export default function IndexScreen() {
  const { isAuthenticated, isLoading, checkAndRefreshToken } = useAuth();

  useEffect(() => {
    // Check and refresh token if needed on app start
    checkAndRefreshToken();
  }, []);

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  // Redirect based on authentication status
  if (isAuthenticated) {
    return <Redirect href="/(tabs)" />;
  } else {
    return <Redirect href="/auth" />;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
  },
});