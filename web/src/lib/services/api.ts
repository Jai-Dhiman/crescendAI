import { createWebApiClient } from '../utils/webApiClient.js';
import { useAuthStore } from '../stores/authStore.js';

// Get access token from auth store
const getAccessToken = () => {
  const { tokens } = useAuthStore.getState();
  return tokens?.accessToken || null;
};

// Create API client instance
export const apiClient = createWebApiClient(getAccessToken);

// Re-export the client for easy access
export default apiClient;
