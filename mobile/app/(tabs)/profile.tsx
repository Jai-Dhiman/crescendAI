import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useAuth } from "../../src/hooks";
import { authService } from "../../src/services/auth";
import { useSettingsStore } from "../../src/stores";

export default function ProfileScreen() {
  const { user, signOut, isSigningOut } = useAuth();
  const {
    settings,
    updateNotificationSettings,
    resetSettings,
  } = useSettingsStore();

  const handleSignOut = () => {
    Alert.alert("Sign Out", "Are you sure you want to sign out?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign Out",
        style: "destructive",
        onPress: signOut,
      },
    ]);
  };

  const showConfigInfo = () => {
    const config = authService.getConfigStatus();
    Alert.alert(
      "App Configuration",
      `Platform: ${config.platform}
OAuth Configured: ${config.hasClientId ? "Yes" : "No"}
Redirect URI: ${config.redirectUri}`,
      [{ text: "OK" }]
    );
  };

  const toggleSetting = (
    section: "notifications" | "audio",
    key: string,
    currentValue: boolean
  ) => {
    if (section === "notifications") {
      updateNotificationSettings({ [key]: !currentValue });
    }
  };

  const renderToggleItem = (
    title: string,
    description: string,
    value: boolean,
    onToggle: () => void
  ) => (
    <TouchableOpacity style={styles.settingItem} onPress={onToggle}>
      <View style={styles.settingContent}>
        <Text style={styles.settingTitle}>{title}</Text>
        <Text style={styles.settingDescription}>{description}</Text>
      </View>
      <View style={[styles.toggle, value && styles.toggleActive]}>
        <Text style={styles.toggleText}>{value ? "ON" : "OFF"}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
    >
      {/* User Info */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Account</Text>
        {user ? (
          <View style={styles.userInfo}>
            <Text style={styles.userName}>{user.name}</Text>
            <Text style={styles.userEmail}>{user.email}</Text>
            <Text style={styles.userDate}>
              Member since {new Date(user.createdAt).toLocaleDateString()}
            </Text>
          </View>
        ) : (
          <Text style={styles.noUserText}>Not signed in</Text>
        )}
      </View>

      {/* Notification Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notifications</Text>

        {renderToggleItem(
          "Practice Reminders",
          "Get reminded to practice regularly",
          settings.notifications.practiceReminders,
          () =>
            toggleSetting(
              "notifications",
              "practiceReminders",
              settings.notifications.practiceReminders
            )
        )}

        {renderToggleItem(
          "Analysis Complete",
          "Notify when analysis is finished",
          settings.notifications.analysisComplete,
          () =>
            toggleSetting(
              "notifications",
              "analysisComplete",
              settings.notifications.analysisComplete
            )
        )}

        {renderToggleItem(
          "Weekly Reports",
          "Receive weekly progress reports",
          settings.notifications.weeklyReports,
          () =>
            toggleSetting(
              "notifications",
              "weeklyReports",
              settings.notifications.weeklyReports
            )
        )}
      </View>

      {/* Audio Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Audio Settings</Text>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingTitle}>Sample Rate</Text>
            <Text style={styles.settingDescription}>Recording quality</Text>
          </View>
          <Text style={styles.settingValue}>
            {settings.audio.sampleRate} Hz
          </Text>
        </View>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingTitle}>Format</Text>
            <Text style={styles.settingDescription}>Audio file format</Text>
          </View>
          <Text style={styles.settingValue}>
            {settings.audio.format.toUpperCase()}
          </Text>
        </View>
      </View>

      {/* App Info & Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>App</Text>

        <TouchableOpacity style={styles.actionItem} onPress={showConfigInfo}>
          <Text style={styles.actionText}>Configuration Info</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionItem} onPress={resetSettings}>
          <Text style={styles.actionText}>Reset Settings</Text>
        </TouchableOpacity>
      </View>

      {/* Sign Out */}
      <View style={styles.section}>
        <TouchableOpacity
          style={[
            styles.signOutButton,
            isSigningOut && styles.signOutButtonDisabled,
          ]}
          onPress={handleSignOut}
          disabled={isSigningOut}
        >
          <Text style={styles.signOutButtonText}>
            {isSigningOut ? "Signing Out..." : "Sign Out"}
          </Text>
        </TouchableOpacity>
      </View>

      {/* App Version */}
      <View style={styles.version}>
        <Text style={styles.versionText}>
          Piano Performance Analyzer v1.0.0
        </Text>
        <Text style={styles.versionText}>Built with Expo & React Native</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  contentContainer: {
    padding: 16,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 16,
    color: "#333",
  },
  userInfo: {
    padding: 16,
    backgroundColor: "#f8f9fa",
    borderRadius: 8,
  },
  userName: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 14,
    color: "#666",
    marginBottom: 8,
  },
  userDate: {
    fontSize: 12,
    color: "#999",
  },
  noUserText: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
    padding: 16,
  },
  settingItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 16,
    paddingHorizontal: 0,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: "500",
    color: "#333",
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 12,
    color: "#666",
  },
  settingValue: {
    fontSize: 14,
    color: "#007AFF",
    fontWeight: "600",
  },
  toggle: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#e9ecef",
    borderRadius: 4,
    minWidth: 50,
  },
  toggleActive: {
    backgroundColor: "#007AFF",
  },
  toggleText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#fff",
    textAlign: "center",
  },
  actionItem: {
    paddingVertical: 16,
    paddingHorizontal: 0,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  actionText: {
    fontSize: 16,
    color: "#007AFF",
    fontWeight: "500",
  },
  signOutButton: {
    backgroundColor: "#ff4444",
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  signOutButtonDisabled: {
    backgroundColor: "#ccc",
  },
  signOutButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  version: {
    alignItems: "center",
    paddingVertical: 24,
  },
  versionText: {
    fontSize: 12,
    color: "#999",
    marginBottom: 2,
  },
});
