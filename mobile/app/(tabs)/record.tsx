import { Audio } from "expo-av";
import * as FileSystem from "expo-file-system";
import { useEffect, useState } from "react";
import { Alert, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { useRecordings } from "../../src/hooks";
import { useRecordingsStore } from "../../src/stores";
import type { Recording } from "../../src/types";

export default function RecordScreen() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [permissionResponse, requestPermission] = Audio.usePermissions();

  const { createRecording, isCreating } = useRecordings();
  const { setIsRecording: setStoreIsRecording } = useRecordingsStore();

  useEffect(() => {
    setStoreIsRecording(isRecording);
  }, [isRecording, setStoreIsRecording]);

  const startRecording = async () => {
    try {
      // Request permissions if not granted
      if (permissionResponse?.status !== "granted") {
        const permission = await requestPermission();
        if (permission.status !== "granted") {
          Alert.alert(
            "Permission Required",
            "Microphone access is required to record audio."
          );
          return;
        }
      }

      // Configure audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      // Start recording
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);
      setRecordingDuration(0);

      // Update duration every second
      const interval = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);

      // Store interval ID for cleanup
      (recording as any).durationInterval = interval;
    } catch (error) {
      console.error("Failed to start recording:", error);
      Alert.alert(
        "Recording Error",
        "Failed to start recording. Please try again."
      );
    }
  };

  const stopRecording = async () => {
    if (!recording) return;

    try {
      setIsRecording(false);

      // Clear duration interval
      if ((recording as any).durationInterval) {
        clearInterval((recording as any).durationInterval);
      }

      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();

      if (uri) {
        // Create recording object
        const recordingData: Omit<Recording, "id" | "createdAt" | "updatedAt"> =
          {
            userId: "temp-user-id", // This will be replaced with actual user ID
            title: `Recording ${new Date().toLocaleString()}`,
            description: "",
            audioUrl: uri,
            localPath: uri,
            duration: recordingDuration,
            status: "recorded",
          };

        // Save recording
        await createRecording(recordingData);

        Alert.alert(
          "Recording Saved",
          `Recording saved successfully (${formatDuration(recordingDuration)})`,
          [{ text: "OK" }]
        );
      }

      // Reset state
      setRecording(null);
      setRecordingDuration(0);
    } catch (error) {
      console.error("Failed to stop recording:", error);
      Alert.alert(
        "Recording Error",
        "Failed to save recording. Please try again."
      );
    }
  };

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Record Performance</Text>

        {isRecording && (
          <View style={styles.recordingIndicator}>
            <Text style={styles.recordingText}>Recording...</Text>
            <Text style={styles.durationText}>
              {formatDuration(recordingDuration)}
            </Text>
          </View>
        )}

        <View style={styles.controls}>
          <TouchableOpacity
            style={[
              styles.recordButton,
              isRecording && styles.recordButtonActive,
              (isCreating || !permissionResponse) &&
                styles.recordButtonDisabled,
            ]}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={isCreating || !permissionResponse}
          >
            <Text
              style={[
                styles.recordButtonText,
                isRecording && styles.recordButtonTextActive,
              ]}
            >
              {isRecording ? "Stop Recording" : "Start Recording"}
            </Text>
          </TouchableOpacity>
        </View>

        {!permissionResponse && (
          <Text style={styles.permissionText}>
            Requesting microphone permission...
          </Text>
        )}

        {permissionResponse?.status === "denied" && (
          <Text style={styles.errorText}>
            Microphone permission denied. Please enable in settings.
          </Text>
        )}

        {isCreating && (
          <Text style={styles.savingText}>Saving recording...</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  content: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 48,
    color: "#333",
  },
  recordingIndicator: {
    alignItems: "center",
    marginBottom: 48,
  },
  recordingText: {
    fontSize: 18,
    color: "#ff4444",
    marginBottom: 8,
    fontWeight: "600",
  },
  durationText: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#ff4444",
    fontFamily: "monospace",
  },
  controls: {
    alignItems: "center",
  },
  recordButton: {
    backgroundColor: "#007AFF",
    paddingHorizontal: 48,
    paddingVertical: 24,
    borderRadius: 50,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  recordButtonActive: {
    backgroundColor: "#ff4444",
  },
  recordButtonDisabled: {
    backgroundColor: "#ccc",
  },
  recordButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
  },
  recordButtonTextActive: {
    color: "#fff",
  },
  permissionText: {
    marginTop: 24,
    fontSize: 14,
    color: "#666",
    textAlign: "center",
  },
  errorText: {
    marginTop: 24,
    fontSize: 14,
    color: "#ff4444",
    textAlign: "center",
  },
  savingText: {
    marginTop: 24,
    fontSize: 14,
    color: "#007AFF",
    textAlign: "center",
  },
});
