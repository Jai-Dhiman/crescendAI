import { useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, Alert } from 'react-native';
import { useRecordings } from '../../src/hooks';
import type { Recording } from '../../src/types';

export default function RecordingsScreen() {
  const { 
    recordings, 
    isLoading, 
    error, 
    refreshRecordings,
    deleteRecording,
    isDeleting 
  } = useRecordings();

  useEffect(() => {
    refreshRecordings();
  }, []);

  const handleDeleteRecording = (recording: Recording) => {
    Alert.alert(
      'Delete Recording',
      `Are you sure you want to delete "${recording.title}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: () => deleteRecording(recording.id)
        }
      ]
    );
  };

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: Recording['status']): string => {
    switch (status) {
      case 'recorded': return '#666';
      case 'uploading': return '#ff9500';
      case 'processing': return '#007AFF';
      case 'analyzed': return '#34C759';
      case 'error': return '#ff4444';
      default: return '#666';
    }
  };

  const renderRecording = ({ item }: { item: Recording }) => (
    <View style={styles.recordingItem}>
      <View style={styles.recordingHeader}>
        <Text style={styles.recordingTitle}>{item.title}</Text>
        <TouchableOpacity 
          onPress={() => handleDeleteRecording(item)}
          disabled={isDeleting}
          style={styles.deleteButton}
        >
          <Text style={styles.deleteButtonText}>Delete</Text>
        </TouchableOpacity>
      </View>
      
      {item.description && (
        <Text style={styles.recordingDescription}>{item.description}</Text>
      )}
      
      <View style={styles.recordingMeta}>
        <Text style={styles.recordingDuration}>
          Duration: {formatDuration(item.duration)}
        </Text>
        <Text style={[styles.recordingStatus, { color: getStatusColor(item.status) }]}>
          {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
        </Text>
      </View>
      
      <Text style={styles.recordingDate}>
        {new Date(item.createdAt).toLocaleDateString()} at{' '}
        {new Date(item.createdAt).toLocaleTimeString()}
      </Text>
    </View>
  );

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.loadingText}>Loading recordings...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Failed to load recordings</Text>
        <TouchableOpacity style={styles.retryButton} onPress={refreshRecordings}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {recordings.length === 0 ? (
        <View style={styles.centerContainer}>
          <Text style={styles.emptyText}>No recordings yet</Text>
          <Text style={styles.emptySubtext}>
            Start recording your piano performances to see them here
          </Text>
        </View>
      ) : (
        <FlatList
          data={recordings}
          renderItem={renderRecording}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
          onRefresh={refreshRecordings}
          refreshing={isLoading}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  listContainer: {
    padding: 16,
  },
  recordingItem: {
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  recordingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  recordingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
    marginRight: 12,
  },
  deleteButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#ff4444',
    borderRadius: 4,
  },
  deleteButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  recordingDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  recordingMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  recordingDuration: {
    fontSize: 14,
    color: '#666',
    fontFamily: 'monospace',
  },
  recordingStatus: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  recordingDate: {
    fontSize: 12,
    color: '#999',
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    fontSize: 16,
    color: '#ff4444',
    textAlign: 'center',
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 6,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
});