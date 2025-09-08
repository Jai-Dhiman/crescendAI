import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../src/services/api';
import { queryKeys } from '../../src/services/queryClient';
import { useAuth } from '../../src/hooks';
import type { Progress, SkillProgression } from '../../src/types';

export default function ProgressScreen() {
  const { user } = useAuth();
  const [selectedTimeframe, setSelectedTimeframe] = useState<'week' | 'month' | 'all'>('week');

  const progressQuery = useQuery({
    queryKey: queryKeys.userProgress(user?.id || ''),
    queryFn: () => {
      if (!user?.id) throw new Error('No user ID');
      return apiClient.getUserProgress(user.id);
    },
    enabled: !!user?.id,
  });

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const renderSkillBar = (label: string, value: number) => (
    <View style={styles.skillItem} key={label}>
      <View style={styles.skillHeader}>
        <Text style={styles.skillLabel}>{label}</Text>
        <Text style={styles.skillValue}>{value}/100</Text>
      </View>
      <View style={styles.skillBarContainer}>
        <View style={[styles.skillBar, { width: `${value}%` }]} />
      </View>
    </View>
  );

  if (progressQuery.isLoading) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.loadingText}>Loading progress...</Text>
      </View>
    );
  }

  if (progressQuery.error || !progressQuery.data) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Failed to load progress data</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => progressQuery.refetch()}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const progress: Progress = progressQuery.data.data;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Overview Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Overview</Text>
        
        <View style={styles.statsGrid}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{progress.totalRecordings}</Text>
            <Text style={styles.statLabel}>Total Recordings</Text>
          </View>
          
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{formatTime(progress.totalPracticeTime)}</Text>
            <Text style={styles.statLabel}>Practice Time</Text>
          </View>
          
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{Math.round(progress.averageScore)}</Text>
            <Text style={styles.statLabel}>Average Score</Text>
          </View>
        </View>
      </View>

      {/* Skill Progression */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Skill Progression</Text>
        
        <View style={styles.skillsContainer}>
          {renderSkillBar('Timing', progress.skillProgression.timing)}
          {renderSkillBar('Pitch Accuracy', progress.skillProgression.pitch)}
          {renderSkillBar('Dynamics', progress.skillProgression.dynamics)}
          {renderSkillBar('Technique', progress.skillProgression.technique)}
        </View>
      </View>

      {/* Weekly Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        
        {progress.weeklyStats && progress.weeklyStats.length > 0 ? (
          <View style={styles.weeklyContainer}>
            {progress.weeklyStats.slice(0, 4).map((week, index) => (
              <View key={week.week} style={styles.weekItem}>
                <Text style={styles.weekLabel}>
                  Week {week.week.split('-W')[1]}
                </Text>
                <Text style={styles.weekTime}>
                  {formatTime(week.practiceTime)}
                </Text>
                <Text style={styles.weekRecordings}>
                  {week.recordingCount} recordings
                </Text>
                <Text style={styles.weekScore}>
                  Avg: {Math.round(week.averageScore)}
                </Text>
              </View>
            ))}
          </View>
        ) : (
          <Text style={styles.noDataText}>
            No activity data available yet. Start recording to see your progress!
          </Text>
        )}
      </View>

      {/* Placeholder for future features */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Insights</Text>
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>
            Detailed insights and recommendations will appear here based on your recordings and analysis results.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  contentContainer: {
    padding: 16,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  skillsContainer: {
    gap: 16,
  },
  skillItem: {
    marginBottom: 8,
  },
  skillHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  skillLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  skillValue: {
    fontSize: 14,
    color: '#666',
    fontFamily: 'monospace',
  },
  skillBarContainer: {
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    overflow: 'hidden',
  },
  skillBar: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  weeklyContainer: {
    gap: 12,
  },
  weekItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
  },
  weekLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  weekTime: {
    fontSize: 12,
    color: '#666',
    flex: 1,
    textAlign: 'center',
  },
  weekRecordings: {
    fontSize: 12,
    color: '#666',
    flex: 1,
    textAlign: 'center',
  },
  weekScore: {
    fontSize: 12,
    fontWeight: '600',
    color: '#007AFF',
    flex: 1,
    textAlign: 'right',
  },
  placeholder: {
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#e9ecef',
    borderStyle: 'dashed',
  },
  placeholderText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
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
  noDataText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
    lineHeight: 20,
  },
});