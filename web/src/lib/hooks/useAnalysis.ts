import { createMutation, createQuery, useQueryClient } from '@tanstack/svelte-query';
import { crescendApi, CrescendApiError } from '../services/crescendApi.js';
import type { 
  UploadResponse, 
  AnalysisResponse, 
  JobStatus, 
  AnalysisResult 
} from '../schemas/api.js';

// Upload mutation
export const useUploadMutation = () => {
  return createMutation({
    mutationFn: async (file: File): Promise<UploadResponse> => {
      return crescendApi.uploadAudio(file);
    },
    onError: (error: CrescendApiError) => {
      console.error('Upload failed:', error.message);
    },
  });
};

// Analysis mutation  
export const useAnalysisMutation = () => {
  return createMutation({
    mutationFn: async (fileId: string): Promise<AnalysisResponse> => {
      return crescendApi.startAnalysis(fileId);
    },
    onError: (error: CrescendApiError) => {
      console.error('Analysis failed to start:', error.message);
    },
  });
};

// Job status query (for polling)
export const useJobStatusQuery = (jobId: string | null, enabled: boolean = true) => {
  return createQuery({
    queryKey: ['jobStatus', jobId],
    queryFn: () => crescendApi.getJobStatus(jobId!),
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Poll every 2 seconds if processing, stop if completed/failed
      return status === 'processing' || status === 'pending' ? 2000 : false;
    },
    retry: (failureCount, error) => {
      // Retry network errors, but not API errors
      return failureCount < 3 && !(error instanceof CrescendApiError);
    },
  });
};

// Analysis result query
export const useAnalysisResultQuery = (resultId: string | null, enabled: boolean = true) => {
  return createQuery({
    queryKey: ['analysisResult', resultId],
    queryFn: () => crescendApi.getAnalysisResult(resultId!),
    enabled: enabled && !!resultId,
    retry: (failureCount, error) => {
      return failureCount < 2 && !(error instanceof CrescendApiError);
    },
  });
};

// Complete upload and analysis workflow
export const useUploadAndAnalyzeMutation = () => {
  const queryClient = useQueryClient();
  
  return createMutation({
    mutationFn: async (params: {
      file: File;
      onProgress?: (stage: 'uploading' | 'analyzing' | 'processing', progress?: number) => void;
    }): Promise<AnalysisResult> => {
      return crescendApi.uploadAndAnalyze(params.file, params.onProgress);
    },
    onSuccess: (result: AnalysisResult) => {
      // Cache the result for future queries
      queryClient.setQueryData(['analysisResult', result.id], result);
    },
    onError: (error: CrescendApiError) => {
      console.error('Upload and analysis workflow failed:', error.message);
    },
  });
};

// Health check query
export const useHealthQuery = () => {
  return createQuery({
    queryKey: ['health'],
    queryFn: () => crescendApi.healthCheck(),
    retry: 2,
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
  });
};

// Custom hook for complete analysis workflow with state management
export const useAnalysisWorkflow = () => {
  const uploadMutation = useUploadMutation();
  const analysisMutation = useAnalysisMutation();
  
  const startWorkflow = async (
    file: File,
    onProgress?: (stage: 'uploading' | 'analyzing' | 'processing', progress?: number) => void
  ) => {
    try {
      // Step 1: Upload
      onProgress?.('uploading');
      const uploadResult = await uploadMutation.mutateAsync(file);
      
      // Step 2: Start analysis
      onProgress?.('analyzing');
      const analysisResult = await analysisMutation.mutateAsync(uploadResult.id);
      
      return {
        fileId: uploadResult.id,
        jobId: analysisResult.job_id,
      };
    } catch (error) {
      throw error;
    }
  };

  return {
    startWorkflow,
    isUploading: uploadMutation.isPending,
    isStartingAnalysis: analysisMutation.isPending,
    error: uploadMutation.error || analysisMutation.error,
  };
};