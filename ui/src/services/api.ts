/**
 * API Service Layer for MaskBench UI
 *
 * This module provides a clean abstraction for data fetching that can be
 * easily switched between mock data and real backend API calls.
 *
 * Usage:
 *   import { api } from './services/api';
 *   const stats = await api.getDashboardStats();
 */

import {
  Experiment,
  DashboardStats,
  VideoFile,
  PoseResult,
  EstimatorId,
  MetricId
} from '../types';
import {
  generateMockDashboardStats,
  generateMockExperimentList,
  generateMockExperiment,
  generateMockPoseResult
} from '../data/mockGenerator';

// Configuration
const API_CONFIG = {
  baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  useMock: import.meta.env.VITE_USE_MOCK !== 'false', // Default to mock
  timeout: 30000,
};

// Types for API responses
interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
}

interface ExperimentCreateRequest {
  name: string;
  datasetName: string;
  videos: string[]; // video paths
  estimators: EstimatorId[];
  metrics: MetricId[];
  confidenceThreshold?: number;
  saveRenderings?: boolean;
  savePoses?: boolean;
}

interface ExperimentRunStatus {
  experimentId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStep?: string;
  error?: string;
}

// Helper for fetch with timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = API_CONFIG.timeout): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

// Mock implementations
const mockApi = {
  async getDashboardStats(): Promise<DashboardStats> {
    await delay(300); // Simulate network delay
    return generateMockDashboardStats();
  },

  async getExperiments(): Promise<Experiment[]> {
    await delay(200);
    return generateMockExperimentList(12);
  },

  async getExperiment(id: string): Promise<Experiment | null> {
    await delay(150);
    const experiments = generateMockExperimentList(12);
    return experiments.find(e => e.id === id) || generateMockExperiment('completed');
  },

  async createExperiment(request: ExperimentCreateRequest): Promise<Experiment> {
    await delay(500);
    return generateMockExperiment('draft', {
      name: request.name,
      config: {
        datasetName: request.datasetName,
        videos: request.videos.map((path, i) => ({
          id: `v${i}`,
          name: path.split('/').pop() || `video_${i}.mp4`,
          path,
          duration: 120,
          fps: 30,
          width: 1920,
          height: 1080,
          size: 100 * 1024 * 1024,
          frameCount: 3600
        })),
        estimators: request.estimators,
        metrics: request.metrics,
        confidenceThreshold: request.confidenceThreshold || 0.3,
        saveRenderings: request.saveRenderings ?? true,
        savePoses: request.savePoses ?? true
      }
    });
  },

  async runExperiment(experimentId: string): Promise<ExperimentRunStatus> {
    await delay(200);
    return {
      experimentId,
      status: 'queued',
      progress: 0,
      currentStep: 'Initializing...'
    };
  },

  async getExperimentStatus(experimentId: string): Promise<ExperimentRunStatus> {
    await delay(100);
    // Simulate progress
    const progress = Math.min(100, Math.random() * 20 + 10);
    return {
      experimentId,
      status: progress >= 100 ? 'completed' : 'processing',
      progress,
      currentStep: progress < 30 ? 'Loading models...' :
                   progress < 60 ? 'Processing videos...' :
                   progress < 90 ? 'Computing metrics...' : 'Finalizing...'
    };
  },

  async getPoseResult(experimentId: string, estimatorId: EstimatorId, videoName: string): Promise<PoseResult> {
    await delay(300);
    const video: VideoFile = {
      id: 'v1',
      name: videoName,
      path: `/videos/${videoName}`,
      duration: 120,
      fps: 30,
      width: 1920,
      height: 1080,
      size: 100 * 1024 * 1024,
      frameCount: 3600
    };
    return generateMockPoseResult(video, estimatorId);
  },

  async getAvailableVideos(): Promise<VideoFile[]> {
    await delay(200);
    return [
      { id: '1', name: 'TED_Talk_01.mp4', path: '/videos/TED_Talk_01.mp4', duration: 180, fps: 30, width: 1920, height: 1080, size: 150*1024*1024, frameCount: 5400 },
      { id: '2', name: 'conversation1_t3.mp4', path: '/videos/conversation1_t3.mp4', duration: 120, fps: 30, width: 1920, height: 1080, size: 100*1024*1024, frameCount: 3600 },
      { id: '3', name: 'interactive1_t1.mp4', path: '/videos/interactive1_t1.mp4', duration: 90, fps: 30, width: 1920, height: 1080, size: 75*1024*1024, frameCount: 2700 },
    ];
  },

  async deleteExperiment(experimentId: string): Promise<void> {
    await delay(200);
    console.log(`[Mock] Deleted experiment ${experimentId}`);
  }
};

// Real API implementations
const realApi = {
  async getDashboardStats(): Promise<DashboardStats> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/dashboard/stats`);
    const json: ApiResponse<DashboardStats> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async getExperiments(): Promise<Experiment[]> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments`);
    const json: ApiResponse<Experiment[]> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async getExperiment(id: string): Promise<Experiment | null> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments/${id}`);
    if (response.status === 404) return null;
    const json: ApiResponse<Experiment> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async createExperiment(request: ExperimentCreateRequest): Promise<Experiment> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    const json: ApiResponse<Experiment> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async runExperiment(experimentId: string): Promise<ExperimentRunStatus> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments/${experimentId}/run`, {
      method: 'POST'
    });
    const json: ApiResponse<ExperimentRunStatus> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async getExperimentStatus(experimentId: string): Promise<ExperimentRunStatus> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments/${experimentId}/status`);
    const json: ApiResponse<ExperimentRunStatus> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async getPoseResult(experimentId: string, estimatorId: EstimatorId, videoName: string): Promise<PoseResult> {
    const response = await fetchWithTimeout(
      `${API_CONFIG.baseUrl}/api/experiments/${experimentId}/poses/${estimatorId}/${encodeURIComponent(videoName)}`
    );
    const json: ApiResponse<PoseResult> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async getAvailableVideos(): Promise<VideoFile[]> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/videos`);
    const json: ApiResponse<VideoFile[]> = await response.json();
    if (json.status === 'error') throw new Error(json.message);
    return json.data;
  },

  async deleteExperiment(experimentId: string): Promise<void> {
    const response = await fetchWithTimeout(`${API_CONFIG.baseUrl}/api/experiments/${experimentId}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      const json = await response.json();
      throw new Error(json.message || 'Failed to delete experiment');
    }
  }
};

// Utility
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Export the API based on configuration
export const api = API_CONFIG.useMock ? mockApi : realApi;

// Export config for debugging
export const apiConfig = API_CONFIG;

// Export types
export type { ExperimentCreateRequest, ExperimentRunStatus, ApiResponse };
