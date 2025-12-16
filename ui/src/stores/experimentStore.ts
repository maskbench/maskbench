import { create } from 'zustand';
import { Experiment, DashboardStats, ExperimentConfig, EstimatorId, MetricId } from '../types';
import {
  generateMockExperimentList,
  generateMockDashboardStats,
  generateMockExperiment
} from '../data/mockGenerator';
import { DEFAULT_ESTIMATORS, DEFAULT_METRICS } from '../data/constants';

interface ExperimentStore {
  // Data
  experiments: Experiment[];
  dashboardStats: DashboardStats | null;
  currentExperiment: Experiment | null;

  // New experiment form state
  newExperimentConfig: Partial<ExperimentConfig>;

  // UI state
  isLoading: boolean;
  error: string | null;

  // Actions
  loadExperiments: () => Promise<void>;
  loadDashboardStats: () => Promise<void>;
  getExperiment: (id: string) => Experiment | undefined;
  setCurrentExperiment: (id: string) => void;
  createExperiment: (config: ExperimentConfig) => Promise<Experiment>;
  deleteExperiment: (id: string) => Promise<void>;
  updateNewExperimentConfig: (config: Partial<ExperimentConfig>) => void;
  resetNewExperimentConfig: () => void;
  simulateExperimentRun: (id: string) => void;
}

const defaultNewExperimentConfig: Partial<ExperimentConfig> = {
  datasetName: '',
  videos: [],
  estimators: DEFAULT_ESTIMATORS,
  metrics: DEFAULT_METRICS,
  confidenceThreshold: 0.3,
  saveRenderings: true,
  savePoses: true
};

export const useExperimentStore = create<ExperimentStore>((set, get) => ({
  experiments: [],
  dashboardStats: null,
  currentExperiment: null,
  newExperimentConfig: { ...defaultNewExperimentConfig },
  isLoading: false,
  error: null,

  loadExperiments: async () => {
    set({ isLoading: true, error: null });
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      const experiments = generateMockExperimentList(12);
      set({ experiments, isLoading: false });
    } catch (error) {
      set({ error: 'Failed to load experiments', isLoading: false });
    }
  },

  loadDashboardStats: async () => {
    set({ isLoading: true, error: null });
    try {
      await new Promise(resolve => setTimeout(resolve, 300));
      const dashboardStats = generateMockDashboardStats();
      set({ dashboardStats, isLoading: false });
    } catch (error) {
      set({ error: 'Failed to load dashboard stats', isLoading: false });
    }
  },

  getExperiment: (id: string) => {
    return get().experiments.find(e => e.id === id);
  },

  setCurrentExperiment: (id: string) => {
    const experiment = get().experiments.find(e => e.id === id);
    set({ currentExperiment: experiment || null });
  },

  createExperiment: async (config: ExperimentConfig) => {
    set({ isLoading: true, error: null });
    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const experiment = generateMockExperiment('queued', {
        name: config.datasetName || 'New Experiment',
        config
      });

      set(state => ({
        experiments: [experiment, ...state.experiments],
        currentExperiment: experiment,
        isLoading: false
      }));

      return experiment;
    } catch (error) {
      set({ error: 'Failed to create experiment', isLoading: false });
      throw error;
    }
  },

  deleteExperiment: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      await new Promise(resolve => setTimeout(resolve, 300));
      set(state => ({
        experiments: state.experiments.filter(e => e.id !== id),
        currentExperiment: state.currentExperiment?.id === id ? null : state.currentExperiment,
        isLoading: false
      }));
    } catch (error) {
      set({ error: 'Failed to delete experiment', isLoading: false });
    }
  },

  updateNewExperimentConfig: (config: Partial<ExperimentConfig>) => {
    set(state => ({
      newExperimentConfig: { ...state.newExperimentConfig, ...config }
    }));
  },

  resetNewExperimentConfig: () => {
    set({ newExperimentConfig: { ...defaultNewExperimentConfig } });
  },

  simulateExperimentRun: (id: string) => {
    // Find experiment and start simulation
    const experiment = get().experiments.find(e => e.id === id);
    if (!experiment) return;

    // Update to processing status
    set(state => ({
      experiments: state.experiments.map(e =>
        e.id === id ? { ...e, status: 'processing' as const, progress: 0 } : e
      )
    }));

    // Simulate progress
    const interval = setInterval(() => {
      const current = get().experiments.find(e => e.id === id);
      if (!current || current.status !== 'processing') {
        clearInterval(interval);
        return;
      }

      const newProgress = Math.min(current.progress + Math.random() * 8 + 2, 100);

      if (newProgress >= 100) {
        // Complete the experiment
        const completed = generateMockExperiment('completed', {
          id,
          name: current.name,
          config: current.config,
          createdAt: current.createdAt
        });

        set(state => ({
          experiments: state.experiments.map(e =>
            e.id === id ? completed : e
          ),
          currentExperiment: state.currentExperiment?.id === id ? completed : state.currentExperiment
        }));
        clearInterval(interval);
      } else {
        set(state => ({
          experiments: state.experiments.map(e =>
            e.id === id ? { ...e, progress: newProgress } : e
          )
        }));
      }
    }, 500);
  }
}));
