// Core types for MaskBench UI

export type ExperimentStatus = 'draft' | 'queued' | 'processing' | 'completed' | 'failed';

export type EstimatorId = 'yolo-nano' | 'yolo-small' | 'yolo-medium' | 'yolo-large' | 'yolo-xlarge'
  | 'mediapipe-lite' | 'mediapipe-full' | 'mediapipe-heavy'
  | 'openpose-body25' | 'openpose-body25b'
  | 'maskanyone-api-mp' | 'maskanyone-api-op' | 'maskanyone-ui-mp' | 'maskanyone-ui-op';

export type MetricId = 'velocity' | 'acceleration' | 'jerk' | 'pck' | 'rmse' | 'euclidean';

export type MaskingStrategy = 'none' | 'blur' | 'pixelate' | 'contour' | 'solid';

export interface Estimator {
  id: EstimatorId;
  name: string;
  shortName: string;
  description: string;
  color: string;
  category: 'yolo' | 'mediapipe' | 'openpose' | 'maskanyone';
}

export interface Metric {
  id: MetricId;
  name: string;
  description: string;
  requiresGroundTruth: boolean;
  lowerIsBetter: boolean;
  unit: string;
}

export interface VideoFile {
  id: string;
  name: string;
  path: string;
  duration: number; // seconds
  fps: number;
  width: number;
  height: number;
  size: number; // bytes
  frameCount: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: ExperimentStatus;
  progress: number; // 0-100
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
  config: ExperimentConfig;
  results?: ExperimentResults;
  error?: string;
}

export interface ExperimentConfig {
  datasetName: string;
  videos: VideoFile[];
  estimators: EstimatorId[];
  metrics: MetricId[];
  maskingStrategy?: MaskingStrategy;
  confidenceThreshold: number;
  saveRenderings: boolean;
  savePoses: boolean;
}

export interface ExperimentResults {
  inferenceTime: Record<EstimatorId, Record<string, number>>; // estimator -> video -> seconds
  metrics: Record<MetricId, MetricResults>;
  poses: Record<EstimatorId, Record<string, PoseResult>>; // estimator -> video -> poses
}

export interface MetricResults {
  byEstimator: Record<EstimatorId, number>;
  byVideo: Record<string, Record<EstimatorId, number>>;
  byKeypoint?: Record<string, Record<EstimatorId, number>>;
}

export interface PoseResult {
  videoName: string;
  fps: number;
  frameCount: number;
  frames: FramePose[];
}

export interface FramePose {
  frameIndex: number;
  persons: PersonPose[];
}

export interface PersonPose {
  id: number;
  keypoints: Keypoint[];
  boundingBox?: BoundingBox;
}

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
  name: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Dashboard stats
export interface DashboardStats {
  totalExperiments: number;
  completedExperiments: number;
  totalVideosProcessed: number;
  availableEstimators: number;
  avgProcessingTime: number; // hours
  weeklyActivity: WeeklyActivity[];
  recentExperiments: Experiment[];
}

export interface WeeklyActivity {
  day: string;
  experiments: number;
  videos: number;
}

// COCO keypoint names
export const COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
] as const;

export const COCO_SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4], // Head
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
  [5, 11], [6, 12], [11, 12], // Torso
  [11, 13], [13, 15], [12, 14], [14, 16] // Legs
] as const;
