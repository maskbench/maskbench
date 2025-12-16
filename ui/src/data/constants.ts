import { Estimator, Metric, EstimatorId, MetricId } from '../types';

export const ESTIMATORS: Record<EstimatorId, Estimator> = {
  'yolo-nano': {
    id: 'yolo-nano',
    name: 'YOLO11-Pose Nano',
    shortName: 'YOLO-n',
    description: 'Fastest, lowest accuracy',
    color: '#000000',
    category: 'yolo'
  },
  'yolo-small': {
    id: 'yolo-small',
    name: 'YOLO11-Pose Small',
    shortName: 'YOLO-s',
    description: 'Fast, good accuracy',
    color: '#1a1a1a',
    category: 'yolo'
  },
  'yolo-medium': {
    id: 'yolo-medium',
    name: 'YOLO11-Pose Medium',
    shortName: 'YOLO-m',
    description: 'Balanced speed/accuracy',
    color: '#333333',
    category: 'yolo'
  },
  'yolo-large': {
    id: 'yolo-large',
    name: 'YOLO11-Pose Large',
    shortName: 'YOLO-l',
    description: 'High accuracy',
    color: '#4d4d4d',
    category: 'yolo'
  },
  'yolo-xlarge': {
    id: 'yolo-xlarge',
    name: 'YOLO11-Pose XLarge',
    shortName: 'YOLO-x',
    description: 'Highest accuracy, slowest',
    color: '#666666',
    category: 'yolo'
  },
  'mediapipe-lite': {
    id: 'mediapipe-lite',
    name: 'MediaPipe Pose Lite',
    shortName: 'MP-lite',
    description: 'Fastest MediaPipe variant',
    color: '#2196f3',
    category: 'mediapipe'
  },
  'mediapipe-full': {
    id: 'mediapipe-full',
    name: 'MediaPipe Pose Full',
    shortName: 'MP-full',
    description: 'Balanced MediaPipe variant',
    color: '#1976d2',
    category: 'mediapipe'
  },
  'mediapipe-heavy': {
    id: 'mediapipe-heavy',
    name: 'MediaPipe Pose Heavy',
    shortName: 'MP-heavy',
    description: 'Most accurate MediaPipe',
    color: '#1565c0',
    category: 'mediapipe'
  },
  'openpose-body25': {
    id: 'openpose-body25',
    name: 'OpenPose BODY_25',
    shortName: 'OP-25',
    description: '25-keypoint body model',
    color: '#ff9800',
    category: 'openpose'
  },
  'openpose-body25b': {
    id: 'openpose-body25b',
    name: 'OpenPose BODY_25B',
    shortName: 'OP-25B',
    description: 'Improved 25-keypoint model',
    color: '#f57c00',
    category: 'openpose'
  },
  'maskanyone-api-mp': {
    id: 'maskanyone-api-mp',
    name: 'MaskAnyone API (MediaPipe)',
    shortName: 'MA-API-MP',
    description: 'Automated pipeline with MediaPipe',
    color: '#9c27b0',
    category: 'maskanyone'
  },
  'maskanyone-api-op': {
    id: 'maskanyone-api-op',
    name: 'MaskAnyone API (OpenPose)',
    shortName: 'MA-API-OP',
    description: 'Automated pipeline with OpenPose',
    color: '#7b1fa2',
    category: 'maskanyone'
  },
  'maskanyone-ui-mp': {
    id: 'maskanyone-ui-mp',
    name: 'MaskAnyone UI (MediaPipe)',
    shortName: 'MA-UI-MP',
    description: 'Human-in-the-loop with MediaPipe',
    color: '#e91e63',
    category: 'maskanyone'
  },
  'maskanyone-ui-op': {
    id: 'maskanyone-ui-op',
    name: 'MaskAnyone UI (OpenPose)',
    shortName: 'MA-UI-OP',
    description: 'Human-in-the-loop with OpenPose',
    color: '#c2185b',
    category: 'maskanyone'
  }
};

export const METRICS: Record<MetricId, Metric> = {
  velocity: {
    id: 'velocity',
    name: 'Velocity',
    description: 'Frame-to-frame keypoint movement',
    requiresGroundTruth: false,
    lowerIsBetter: true,
    unit: 'px/frame'
  },
  acceleration: {
    id: 'acceleration',
    name: 'Acceleration',
    description: 'Rate of velocity change',
    requiresGroundTruth: false,
    lowerIsBetter: true,
    unit: 'px/frame²'
  },
  jerk: {
    id: 'jerk',
    name: 'Jerk',
    description: 'Rate of acceleration change',
    requiresGroundTruth: false,
    lowerIsBetter: true,
    unit: 'px/frame³'
  },
  pck: {
    id: 'pck',
    name: 'PCK',
    description: 'Percentage of Correct Keypoints',
    requiresGroundTruth: true,
    lowerIsBetter: false,
    unit: '%'
  },
  rmse: {
    id: 'rmse',
    name: 'RMSE',
    description: 'Root Mean Square Error',
    requiresGroundTruth: true,
    lowerIsBetter: true,
    unit: 'px'
  },
  euclidean: {
    id: 'euclidean',
    name: 'Euclidean Distance',
    description: 'Average distance from ground truth',
    requiresGroundTruth: true,
    lowerIsBetter: true,
    unit: 'px'
  }
};

// Default estimators for quick start
export const DEFAULT_ESTIMATORS: EstimatorId[] = [
  'yolo-medium',
  'mediapipe-full',
  'openpose-body25'
];

// Default metrics for kinematic analysis
export const DEFAULT_METRICS: MetricId[] = [
  'velocity',
  'acceleration',
  'jerk'
];

// Estimator categories for grouping in UI
export const ESTIMATOR_CATEGORIES = {
  yolo: {
    name: 'YOLO11-Pose',
    description: 'Fast, robust detection',
    color: '#000000'
  },
  mediapipe: {
    name: 'MediaPipe Pose',
    description: 'CPU-friendly, single-person',
    color: '#2196f3'
  },
  openpose: {
    name: 'OpenPose',
    description: 'Multi-person support',
    color: '#ff9800'
  },
  maskanyone: {
    name: 'MaskAnyone',
    description: 'Smoothest output',
    color: '#9c27b0'
  }
};
