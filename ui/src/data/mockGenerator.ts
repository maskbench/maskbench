import {
  Experiment,
  ExperimentStatus,
  VideoFile,
  DashboardStats,
  WeeklyActivity,
  EstimatorId,
  MetricId,
  PoseResult,
  FramePose,
  PersonPose,
  Keypoint,
  COCO_KEYPOINTS,
  MetricResults
} from '../types';
import { ESTIMATORS, DEFAULT_ESTIMATORS, DEFAULT_METRICS } from './constants';

// Utility functions
const randomId = () => Math.random().toString(36).substring(2, 10);
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
const randomFloat = (min: number, max: number, decimals = 2) =>
  parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const randomDate = (daysBack: number) => {
  const date = new Date();
  date.setDate(date.getDate() - randomInt(0, daysBack));
  date.setHours(randomInt(8, 20), randomInt(0, 59), randomInt(0, 59));
  return date;
};

// Sample video names
const VIDEO_NAMES = [
  'TED_Talk_Gesture_01.mp4',
  'TED_Talk_Gesture_02.mp4',
  'classroom_recording_01.mp4',
  'interview_session_01.mp4',
  'ted_kid_education.mp4',
  'let_curiosity_lead.mp4',
  'conversation1_t3-cam08.mp4',
  'interactive1_t1-cam06.mp4',
  'monologue_male_t2-cam04.mp4',
  'sports_motion_01.mp4',
  'dance_performance_01.mp4',
  'sign_language_01.mp4'
];

const EXPERIMENT_NAMES = [
  'TED Talks Evaluation',
  'Masked Video Comparison',
  'Tragic Talkers Ground Truth',
  'Quick Test - Single Video',
  'Classroom Recording Batch',
  'Clinical Gait Analysis',
  'Interview De-identification',
  'Sports Motion Capture',
  'Gesture Study Pilot',
  'Multi-person Interaction',
  'Occlusion Test Set',
  'Sign Language Analysis'
];

// Generate mock video file
export function generateMockVideo(name?: string): VideoFile {
  const videoName = name || randomChoice(VIDEO_NAMES);
  const fps = randomChoice([24, 25, 30, 60]);
  const duration = randomInt(10, 300);

  return {
    id: randomId(),
    name: videoName,
    path: `/datasets/videos/${videoName}`,
    duration,
    fps,
    width: randomChoice([1280, 1920, 3840]),
    height: randomChoice([720, 1080, 2160]),
    size: randomInt(10, 500) * 1024 * 1024,
    frameCount: Math.floor(duration * fps)
  };
}

// Generate mock pose for a single frame
export function generateMockPose(frameWidth: number, frameHeight: number, jitter = 5): PersonPose {
  const centerX = frameWidth / 2;
  const centerY = frameHeight / 2;

  // Base pose (standing person)
  const basePose: Record<string, [number, number]> = {
    nose: [centerX, centerY - 150],
    left_eye: [centerX - 15, centerY - 160],
    right_eye: [centerX + 15, centerY - 160],
    left_ear: [centerX - 30, centerY - 155],
    right_ear: [centerX + 30, centerY - 155],
    left_shoulder: [centerX - 60, centerY - 100],
    right_shoulder: [centerX + 60, centerY - 100],
    left_elbow: [centerX - 80, centerY - 30],
    right_elbow: [centerX + 80, centerY - 30],
    left_wrist: [centerX - 90, centerY + 30],
    right_wrist: [centerX + 90, centerY + 30],
    left_hip: [centerX - 40, centerY + 20],
    right_hip: [centerX + 40, centerY + 20],
    left_knee: [centerX - 45, centerY + 100],
    right_knee: [centerX + 45, centerY + 100],
    left_ankle: [centerX - 50, centerY + 180],
    right_ankle: [centerX + 50, centerY + 180]
  };

  const keypoints: Keypoint[] = COCO_KEYPOINTS.map(name => {
    const [baseX, baseY] = basePose[name];
    return {
      name,
      x: baseX + (Math.random() - 0.5) * jitter * 2,
      y: baseY + (Math.random() - 0.5) * jitter * 2,
      confidence: randomFloat(0.7, 1.0, 3)
    };
  });

  return {
    id: 0,
    keypoints,
    boundingBox: {
      x: centerX - 100,
      y: centerY - 180,
      width: 200,
      height: 380
    }
  };
}

// Generate mock pose result for a video
export function generateMockPoseResult(video: VideoFile, estimatorId: EstimatorId): PoseResult {
  // Different estimators have different jitter characteristics
  const jitterMap: Record<string, number> = {
    'mediapipe': 15,
    'openpose': 8,
    'yolo': 6,
    'maskanyone': 3
  };

  const category = ESTIMATORS[estimatorId].category;
  const jitter = jitterMap[category] || 5;

  const frames: FramePose[] = [];
  for (let i = 0; i < Math.min(video.frameCount, 300); i++) {
    frames.push({
      frameIndex: i,
      persons: [generateMockPose(video.width, video.height, jitter)]
    });
  }

  return {
    videoName: video.name,
    fps: video.fps,
    frameCount: video.frameCount,
    frames
  };
}

// Generate mock metric results
export function generateMockMetricResults(
  estimators: EstimatorId[],
  videos: VideoFile[],
  metricId: MetricId
): MetricResults {
  // Base values per estimator category (for kinematic metrics - lower is better)
  const baseValues: Record<string, Record<MetricId, number>> = {
    yolo: { velocity: 1.3, acceleration: 1.4, jerk: 2.4, pck: 92, rmse: 8, euclidean: 7 },
    mediapipe: { velocity: 3.2, acceleration: 4.5, jerk: 7.9, pck: 88, rmse: 12, euclidean: 10 },
    openpose: { velocity: 1.6, acceleration: 2.2, jerk: 3.4, pck: 90, rmse: 10, euclidean: 9 },
    maskanyone: { velocity: 1.1, acceleration: 1.1, jerk: 1.8, pck: 94, rmse: 6, euclidean: 5 }
  };

  const byEstimator: Record<EstimatorId, number> = {} as Record<EstimatorId, number>;
  const byVideo: Record<string, Record<EstimatorId, number>> = {};

  estimators.forEach(estId => {
    const category = ESTIMATORS[estId].category;
    const base = baseValues[category][metricId];
    byEstimator[estId] = base + randomFloat(-0.3, 0.3);
  });

  videos.forEach(video => {
    byVideo[video.name] = {} as Record<EstimatorId, number>;
    estimators.forEach(estId => {
      const category = ESTIMATORS[estId].category;
      const base = baseValues[category][metricId];
      byVideo[video.name][estId] = base + randomFloat(-0.5, 0.5);
    });
  });

  // Per-keypoint results for detailed analysis
  const byKeypoint: Record<string, Record<EstimatorId, number>> = {};
  COCO_KEYPOINTS.forEach(kp => {
    byKeypoint[kp] = {} as Record<EstimatorId, number>;
    // Wrists and ankles have higher values (more movement)
    const multiplier = kp.includes('wrist') || kp.includes('ankle') ? 1.5 :
                       kp.includes('elbow') || kp.includes('knee') ? 1.2 : 1.0;

    estimators.forEach(estId => {
      byKeypoint[kp][estId] = byEstimator[estId] * multiplier + randomFloat(-0.2, 0.2);
    });
  });

  return { byEstimator, byVideo, byKeypoint };
}

// Generate a complete mock experiment
export function generateMockExperiment(
  status: ExperimentStatus = 'completed',
  overrides?: Partial<Experiment>
): Experiment {
  const videoCount = randomInt(1, 15);
  const videos = Array.from({ length: videoCount }, () => generateMockVideo());
  const estimators = overrides?.config?.estimators ||
    (randomChoice([DEFAULT_ESTIMATORS, ['yolo-medium'], ['yolo-medium', 'mediapipe-full']]));
  const metrics = overrides?.config?.metrics || DEFAULT_METRICS;

  const createdAt = randomDate(30);
  const completedAt = status === 'completed' ? new Date(createdAt.getTime() + randomInt(60, 3600) * 1000) : undefined;

  const experiment: Experiment = {
    id: randomId(),
    name: overrides?.name || randomChoice(EXPERIMENT_NAMES),
    description: `${videoCount} videos, ${estimators.length} estimators`,
    status,
    progress: status === 'completed' ? 100 : status === 'processing' ? randomInt(10, 90) : 0,
    createdAt,
    updatedAt: completedAt || createdAt,
    completedAt,
    config: {
      datasetName: 'custom',
      videos,
      estimators: estimators as EstimatorId[],
      metrics: metrics as MetricId[],
      confidenceThreshold: 0.3,
      saveRenderings: true,
      savePoses: true
    },
    ...overrides
  };

  // Generate results for completed experiments
  if (status === 'completed') {
    const metricResults: Record<MetricId, MetricResults> = {} as Record<MetricId, MetricResults>;
    metrics.forEach(m => {
      metricResults[m as MetricId] = generateMockMetricResults(
        estimators as EstimatorId[],
        videos,
        m as MetricId
      );
    });

    const inferenceTime: Record<EstimatorId, Record<string, number>> = {} as Record<EstimatorId, Record<string, number>>;
    (estimators as EstimatorId[]).forEach(est => {
      inferenceTime[est] = {};
      videos.forEach(v => {
        // Different estimators have different speeds
        const baseTime = ESTIMATORS[est].category === 'mediapipe' ? 0.3 :
                        ESTIMATORS[est].category === 'yolo' ? 0.8 :
                        ESTIMATORS[est].category === 'openpose' ? 1.2 : 2.0;
        inferenceTime[est][v.name] = v.duration * baseTime + randomFloat(-5, 10);
      });
    });

    experiment.results = {
      inferenceTime,
      metrics: metricResults,
      poses: {} as Record<EstimatorId, Record<string, PoseResult>> // Poses generated on demand
    };
  }

  return experiment;
}

// Generate dashboard stats
export function generateMockDashboardStats(): DashboardStats {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const weeklyActivity: WeeklyActivity[] = days.map(day => ({
    day,
    experiments: randomInt(0, 4),
    videos: randomInt(2, 15)
  }));

  const recentExperiments = [
    generateMockExperiment('completed', { name: 'TED Talks Evaluation' }),
    generateMockExperiment('processing', { name: 'Masked Video Comparison' }),
    generateMockExperiment('draft', { name: 'Tragic Talkers Ground Truth' })
  ];

  return {
    totalExperiments: randomInt(10, 25),
    completedExperiments: randomInt(8, 20),
    totalVideosProcessed: randomInt(40, 100),
    availableEstimators: Object.keys(ESTIMATORS).length,
    avgProcessingTime: randomFloat(1.5, 3.5, 1),
    weeklyActivity,
    recentExperiments
  };
}

// Generate a list of experiments for history page
export function generateMockExperimentList(count = 12): Experiment[] {
  const statuses: ExperimentStatus[] = ['completed', 'completed', 'completed', 'completed',
                                         'processing', 'failed', 'draft'];

  return Array.from({ length: count }, (_, i) => {
    const status = i < 2 ? statuses[i] : randomChoice(statuses);
    return generateMockExperiment(status, {
      name: EXPERIMENT_NAMES[i % EXPERIMENT_NAMES.length]
    });
  }).sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
}
