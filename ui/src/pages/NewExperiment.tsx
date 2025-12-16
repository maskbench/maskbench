import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Add,
  TrashCan,
  Upload,
  Play,
  Checkmark,
  Close,
  ChemistryReference,
  Bot,
  Ruler,
  Settings,
  Terminal,
  Copy
} from '@carbon/icons-react';
import {
  Button,
  TextInput,
  Select,
  SelectItem,
  Checkbox,
  Slider,
  Modal,
  ProgressBar,
  InlineNotification
} from '@carbon/react';

import { useExperimentStore } from '../stores/experimentStore';
import { ESTIMATORS, METRICS, ESTIMATOR_CATEGORIES } from '../data/constants';
import { generateMockVideo } from '../data/mockGenerator';
import { EstimatorId, MetricId, VideoFile } from '../types';
import './NewExperiment.scss';

const DATASETS = [
  { id: 'custom', label: 'Upload Videos' },
  { id: 'ted-kid', label: 'TED Kid Video (Demo)' },
  { id: 'tragic-talkers', label: 'Tragic Talkers (with GT)' }
];

export function NewExperiment() {
  const navigate = useNavigate();
  const { newExperimentConfig, updateNewExperimentConfig, createExperiment, simulateExperimentRun } = useExperimentStore();

  const [experimentName, setExperimentName] = useState('TED_Gesture_Study');
  const [datasetId, setDatasetId] = useState('custom');
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [selectedEstimators, setSelectedEstimators] = useState<EstimatorId[]>([
    'yolo-medium', 'mediapipe-full', 'openpose-body25'
  ]);
  const [selectedMetrics, setSelectedMetrics] = useState<MetricId[]>([
    'velocity', 'acceleration', 'jerk'
  ]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(30);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');

  // Handle drag & drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(f =>
      f.type.startsWith('video/') || f.name.match(/\.(mp4|avi|mov|mkv)$/i)
    );
    if (files.length > 0) {
      const newVideos = files.map(f => generateMockVideo(f.name));
      setVideos(prev => [...prev, ...newVideos]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const newVideos = files.map(f => generateMockVideo(f.name));
    setVideos(prev => [...prev, ...newVideos]);
  };

  const removeVideo = (id: string) => {
    setVideos(prev => prev.filter(v => v.id !== id));
  };

  const toggleEstimator = (id: EstimatorId) => {
    setSelectedEstimators(prev =>
      prev.includes(id) ? prev.filter(e => e !== id) : [...prev, id]
    );
  };

  const toggleMetric = (id: MetricId) => {
    setSelectedMetrics(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  // Handle dataset change (auto-populate demo videos)
  const handleDatasetChange = (id: string) => {
    setDatasetId(id);
    if (id === 'ted-kid') {
      setVideos([generateMockVideo('ted_kid_education_for_all.mp4')]);
    } else if (id === 'tragic-talkers') {
      setVideos([
        generateMockVideo('conversation1_t3-cam08.mp4'),
        generateMockVideo('interactive1_t1-cam06.mp4'),
        generateMockVideo('monologue_male_t2-cam04.mp4')
      ]);
      // Enable ground-truth metrics
      setSelectedMetrics(prev => [...new Set([...prev, 'pck' as MetricId, 'rmse' as MetricId])]);
    } else {
      setVideos([]);
    }
  };

  // Simulate experiment run
  const runExperiment = async () => {
    if (videos.length === 0) return;

    setIsRunning(true);
    setProgress(0);

    const statuses = [
      'Initializing pose estimators...',
      'Loading model weights...',
      `Processing video 1/${videos.length}...`,
      'Running pose estimation...',
      'Computing metrics...',
      'Generating visualizations...',
      'Finalizing results...'
    ];

    let currentProgress = 0;
    let statusIndex = 0;

    const interval = setInterval(() => {
      currentProgress += Math.random() * 8 + 2;

      if (currentProgress >= 100) {
        currentProgress = 100;
        clearInterval(interval);

        setTimeout(() => {
          setIsRunning(false);
          navigate('/results/demo');
        }, 500);
      }

      setProgress(currentProgress);

      const newStatusIndex = Math.min(
        Math.floor((currentProgress / 100) * statuses.length),
        statuses.length - 1
      );
      if (newStatusIndex !== statusIndex) {
        statusIndex = newStatusIndex;
        setProgressStatus(statuses[statusIndex]);
      }
    }, 300);
  };

  const cliCommand = `docker compose up -e MASKBENCH_CONFIG_FILE=config/${experimentName.toLowerCase().replace(/\s+/g, '-')}.yml`;

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const formatSize = (bytes: number) => {
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  };

  const estimatedTime = videos.length * selectedEstimators.length * 2;

  return (
    <div className="new-experiment">
      <div className="new-experiment__layout">
        {/* Sidebar */}
        <aside className="new-experiment__sidebar">
          <div className="sidebar-section">
            <h4 className="sidebar-section__title">
              <ChemistryReference size={14} /> Experiment Name
            </h4>
            <TextInput
              id="experiment-name"
              labelText=""
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
            />
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">
              <Upload size={14} /> Dataset Source
            </h4>
            <Select
              id="dataset-select"
              labelText=""
              value={datasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
            >
              {DATASETS.map(ds => (
                <SelectItem key={ds.id} value={ds.id} text={ds.label} />
              ))}
            </Select>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">
              <Bot size={14} /> Pose Estimators
              <span className="count-badge">{selectedEstimators.length}</span>
            </h4>
            <div className="estimator-list">
              {Object.entries(ESTIMATOR_CATEGORIES).map(([category, info]) => {
                const categoryEstimators = Object.values(ESTIMATORS).filter(
                  e => e.category === category
                );
                const mainEstimator = categoryEstimators[Math.floor(categoryEstimators.length / 2)];

                return (
                  <div
                    key={category}
                    className={`estimator-card ${selectedEstimators.includes(mainEstimator.id) ? 'estimator-card--selected' : ''}`}
                    onClick={() => toggleEstimator(mainEstimator.id)}
                  >
                    <div
                      className="estimator-card__dot"
                      style={{ background: info.color }}
                    />
                    <div className="estimator-card__info">
                      <span className="estimator-card__name">{info.name}</span>
                      <span className="estimator-card__desc">{info.description}</span>
                    </div>
                    {selectedEstimators.includes(mainEstimator.id) && (
                      <Checkmark size={16} />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">
              <Ruler size={14} /> Metrics
            </h4>
            <div className="metrics-grid">
              {Object.values(METRICS).map(metric => (
                <div
                  key={metric.id}
                  className={`metric-chip ${selectedMetrics.includes(metric.id) ? 'metric-chip--selected' : ''} ${metric.requiresGroundTruth && datasetId !== 'tragic-talkers' ? 'metric-chip--disabled' : ''}`}
                  onClick={() => {
                    if (!metric.requiresGroundTruth || datasetId === 'tragic-talkers') {
                      toggleMetric(metric.id);
                    }
                  }}
                >
                  {selectedMetrics.includes(metric.id) && <Checkmark size={12} />}
                  {metric.name}
                </div>
              ))}
            </div>
            {datasetId !== 'tragic-talkers' && (
              <p className="sidebar-section__note">
                PCK/RMSE require ground truth data
              </p>
            )}
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">
              <Settings size={14} /> Advanced Options
            </h4>
            <details className="advanced-options">
              <summary>Confidence thresholds, output format...</summary>
              <div className="advanced-options__content">
                <Slider
                  labelText="Confidence Threshold"
                  value={confidenceThreshold}
                  min={0}
                  max={100}
                  step={5}
                  onChange={({ value }) => setConfidenceThreshold(value)}
                />
                <Checkbox
                  id="save-coco"
                  labelText="Save in COCO format"
                  defaultChecked
                />
              </div>
            </details>
          </div>

          <Button
            kind="primary"
            renderIcon={Play}
            onClick={runExperiment}
            disabled={videos.length === 0 || isRunning}
            style={{ width: '100%' }}
          >
            Run Experiment
          </Button>
        </aside>

        {/* Main Content */}
        <main className="new-experiment__main">
          {/* Step Indicator */}
          <div className="step-indicator">
            <div className="step step--completed">
              <div className="step__dot"><Checkmark size={14} /></div>
              <span className="step__label">Configure</span>
            </div>
            <div className="step step--active">
              <div className="step__dot">2</div>
              <span className="step__label">Upload</span>
            </div>
            <div className="step">
              <div className="step__dot">3</div>
              <span className="step__label">Review</span>
            </div>
            <div className="step">
              <div className="step__dot">4</div>
              <span className="step__label">Run</span>
            </div>
          </div>

          {/* Upload Section */}
          <div className="mb-card upload-section">
            <div className="upload-section__header">
              <div>
                <h1 className="upload-section__title">
                  Upload <span>Videos</span>
                </h1>
                <p className="upload-section__subtitle">
                  Add videos to your dataset for benchmarking
                </p>
              </div>
            </div>

            <div
              className={`drop-zone ${videos.length > 0 ? 'drop-zone--has-files' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById('file-input')?.click()}
            >
              <input
                id="file-input"
                type="file"
                multiple
                accept="video/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <div className="drop-zone__icon">
                <Upload size={24} />
              </div>
              <h3 className="drop-zone__title">Drag and drop videos here</h3>
              <p className="drop-zone__hint">
                Supports MP4, AVI, MOV, MKV - No file size limit
              </p>
              <Button kind="secondary" size="sm" renderIcon={Add}>
                Select Files
              </Button>
            </div>

            {videos.length > 0 && (
              <div className="video-list">
                <div className="video-list__header">
                  <h4>Uploaded Videos ({videos.length})</h4>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={TrashCan}
                    onClick={() => setVideos([])}
                  >
                    Clear All
                  </Button>
                </div>
                <div className="video-grid">
                  {videos.map(video => (
                    <div key={video.id} className="video-card">
                      <button
                        className="video-card__remove"
                        onClick={() => removeVideo(video.id)}
                      >
                        <Close size={12} />
                      </button>
                      <div className="video-card__thumb">
                        <span className="video-card__duration">
                          {formatDuration(video.duration)}
                        </span>
                      </div>
                      <div className="video-card__info">
                        <span className="video-card__name" title={video.name}>
                          {video.name}
                        </span>
                        <span className="video-card__size">
                          {formatSize(video.size)}
                        </span>
                      </div>
                    </div>
                  ))}
                  <div
                    className="video-card video-card--add"
                    onClick={() => document.getElementById('file-input')?.click()}
                  >
                    <Add size={20} />
                    <span>Add more</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Config Summary */}
          <div className="config-summary">
            <h4>Experiment Summary</h4>
            <div className="config-summary__row">
              <span className="config-summary__label">Name</span>
              <span className="config-summary__value">{experimentName}</span>
            </div>
            <div className="config-summary__row">
              <span className="config-summary__label">Videos</span>
              <span className="config-summary__value">{videos.length} videos</span>
            </div>
            <div className="config-summary__row">
              <span className="config-summary__label">Estimators</span>
              <span className="config-summary__value">{selectedEstimators.length} selected</span>
            </div>
            <div className="config-summary__row">
              <span className="config-summary__label">Metrics</span>
              <span className="config-summary__value">
                {selectedMetrics.slice(0, 3).join(', ')}
                {selectedMetrics.length > 3 && '...'}
              </span>
            </div>
            <div className="config-summary__row">
              <span className="config-summary__label">Est. Time</span>
              <span className="config-summary__value">
                {videos.length > 0 ? `~${estimatedTime} min` : '-'}
              </span>
            </div>
          </div>

          {/* CLI Panel */}
          <div className="cli-panel">
            <div className="cli-panel__header">
              <span><Terminal size={14} /> Equivalent CLI Command</span>
              <Button
                kind="ghost"
                size="sm"
                renderIcon={Copy}
                onClick={() => navigator.clipboard.writeText(cliCommand)}
              >
                Copy
              </Button>
            </div>
            <div className="cli-panel__content">
              <code>{cliCommand}</code>
            </div>
          </div>
        </main>
      </div>

      {/* Progress Modal */}
      <Modal
        open={isRunning}
        modalHeading="Running Experiment"
        passiveModal
        preventCloseOnClickOutside
        size="sm"
      >
        <div className="progress-modal">
          <div className="progress-modal__icon">
            <Settings size={32} className="mb-spin" />
          </div>
          <p className="progress-modal__status">{progressStatus}</p>
          <ProgressBar
            label={`${Math.round(progress)}% complete`}
            value={progress}
            max={100}
          />
          <Button
            kind="ghost"
            onClick={() => setIsRunning(false)}
            style={{ marginTop: '1rem' }}
          >
            Cancel
          </Button>
        </div>
      </Modal>
    </div>
  );
}
