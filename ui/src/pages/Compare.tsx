import { useState } from 'react';
import { Download, Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from '@carbon/icons-react';
import { Button, Select, SelectItem, Slider } from '@carbon/react';
import { ESTIMATORS } from '../data/constants';
import { EstimatorId, COCO_KEYPOINTS, COCO_SKELETON } from '../types';
import './Compare.scss';

export function Compare() {
  const [leftEstimator, setLeftEstimator] = useState<EstimatorId>('yolo-medium');
  const [rightEstimator, setRightEstimator] = useState<EstimatorId>('mediapipe-full');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const totalFrames = 4620;
  const fps = 30;

  const formatTime = (frame: number) => {
    const totalSeconds = frame / fps;
    const mins = Math.floor(totalSeconds / 60);
    const secs = (totalSeconds % 60).toFixed(2).padStart(5, '0');
    return `${mins}:${secs}`;
  };

  // Generate random pose for visualization
  const generatePose = (jitter: number) => {
    const basePose = [
      [320, 80], [280, 140], [360, 140], [240, 200], [400, 200],
      [220, 260], [420, 260], [290, 240], [350, 240], [280, 320],
      [360, 320], [270, 400], [370, 400], [305, 75], [335, 75], [295, 70], [345, 70]
    ];

    return basePose.map(([x, y]) => [
      x + (Math.random() - 0.5) * jitter,
      y + (Math.random() - 0.5) * jitter
    ]);
  };

  const leftPose = generatePose(ESTIMATORS[leftEstimator].category === 'mediapipe' ? 15 : 5);
  const rightPose = generatePose(ESTIMATORS[rightEstimator].category === 'mediapipe' ? 15 : 5);

  const renderSkeleton = (pose: number[][], color: string) => (
    <svg viewBox="0 0 640 480" className="skeleton-svg">
      {/* Bones */}
      {COCO_SKELETON.map(([i, j], idx) => (
        <line
          key={`bone-${idx}`}
          x1={pose[i]?.[0]}
          y1={pose[i]?.[1]}
          x2={pose[j]?.[0]}
          y2={pose[j]?.[1]}
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
        />
      ))}
      {/* Joints */}
      {pose.map(([x, y], idx) => (
        <circle
          key={`joint-${idx}`}
          cx={x}
          cy={y}
          r="5"
          fill="white"
          stroke={color}
          strokeWidth="2"
        />
      ))}
    </svg>
  );

  // Mock metrics
  const metrics = {
    velocity: { left: 1.8, right: 4.2 },
    acceleration: { left: 1.2, right: 3.8 },
    jerk: { left: 2.1, right: 6.9 },
    keypoints: { left: '17/17', right: '15/17' }
  };

  return (
    <div className="compare">
      {/* Header */}
      <div className="compare__header mb-fade-in">
        <div>
          <h1 className="compare__title">
            Side-by-Side <span>Comparison</span>
          </h1>
          <p className="compare__subtitle">Compare pose estimator outputs frame by frame</p>
        </div>
        <div className="compare__actions">
          <Select
            id="video-select"
            labelText=""
            size="sm"
            defaultValue="ted1"
          >
            <SelectItem value="ted1" text="TED_Talk_Gesture_01.mp4" />
            <SelectItem value="ted2" text="TED_Talk_Gesture_02.mp4" />
            <SelectItem value="ted3" text="TED_Talk_Gesture_03.mp4" />
          </Select>
          <Button kind="secondary" size="sm" renderIcon={Download}>
            Export Comparison
          </Button>
        </div>
      </div>

      {/* Video Panels */}
      <div className="compare__panels mb-fade-in">
        {/* Left Panel */}
        <div className="video-panel">
          <div className="video-panel__header">
            <div
              className="color-dot"
              style={{ background: ESTIMATORS[leftEstimator].color }}
            />
            <Select
              id="left-estimator"
              labelText=""
              size="sm"
              value={leftEstimator}
              onChange={(e) => setLeftEstimator(e.target.value as EstimatorId)}
            >
              {Object.values(ESTIMATORS).slice(0, 8).map(est => (
                <SelectItem key={est.id} value={est.id} text={est.name} />
              ))}
            </Select>
          </div>
          <div className="video-canvas">
            {renderSkeleton(leftPose, ESTIMATORS[leftEstimator].color)}
            <div className="video-label">{ESTIMATORS[leftEstimator].shortName}</div>
            <div className="video-fps">{fps} FPS</div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="video-panel">
          <div className="video-panel__header">
            <div
              className="color-dot"
              style={{ background: ESTIMATORS[rightEstimator].color }}
            />
            <Select
              id="right-estimator"
              labelText=""
              size="sm"
              value={rightEstimator}
              onChange={(e) => setRightEstimator(e.target.value as EstimatorId)}
            >
              {Object.values(ESTIMATORS).slice(0, 8).map(est => (
                <SelectItem key={est.id} value={est.id} text={est.name} />
              ))}
            </Select>
          </div>
          <div className="video-canvas">
            {renderSkeleton(rightPose, ESTIMATORS[rightEstimator].color)}
            <div className="video-label">{ESTIMATORS[rightEstimator].shortName}</div>
            <div className="video-fps">{fps} FPS</div>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="timeline mb-fade-in">
        <div className="timeline__controls">
          <div className="play-controls">
            <Button kind="ghost" size="sm" hasIconOnly iconDescription="Skip Back" renderIcon={SkipBack} />
            <Button kind="ghost" size="sm" hasIconOnly iconDescription="Previous Frame" renderIcon={ChevronLeft} />
            <Button
              kind="primary"
              size="sm"
              hasIconOnly
              iconDescription={isPlaying ? 'Pause' : 'Play'}
              renderIcon={isPlaying ? Pause : Play}
              onClick={() => setIsPlaying(!isPlaying)}
            />
            <Button kind="ghost" size="sm" hasIconOnly iconDescription="Next Frame" renderIcon={ChevronRight} />
            <Button kind="ghost" size="sm" hasIconOnly iconDescription="Skip Forward" renderIcon={SkipForward} />
          </div>

          <div className="time-display">
            <span>{formatTime(currentFrame)}</span> / <span>2:34.00</span>
          </div>

          <div className="frame-display">
            Frame: <strong>{currentFrame + 1}</strong> / {totalFrames}
          </div>

          <div className="speed-controls">
            <span>Speed:</span>
            {[0.25, 0.5, 1, 2].map(speed => (
              <button
                key={speed}
                className={`speed-btn ${playbackSpeed === speed ? 'speed-btn--active' : ''}`}
                onClick={() => setPlaybackSpeed(speed)}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>

        <div className="timeline__track">
          <Slider
            labelText=""
            hideTextInput
            min={0}
            max={totalFrames}
            value={currentFrame}
            onChange={({ value }) => setCurrentFrame(value)}
          />
        </div>
      </div>

      {/* Metrics */}
      <div className="compare__metrics mb-fade-in">
        {Object.entries(metrics).map(([key, { left, right }]) => {
          const leftNum = typeof left === 'string' ? parseInt(left) : left;
          const rightNum = typeof right === 'string' ? parseInt(right) : right;
          const leftBetter = leftNum < rightNum;
          const diff = Math.abs(((leftNum - rightNum) / rightNum) * 100).toFixed(0);

          return (
            <div key={key} className="metric-card">
              <h4>{key.charAt(0).toUpperCase() + key.slice(1)}</h4>
              <div className="metric-card__values">
                <div className={`metric-value ${leftBetter ? 'metric-value--better' : 'metric-value--worse'}`}>
                  <span className="metric-value__number">{left}</span>
                  <span className="metric-value__label">{ESTIMATORS[leftEstimator].shortName}</span>
                </div>
                <div className={`metric-value ${!leftBetter ? 'metric-value--better' : 'metric-value--worse'}`}>
                  <span className="metric-value__number">{right}</span>
                  <span className="metric-value__label">{ESTIMATORS[rightEstimator].shortName}</span>
                </div>
              </div>
              {key !== 'keypoints' && (
                <div className={`metric-card__diff ${leftBetter ? 'positive' : 'negative'}`}>
                  {leftBetter ? ESTIMATORS[leftEstimator].shortName : ESTIMATORS[rightEstimator].shortName} {diff}% lower
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
