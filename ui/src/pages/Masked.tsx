import { useState } from 'react';
import { Add, Play, Download, Information, Checkmark, Close } from '@carbon/icons-react';
import { Button, Select, SelectItem } from '@carbon/react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

import { ESTIMATORS } from '../data/constants';
import './Masked.scss';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const MASKING_STRATEGIES = [
  { id: 'blur', name: 'Blurring', description: 'Gaussian blur', icon: '~', pck: 86 },
  { id: 'contour', name: 'Contours', description: 'Edge outlines only', icon: '/', pck: 44 },
  { id: 'pixel', name: 'Pixelation', description: 'Mosaic blocks', icon: '#', pck: 23 },
  { id: 'solid', name: 'Solid Fill', description: 'Uniform color', icon: '[]', pck: 11 }
];

const PCK_DATA = {
  'yolo-medium': { blur: 95, pixel: 9, contour: 93, solid: 32 },
  'mediapipe-full': { blur: 95, pixel: 81, contour: 56, solid: 34 },
  'openpose-body25': { blur: 88, pixel: 10, contour: 62, solid: 1 },
  'maskanyone-ui-mp': { blur: 95, pixel: 63, contour: 0, solid: 7 }
};

const estimatorList = ['yolo-medium', 'mediapipe-full', 'openpose-body25', 'maskanyone-ui-mp'] as const;
const strategies = ['blur', 'pixel', 'contour', 'solid'] as const;

export function Masked() {
  const [selectedStrategy, setSelectedStrategy] = useState('blur');

  const getHeatmapColor = (val: number) => {
    if (val >= 80) return '#24a148';
    if (val >= 60) return '#7cb342';
    if (val >= 40) return '#fdd835';
    if (val >= 20) return '#ff9800';
    return '#da1e28';
  };

  const getPckBadgeClass = (pck: number) => {
    if (pck >= 70) return 'pck-badge--high';
    if (pck >= 40) return 'pck-badge--medium';
    return 'pck-badge--low';
  };

  // Chart data
  const avgByStrategy = strategies.map(s => {
    const sum = estimatorList.reduce((acc, e) => acc + PCK_DATA[e][s], 0);
    return Math.round(sum / estimatorList.length);
  });

  const chartData = {
    labels: ['Blurring', 'Pixelation', 'Contours', 'Solid Fill'],
    datasets: [{
      label: 'Average PCK %',
      data: avgByStrategy,
      backgroundColor: ['#24a148', '#ff9800', '#fdd835', '#da1e28'],
      borderRadius: 4
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: { beginAtZero: true, max: 100, ticks: { callback: (v: string | number) => `${v}%` } },
      x: { grid: { display: false } }
    }
  };

  return (
    <div className="masked">
      <div className="masked__layout">
        {/* Sidebar */}
        <aside className="masked__sidebar">
          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Experiment</h4>
            <div className="experiment-card" style={{ borderLeftColor: '#8a3ffc' }}>
              <p className="experiment-card__name">Masked Video Evaluation</p>
              <p className="experiment-card__meta">3 videos - 4 strategies</p>
              <p className="experiment-card__date">Dec 14, 2025</p>
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Masking Strategies</h4>
            <div className="strategy-list">
              {MASKING_STRATEGIES.map(strategy => (
                <div
                  key={strategy.id}
                  className={`strategy-card ${selectedStrategy === strategy.id ? 'strategy-card--selected' : ''}`}
                  onClick={() => setSelectedStrategy(strategy.id)}
                >
                  <div className="strategy-card__icon">{strategy.icon}</div>
                  <div className="strategy-card__info">
                    <span className="strategy-card__name">{strategy.name}</span>
                    <span className="strategy-card__desc">{strategy.description}</span>
                  </div>
                  <span className={`pck-badge ${getPckBadgeClass(strategy.pck)}`}>
                    {strategy.pck}% PCK
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Test Videos</h4>
            <div className="video-list-mini">
              {['ted_kid_video.mp4', 'let_curiosity_lead.mp4', 'interactive1_t1.mp4'].map(name => (
                <div key={name} className="video-item-mini">
                  <span>{name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <Button kind="secondary" size="sm" renderIcon={Download} style={{ width: '100%', marginBottom: 8 }}>
              Export Results
            </Button>
            <Button kind="ghost" size="sm" style={{ width: '100%' }}>
              Download Plots
            </Button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="masked__main">
          {/* Header */}
          <div className="masked__header mb-fade-in">
            <div>
              <h1 className="masked__title">
                Masked Video <span>Evaluation</span>
              </h1>
              <p className="masked__subtitle">
                Evaluate pose estimation performance on privacy-masked videos
              </p>
            </div>
            <div className="masked__actions">
              <Button kind="secondary" size="sm" renderIcon={Add}>Add Strategy</Button>
              <Button kind="primary" size="sm" renderIcon={Play}>Run Evaluation</Button>
            </div>
          </div>

          {/* Info Banner */}
          <div className="info-banner mb-fade-in">
            <Information size={20} />
            <div>
              <strong>How this works:</strong> We compare pose estimation on masked videos
              against the original (unmasked) baseline. PCK measures what percentage of
              keypoints are detected correctly relative to the original.
            </div>
          </div>

          {/* Visual Comparison */}
          <div className="mb-card mb-mb-5 mb-fade-in">
            <div className="mb-card__header">
              <h3 className="mb-card__title">Visual Comparison</h3>
              <span className="mb-text-xs mb-text-gray">Click to select masking strategy</span>
            </div>
            <div className="mb-card__body">
              <div className="masking-grid">
                {['Original', ...MASKING_STRATEGIES.map(s => s.name)].map((name, i) => (
                  <div
                    key={name}
                    className={`masking-preview ${i === 0 ? 'masking-preview--active' : ''}`}
                  >
                    <div className="masking-preview__thumb">
                      <svg viewBox="0 0 60 90" className={i > 0 ? `effect-${strategies[i-1]}` : ''}>
                        <ellipse cx="30" cy="12" rx="10" ry="12" fill={i === 0 ? '#f5d0c5' : i === 4 ? '#000' : '#888'} />
                        <rect x="20" y="24" width="20" height="35" rx="3" fill={i === 0 ? '#4a90d9' : i === 4 ? '#000' : '#666'} />
                        <rect x="22" y="59" width="7" height="30" rx="2" fill={i === 0 ? '#3a3a5a' : i === 4 ? '#000' : '#555'} />
                        <rect x="31" y="59" width="7" height="30" rx="2" fill={i === 0 ? '#3a3a5a' : i === 4 ? '#000' : '#555'} />
                      </svg>
                    </div>
                    <div className="masking-preview__label">{name}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Results Grid */}
          <div className="mb-grid mb-grid--2 mb-mb-5">
            {/* Heatmap */}
            <div className="mb-card mb-fade-in">
              <div className="mb-card__header">
                <h3 className="mb-card__title">PCK by Strategy x Estimator</h3>
                <span className="mb-text-xs mb-text-gray">Higher is better</span>
              </div>
              <div className="mb-card__body">
                <div className="heatmap">
                  <div className="heatmap__header"></div>
                  {strategies.map(s => (
                    <div key={s} className="heatmap__header">
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                    </div>
                  ))}

                  {estimatorList.map(estId => (
                    <>
                      <div key={`${estId}-label`} className="heatmap__row-label">
                        <div className="color-dot" style={{ background: ESTIMATORS[estId].color }} />
                        {ESTIMATORS[estId].shortName}
                      </div>
                      {strategies.map(s => (
                        <div
                          key={`${estId}-${s}`}
                          className="heatmap__cell"
                          style={{
                            background: getHeatmapColor(PCK_DATA[estId][s]),
                            color: PCK_DATA[estId][s] > 50 ? '#fff' : '#000'
                          }}
                        >
                          {PCK_DATA[estId][s]}%
                        </div>
                      ))}
                    </>
                  ))}
                </div>
              </div>
            </div>

            {/* Bar Chart */}
            <div className="mb-card mb-fade-in">
              <div className="mb-card__header">
                <h3 className="mb-card__title">Average PCK by Strategy</h3>
              </div>
              <div className="mb-card__body">
                <div className="chart-container">
                  <Bar data={chartData} options={chartOptions} />
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="mb-grid mb-grid--2 mb-fade-in">
            <div className="recommendation-card recommendation-card--success">
              <div className="recommendation-card__header">
                <Checkmark size={20} />
                <h3>Recommended: Blurring</h3>
              </div>
              <p>
                Blurring offers the best trade-off between privacy and pose estimation quality,
                maintaining <strong>86% average PCK</strong> across all estimators.
              </p>
            </div>
            <div className="recommendation-card recommendation-card--error">
              <div className="recommendation-card__header">
                <Close size={20} />
                <h3>Avoid: Solid Fill & Pixelation</h3>
              </div>
              <p>
                Solid fill removes nearly all information (11% PCK). Pixelation severely degrades
                most estimators except MediaPipe (81% PCK).
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
