import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Download,
  Share,
  Renew,
  Trophy,
  DataTable,
  Video,
  Play
} from '@carbon/icons-react';
import { Button, Tabs, Tab, TabList, TabPanels, TabPanel } from '@carbon/react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';

import { useExperimentStore } from '../stores/experimentStore';
import { generateMockExperiment, generateMockMetricResults } from '../data/mockGenerator';
import { ESTIMATORS, METRICS } from '../data/constants';
import { Experiment, MetricId, COCO_KEYPOINTS, MetricResults } from '../types';
import './Results.scss';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
);

export function Results() {
  const { id } = useParams();
  const { getExperiment } = useExperimentStore();

  // Use mock data for demo
  const [experiment] = useState<Experiment>(() =>
    getExperiment(id || '') || generateMockExperiment('completed', {
      name: 'TED Talks Evaluation',
      config: {
        datasetName: 'TED Talks',
        videos: Array.from({ length: 10 }, (_, i) => ({
          id: `v${i}`,
          name: `TED_Talk_${i + 1}.mp4`,
          path: '',
          duration: 120 + i * 30,
          fps: 30,
          width: 1920,
          height: 1080,
          size: 150 * 1024 * 1024,
          frameCount: (120 + i * 30) * 30
        })),
        estimators: ['yolo-medium', 'mediapipe-full', 'openpose-body25', 'maskanyone-api-mp'],
        metrics: ['velocity', 'acceleration', 'jerk'],
        confidenceThreshold: 0.3,
        saveRenderings: true,
        savePoses: true
      }
    })
  );

  const [selectedMetric, setSelectedMetric] = useState<MetricId>('acceleration');

  const estimators = experiment.config.estimators;

  // Helper to get metric results (from experiment or generate mock)
  const getMetricResults = (metricId: MetricId): MetricResults => {
    return experiment.results?.metrics?.[metricId] ||
      generateMockMetricResults(estimators, experiment.config.videos, metricId);
  };

  const metricResults = getMetricResults(selectedMetric);

  // Sort estimators by metric value
  const sortedEstimators = [...estimators].sort((a, b) => {
    const aVal = metricResults.byEstimator[a] || 0;
    const bVal = metricResults.byEstimator[b] || 0;
    return METRICS[selectedMetric].lowerIsBetter ? aVal - bVal : bVal - aVal;
  });

  const byEstimatorValues = Object.values(metricResults.byEstimator) as number[];
  const maxValue = Math.max(...byEstimatorValues);

  // Distribution chart data (simulated)
  const distributionData = {
    datasets: sortedEstimators.map(estId => {
      const est = ESTIMATORS[estId];
      const peak = metricResults.byEstimator[estId];
      const spread = est.category === 'mediapipe' ? 8 : est.category === 'maskanyone' ? 2 : 4;

      const data = [];
      for (let x = 0; x <= 10; x += 0.5) {
        data.push({ x, y: Math.exp(-Math.pow(x - peak, 2) / spread) * 100 });
      }

      return {
        label: est.shortName,
        data,
        borderColor: est.color,
        backgroundColor: `${est.color}20`,
        fill: true,
        tension: 0.4
      };
    })
  };

  // Keypoint chart data
  const keypointData = {
    labels: COCO_KEYPOINTS.map(kp => kp.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())),
    datasets: sortedEstimators.map(estId => {
      const est = ESTIMATORS[estId];
      const byKeypoint = metricResults.byKeypoint || {};

      return {
        label: est.shortName,
        data: COCO_KEYPOINTS.map(kp => byKeypoint[kp]?.[estId] || Math.random() * 3 + 1),
        backgroundColor: est.color,
        borderRadius: 2
      };
    })
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { boxWidth: 12, font: { size: 11 } }
      }
    },
    scales: {
      x: { grid: { display: false } },
      y: { grid: { color: '#e0e0e0' } }
    }
  };

  const getRankTag = (index: number) => {
    if (index === 0) return 'rank-tag--gold';
    if (index === 1) return 'rank-tag--silver';
    if (index === 2) return 'rank-tag--bronze';
    return 'rank-tag--default';
  };

  return (
    <div className="results">
      {/* Page Header */}
      <div className="results__header mb-fade-in">
        <div>
          <h1 className="results__title">
            {experiment.name.split(' ').slice(0, -1).join(' ')} <span>{experiment.name.split(' ').pop()}</span>
          </h1>
          <p className="results__subtitle">
            {experiment.config.videos.length} videos
            {' \u2022 '}
            {experiment.config.estimators.length} estimators
            {' \u2022 '}
            Kinematic metrics
          </p>
        </div>
        <div className="results__actions">
          <Button kind="ghost" size="sm" renderIcon={Renew}>Re-run</Button>
          <Button kind="secondary" size="sm" renderIcon={Share}>Share</Button>
        </div>
      </div>

      {/* Sidebar */}
      <div className="results__layout">
        <aside className="results__sidebar">
          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Experiment</h4>
            <div className="experiment-card">
              <p className="experiment-card__name">{experiment.name}</p>
              <p className="experiment-card__meta">
                {experiment.config.videos.length} videos - Completed
              </p>
              <p className="experiment-card__date">
                {experiment.completedAt?.toLocaleDateString()}
              </p>
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Estimators</h4>
            <div className="estimator-legend">
              {estimators.map(estId => (
                <div key={estId} className="legend-item">
                  <div
                    className="legend-item__dot"
                    style={{ background: ESTIMATORS[estId].color }}
                  />
                  <span>{ESTIMATORS[estId].shortName}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Metric</h4>
            <div className="metric-radio">
              {experiment.config.metrics.map(metricId => (
                <label key={metricId} className="metric-radio__item">
                  <input
                    type="radio"
                    name="metric"
                    value={metricId}
                    checked={selectedMetric === metricId}
                    onChange={() => setSelectedMetric(metricId as MetricId)}
                  />
                  <span>{METRICS[metricId as MetricId].name}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <h4 className="sidebar-section__title">Export</h4>
            <div className="export-buttons">
              <Button kind="secondary" size="sm" renderIcon={Download} style={{ width: '100%', marginBottom: 8 }}>
                Download Plots
              </Button>
              <Button kind="secondary" size="sm" renderIcon={Download} style={{ width: '100%', marginBottom: 8 }}>
                Export CSV
              </Button>
              <Button kind="secondary" size="sm" renderIcon={Download} style={{ width: '100%' }}>
                Export JSON
              </Button>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="results__main">
          <Tabs>
            <TabList aria-label="Results tabs">
              <Tab>Overview</Tab>
              <Tab>Distribution</Tab>
              <Tab>Per-Keypoint</Tab>
              <Tab>Videos</Tab>
            </TabList>

            <TabPanels>
              {/* Overview Tab */}
              <TabPanel>
                {/* Ranking */}
                <div className="mb-card mb-mb-5 mb-slide-up">
                  <div className="mb-card__header">
                    <div className="mb-flex mb-flex--center mb-flex--gap-sm">
                      <Trophy size={16} style={{ color: '#f1c21b' }} />
                      <h3 className="mb-card__title">Estimator Ranking</h3>
                      <span className="mb-text-xs mb-text-gray">
                        (by {METRICS[selectedMetric].name} - {METRICS[selectedMetric].lowerIsBetter ? 'lower' : 'higher'} is better)
                      </span>
                    </div>
                  </div>
                  <div className="ranking-list">
                    {sortedEstimators.map((estId, index) => {
                      const est = ESTIMATORS[estId];
                      const value = metricResults.byEstimator[estId];
                      const pct = (1 - value / (maxValue * 1.2)) * 100;

                      return (
                        <div key={estId} className="ranking-row mb-slide-up" style={{ animationDelay: `${index * 0.1}s` }}>
                          <span className={`rank-tag ${getRankTag(index)}`}>{index + 1}</span>
                          <div className="legend-item__dot" style={{ background: est.color }} />
                          <span className="ranking-row__name">{est.name}</span>
                          <div className="ranking-bar">
                            <div
                              className="ranking-bar__fill"
                              style={{ width: `${pct}%`, background: est.color }}
                            />
                          </div>
                          <span className="ranking-row__value">{value.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Charts Row */}
                <div className="mb-grid mb-grid--2 mb-mb-5">
                  <div className="mb-card mb-slide-up mb-delay-2">
                    <div className="mb-card__header">
                      <h3 className="mb-card__title">{METRICS[selectedMetric].name} Distribution</h3>
                    </div>
                    <div className="mb-card__body">
                      <div className="chart-container">
                        <Line data={distributionData} options={{
                          ...chartOptions,
                          scales: {
                            x: {
                              type: 'linear',
                              title: { display: true, text: `${METRICS[selectedMetric].name} (${METRICS[selectedMetric].unit})` }
                            },
                            y: { title: { display: true, text: '% of Keypoints' } }
                          }
                        }} />
                      </div>
                    </div>
                  </div>
                  <div className="mb-card mb-slide-up mb-delay-3">
                    <div className="mb-card__header">
                      <h3 className="mb-card__title">Per-Keypoint {METRICS[selectedMetric].name}</h3>
                    </div>
                    <div className="mb-card__body">
                      <div className="chart-container">
                        <Bar data={keypointData} options={{
                          ...chartOptions,
                          scales: {
                            x: { grid: { display: false }, ticks: { maxRotation: 45, font: { size: 9 } } },
                            y: { grid: { color: '#e0e0e0' } }
                          }
                        }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Summary Table */}
                <div className="mb-card mb-slide-up mb-delay-4">
                  <div className="mb-card__header">
                    <div className="mb-flex mb-flex--center mb-flex--gap-sm">
                      <DataTable size={16} />
                      <h3 className="mb-card__title">Summary Table</h3>
                    </div>
                    <Button kind="ghost" size="sm" renderIcon={Download}>Export</Button>
                  </div>
                  <div className="summary-table-wrapper">
                    <table className="summary-table">
                      <thead>
                        <tr>
                          <th>Estimator</th>
                          {experiment.config.metrics.map(m => (
                            <th key={m} className="num">{METRICS[m as MetricId].name}</th>
                          ))}
                          <th className="num">Inference Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedEstimators.map(estId => {
                          const est = ESTIMATORS[estId];
                          return (
                            <tr key={estId}>
                              <td>
                                <div className="mb-flex mb-flex--center mb-flex--gap-sm">
                                  <div className="legend-item__dot" style={{ background: est.color }} />
                                  {est.name}
                                </div>
                              </td>
                              {experiment.config.metrics.map(m => {
                                const mResults = getMetricResults(m as MetricId);
                                const value = mResults.byEstimator[estId];
                                const allVals = Object.values(mResults.byEstimator) as number[];
                                const isBest = METRICS[m as MetricId].lowerIsBetter
                                  ? value === Math.min(...allVals)
                                  : value === Math.max(...allVals);
                                const isWorst = METRICS[m as MetricId].lowerIsBetter
                                  ? value === Math.max(...allVals)
                                  : value === Math.min(...allVals);

                                return (
                                  <td
                                    key={m}
                                    className={`num ${isBest ? 'best' : ''} ${isWorst ? 'worst' : ''}`}
                                  >
                                    {value.toFixed(2)}
                                  </td>
                                );
                              })}
                              <td className="num">
                                {est.category === 'mediapipe' ? '0.3s' :
                                 est.category === 'yolo' ? '0.8s' :
                                 est.category === 'openpose' ? '1.2s' : '2.1s'}/frame
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </TabPanel>

              {/* Distribution Tab */}
              <TabPanel>
                <div className="mb-card">
                  <div className="mb-card__header">
                    <h3 className="mb-card__title">{METRICS[selectedMetric].name} Distribution (Full View)</h3>
                  </div>
                  <div className="mb-card__body">
                    <div className="chart-container" style={{ height: 350 }}>
                      <Line data={distributionData} options={{
                        ...chartOptions,
                        scales: {
                          x: {
                            type: 'linear',
                            title: { display: true, text: `${METRICS[selectedMetric].name} (${METRICS[selectedMetric].unit})` }
                          },
                          y: { title: { display: true, text: '% of Keypoints' } }
                        }
                      }} />
                    </div>
                  </div>
                </div>
                <div className="interpretation-card">
                  <h4>Interpretation</h4>
                  <p>
                    Curves concentrated toward zero indicate smoother pose estimation.
                    MaskAnyone-MediaPipe (purple) shows the highest proportion of low-acceleration keypoints,
                    suggesting the most stable trajectories. MediaPipe Pose (blue) has the flattest distribution,
                    indicating more jittery output.
                  </p>
                </div>
              </TabPanel>

              {/* Per-Keypoint Tab */}
              <TabPanel>
                <div className="mb-card">
                  <div className="mb-card__header">
                    <h3 className="mb-card__title">Median {METRICS[selectedMetric].name} by Keypoint</h3>
                  </div>
                  <div className="mb-card__body">
                    <div className="chart-container" style={{ height: 400 }}>
                      <Bar data={keypointData} options={chartOptions} />
                    </div>
                  </div>
                </div>
                <div className="interpretation-card">
                  <h4>Key Findings</h4>
                  <ul>
                    <li>Wrists and ankles show highest acceleration across all estimators (expected due to movement)</li>
                    <li>Nose, eyes, and ears remain most stable</li>
                    <li>MaskAnyone-MP achieves lowest values for all keypoints</li>
                  </ul>
                </div>
              </TabPanel>

              {/* Videos Tab */}
              <TabPanel>
                <div className="mb-card">
                  <div className="mb-card__header">
                    <h3 className="mb-card__title">Rendered Videos</h3>
                    <p className="mb-text-xs mb-text-gray">Click to preview pose overlay for each estimator</p>
                  </div>
                  <div className="video-results-list">
                    {experiment.config.videos.slice(0, 5).map(video => (
                      <div key={video.id} className="video-result-row">
                        <div className="video-result-row__thumb">
                          <Video size={20} />
                        </div>
                        <div className="video-result-row__info">
                          <span className="video-result-row__name">{video.name}</span>
                          <span className="video-result-row__duration">
                            {Math.floor(video.duration / 60)}:{(video.duration % 60).toString().padStart(2, '0')}
                          </span>
                        </div>
                        <div className="video-result-row__dots">
                          {estimators.map(estId => (
                            <div
                              key={estId}
                              className="legend-item__dot"
                              style={{ background: ESTIMATORS[estId].color }}
                              title={ESTIMATORS[estId].name}
                            />
                          ))}
                        </div>
                        <Button kind="ghost" size="sm" renderIcon={Play}>Preview</Button>
                        <Button kind="ghost" size="sm" renderIcon={Download} hasIconOnly iconDescription="Download" />
                      </div>
                    ))}
                  </div>
                </div>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </main>
      </div>
    </div>
  );
}
