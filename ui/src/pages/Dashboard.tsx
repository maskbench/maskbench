import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Add,
  Video,
  Folder,
  FaceMask,
  Scale,
  ChemistryReference,
  Checkmark,
  Bot,
  Time
} from '@carbon/icons-react';
import { Button, Loading } from '@carbon/react';
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

import { StatCard, QuickActionCard, ExperimentRow } from '../components';
import { useExperimentStore } from '../stores/experimentStore';
import './Dashboard.scss';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export function Dashboard() {
  const navigate = useNavigate();
  const { dashboardStats, loadDashboardStats, isLoading } = useExperimentStore();

  useEffect(() => {
    loadDashboardStats();
  }, [loadDashboardStats]);

  if (isLoading || !dashboardStats) {
    return (
      <div className="dashboard-loading">
        <Loading withOverlay={false} />
      </div>
    );
  }

  const activityChartData = {
    labels: dashboardStats.weeklyActivity.map(d => d.day),
    datasets: [
      {
        label: 'Videos',
        data: dashboardStats.weeklyActivity.map(d => d.videos),
        backgroundColor: '#000000',
        borderRadius: 4
      },
      {
        label: 'Experiments',
        data: dashboardStats.weeklyActivity.map(d => d.experiments),
        backgroundColor: '#8d8d8d',
        borderRadius: 4
      }
    ]
  };

  const performanceChartData = {
    labels: ['YOLO11', 'MediaPipe', 'OpenPose', 'MaskAnyone-MP'],
    datasets: [{
      label: 'Acceleration (lower is better)',
      data: [1.46, 4.52, 2.22, 1.08],
      backgroundColor: ['#000000', '#0f62fe', '#ff832b', '#8a3ffc'],
      borderRadius: 4
    }]
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

  const horizontalChartOptions = {
    ...chartOptions,
    indexAxis: 'y' as const,
    plugins: { legend: { display: false } }
  };

  return (
    <div className="dashboard">
      {/* Page Header */}
      <div className="mb-flex mb-flex--between mb-flex--center mb-mb-6 mb-fade-in">
        <div className="mb-page-header">
          <h1 className="mb-page-header__title">
            Pose Estimation <span>Benchmarking</span>
          </h1>
          <p className="mb-page-header__subtitle">
            Compare estimators, evaluate metrics, make informed decisions
          </p>
          <div className="mb-page-header__accent" />
        </div>
        <Button
          kind="primary"
          renderIcon={Add}
          onClick={() => navigate('/new')}
        >
          New Experiment
        </Button>
      </div>

      {/* Quick Actions */}
      <section className="mb-mb-6">
        <h4 className="section-title">
          <ChemistryReference size={16} />
          Quick Start
        </h4>
        <div className="mb-grid mb-grid--4">
          <QuickActionCard
            icon={<Video size={20} />}
            title="Single Video"
            description="Benchmark one video across all estimators"
            to="/new?mode=single"
            className="mb-slide-up mb-delay-1"
          />
          <QuickActionCard
            icon={<Folder size={20} />}
            title="Dataset"
            description="Evaluate entire dataset with metrics"
            to="/new?mode=dataset"
            className="mb-slide-up mb-delay-2"
          />
          <QuickActionCard
            icon={<FaceMask size={20} />}
            title="Masked Videos"
            description="Test estimators on de-identified data"
            to="/masked"
            className="mb-slide-up mb-delay-3"
          />
          <QuickActionCard
            icon={<Scale size={20} />}
            title="Compare"
            description="Side-by-side estimator comparison"
            to="/compare"
            className="mb-slide-up mb-delay-4"
          />
        </div>
      </section>

      {/* Stats */}
      <section className="mb-mb-6">
        <div className="mb-grid mb-grid--4">
          <StatCard
            icon={<ChemistryReference size={20} />}
            value={dashboardStats.totalExperiments}
            label="Experiments Run"
            trend="+3 this week"
            className="mb-slide-up mb-delay-1"
          />
          <StatCard
            icon={<Video size={20} />}
            value={dashboardStats.totalVideosProcessed}
            label="Videos Processed"
            trend="+12 this week"
            iconVariant="success"
            className="mb-slide-up mb-delay-2"
          />
          <StatCard
            icon={<Bot size={20} />}
            value={dashboardStats.availableEstimators}
            label="Estimators Available"
            iconVariant="gray"
            className="mb-slide-up mb-delay-3"
          />
          <StatCard
            icon={<Time size={20} />}
            value={`${dashboardStats.avgProcessingTime}h`}
            label="Avg Processing Time"
            trend="-15min vs last week"
            iconVariant="warning"
            className="mb-slide-up mb-delay-4"
          />
        </div>
      </section>

      {/* Charts */}
      <section className="mb-mb-6">
        <div className="mb-grid mb-grid--2">
          <div className="mb-card mb-slide-up mb-delay-2">
            <div className="mb-card__header">
              <h3 className="mb-card__title">Weekly Activity</h3>
            </div>
            <div className="mb-card__body">
              <div className="chart-container">
                <Bar data={activityChartData} options={chartOptions} />
              </div>
            </div>
          </div>
          <div className="mb-card mb-slide-up mb-delay-3">
            <div className="mb-card__header">
              <h3 className="mb-card__title">Estimator Performance</h3>
            </div>
            <div className="mb-card__body">
              <div className="chart-container">
                <Bar data={performanceChartData} options={horizontalChartOptions} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Recent Experiments */}
      <section className="mb-slide-up mb-delay-4">
        <h4 className="section-title">
          <Time size={16} />
          Recent Experiments
        </h4>
        <div className="mb-card">
          {dashboardStats.recentExperiments.map(experiment => (
            <ExperimentRow key={experiment.id} experiment={experiment} />
          ))}
        </div>
      </section>
    </div>
  );
}
