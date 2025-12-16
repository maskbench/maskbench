import { useNavigate } from 'react-router-dom';
import { Checkmark, InProgress, WarningAlt, Document, View, Edit } from '@carbon/icons-react';
import { Button } from '@carbon/react';
import { Experiment } from '../types';

interface ExperimentRowProps {
  experiment: Experiment;
}

const statusConfig = {
  completed: {
    icon: Checkmark,
    tagClass: 'mb-tag--success',
    iconClass: 'mb-stat__icon--success',
    label: 'Completed'
  },
  processing: {
    icon: InProgress,
    tagClass: 'mb-tag--warning',
    iconClass: 'mb-stat__icon--warning',
    label: 'Processing'
  },
  failed: {
    icon: WarningAlt,
    tagClass: 'mb-tag--error',
    iconClass: 'mb-stat__icon--error',
    label: 'Failed'
  },
  draft: {
    icon: Document,
    tagClass: 'mb-tag--gray',
    iconClass: 'mb-stat__icon--gray',
    label: 'Draft'
  },
  queued: {
    icon: InProgress,
    tagClass: 'mb-tag--info',
    iconClass: 'mb-stat__icon--gray',
    label: 'Queued'
  }
};

export function ExperimentRow({ experiment }: ExperimentRowProps) {
  const navigate = useNavigate();
  const status = statusConfig[experiment.status];
  const StatusIcon = status.icon;

  const handleView = () => {
    navigate(`/results/${experiment.id}`);
  };

  return (
    <div className="experiment-row">
      <div className={`experiment-row__icon ${status.iconClass}`}>
        <StatusIcon size={16} className={experiment.status === 'processing' ? 'mb-spin' : ''} />
      </div>

      <div className="experiment-row__info">
        <h3 className="experiment-row__name">{experiment.name}</h3>
        <p className="experiment-row__meta">
          {experiment.config.videos.length} videos
          {' \u2022 '}
          {experiment.config.estimators.length} estimators
          {' \u2022 '}
          {experiment.config.metrics.join(', ')}
        </p>
      </div>

      <span className={`mb-tag ${status.tagClass}`}>
        {status.label}
      </span>

      {experiment.status === 'processing' && (
        <div className="experiment-row__progress">
          <div className="mb-progress" style={{ width: 100 }}>
            <div
              className="mb-progress__fill"
              style={{ width: `${experiment.progress}%` }}
            />
          </div>
          <span className="mb-text-xs mb-text-gray">{Math.round(experiment.progress)}%</span>
        </div>
      )}

      <Button
        kind="ghost"
        size="sm"
        renderIcon={experiment.status === 'completed' ? View : Edit}
        onClick={handleView}
      >
        {experiment.status === 'completed' ? 'View' : 'Edit'}
      </Button>
    </div>
  );
}
