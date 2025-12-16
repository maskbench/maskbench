import { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';

interface QuickActionCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  to: string;
  className?: string;
}

export function QuickActionCard({
  icon,
  title,
  description,
  to,
  className = ''
}: QuickActionCardProps) {
  const navigate = useNavigate();

  return (
    <div
      className={`mb-card mb-card--elevated quick-action-card ${className}`}
      onClick={() => navigate(to)}
      style={{ cursor: 'pointer' }}
    >
      <div className="quick-action-card__icon">
        {icon}
      </div>
      <h3 className="quick-action-card__title">{title}</h3>
      <p className="quick-action-card__description">{description}</p>
    </div>
  );
}
