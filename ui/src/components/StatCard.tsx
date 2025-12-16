import { ReactNode } from 'react';

interface StatCardProps {
  icon: ReactNode;
  value: string | number;
  label: string;
  trend?: string;
  iconVariant?: 'default' | 'success' | 'warning' | 'error' | 'gray';
  className?: string;
}

export function StatCard({
  icon,
  value,
  label,
  trend,
  iconVariant = 'default',
  className = ''
}: StatCardProps) {
  return (
    <div className={`mb-card mb-stat ${className}`}>
      <div className={`mb-stat__icon mb-stat__icon--${iconVariant}`}>
        {icon}
      </div>
      <div className="mb-stat__value">{value}</div>
      <div className="mb-stat__label">{label}</div>
      {trend && <div className="mb-stat__trend">{trend}</div>}
    </div>
  );
}
