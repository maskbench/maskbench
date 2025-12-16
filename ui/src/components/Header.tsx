import { NavLink } from 'react-router-dom';
import { Settings, Book, ChartBar } from '@carbon/icons-react';
import { Button } from '@carbon/react';

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/new', label: 'New Experiment' },
  { path: '/results', label: 'Results' },
  { path: '/compare', label: 'Compare' },
  { path: '/masked', label: 'Masked Videos' },
  { path: '/history', label: 'History' }
];

export function Header() {
  return (
    <header className="app-header">
      <div className="mb-flex mb-flex--center mb-flex--gap-lg">
        <NavLink to="/" className="app-header__logo">
          <ChartBar size={24} />
          MaskBench
        </NavLink>

        <nav className="app-header__nav">
          {navItems.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `app-header__nav-link ${isActive ? 'app-header__nav-link--active' : ''}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="app-header__actions">
        <Button
          kind="ghost"
          size="sm"
          renderIcon={Book}
          hasIconOnly
          iconDescription="Documentation"
          tooltipPosition="bottom"
        />
        <Button
          kind="ghost"
          size="sm"
          renderIcon={Settings}
          hasIconOnly
          iconDescription="Settings"
          tooltipPosition="bottom"
        />
      </div>
    </header>
  );
}
