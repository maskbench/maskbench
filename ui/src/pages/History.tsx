import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Add,
  Search,
  TrashCan,
  Download,
  Copy,
  View,
  Edit,
  Renew,
  Checkmark,
  InProgress,
  WarningAlt,
  Document,
  ChemistryReference
} from '@carbon/icons-react';
import {
  Button,
  DataTable,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  TableSelectAll,
  TableSelectRow,
  TableToolbar,
  TableToolbarContent,
  TableToolbarSearch,
  TableContainer,
  Pagination,
  Tag,
  Modal
} from '@carbon/react';

import { useExperimentStore } from '../stores/experimentStore';
import { ESTIMATORS } from '../data/constants';
import { Experiment } from '../types';
import './History.scss';

const statusConfig = {
  completed: { kind: 'green' as const, icon: Checkmark, label: 'Completed' },
  processing: { kind: 'blue' as const, icon: InProgress, label: 'Processing' },
  failed: { kind: 'red' as const, icon: WarningAlt, label: 'Failed' },
  draft: { kind: 'gray' as const, icon: Document, label: 'Draft' },
  queued: { kind: 'cyan' as const, icon: InProgress, label: 'Queued' }
};

export function History() {
  const navigate = useNavigate();
  const { experiments, loadExperiments, deleteExperiment, isLoading } = useExperimentStore();
  const [selectedRows, setSelectedRows] = useState<string[]>([]);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  const headers = [
    { key: 'name', header: 'Experiment' },
    { key: 'videos', header: 'Videos' },
    { key: 'estimators', header: 'Estimators' },
    { key: 'date', header: 'Date' },
    { key: 'status', header: 'Status' },
    { key: 'actions', header: 'Actions' }
  ];

  const rows = experiments.map(exp => ({
    id: exp.id,
    name: exp.name,
    description: exp.description,
    videos: exp.config.videos.length,
    estimators: exp.config.estimators,
    date: exp.createdAt.toLocaleDateString(),
    status: exp.status,
    progress: exp.progress
  }));

  const paginatedRows = rows.slice((page - 1) * pageSize, page * pageSize);

  const stats = {
    total: experiments.length,
    completed: experiments.filter(e => e.status === 'completed').length,
    processing: experiments.filter(e => e.status === 'processing').length,
    failed: experiments.filter(e => e.status === 'failed').length
  };

  const handleDelete = async () => {
    for (const id of selectedRows) {
      await deleteExperiment(id);
    }
    setSelectedRows([]);
    setShowDeleteModal(false);
  };

  return (
    <div className="history">
      {/* Header */}
      <div className="history__header mb-fade-in">
        <div>
          <h1 className="history__title">
            Experiment <span>History</span>
          </h1>
          <p className="history__subtitle">
            View, manage, and export past benchmarking experiments
          </p>
        </div>
        <div className="history__actions">
          <Button kind="ghost" size="sm" renderIcon={Download}>Export All</Button>
          <Button kind="primary" renderIcon={Add} onClick={() => navigate('/new')}>
            New Experiment
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="history__stats mb-fade-in">
        <div className="stat-mini">
          <div className="stat-mini__icon"><ChemistryReference size={16} /></div>
          <div className="stat-mini__content">
            <span className="stat-mini__value">{stats.total}</span>
            <span className="stat-mini__label">Total</span>
          </div>
        </div>
        <div className="stat-mini stat-mini--success">
          <div className="stat-mini__icon"><Checkmark size={16} /></div>
          <div className="stat-mini__content">
            <span className="stat-mini__value">{stats.completed}</span>
            <span className="stat-mini__label">Completed</span>
          </div>
        </div>
        <div className="stat-mini stat-mini--warning">
          <div className="stat-mini__icon"><InProgress size={16} /></div>
          <div className="stat-mini__content">
            <span className="stat-mini__value">{stats.processing}</span>
            <span className="stat-mini__label">Processing</span>
          </div>
        </div>
        <div className="stat-mini stat-mini--error">
          <div className="stat-mini__icon"><WarningAlt size={16} /></div>
          <div className="stat-mini__content">
            <span className="stat-mini__value">{stats.failed}</span>
            <span className="stat-mini__label">Failed</span>
          </div>
        </div>
      </div>

      {/* Table */}
      <DataTable rows={paginatedRows} headers={headers}>
        {({ rows, headers, getTableProps, getHeaderProps, getRowProps, getSelectionProps, selectedRows: selected }) => (
          <TableContainer>
            <TableToolbar>
              <TableToolbarContent>
                <TableToolbarSearch onChange={() => {}} />
                {selected.length > 0 && (
                  <>
                    <Button kind="ghost" size="sm" renderIcon={Download}>Export</Button>
                    <Button kind="ghost" size="sm" renderIcon={Copy}>Duplicate</Button>
                    <Button kind="danger--ghost" size="sm" renderIcon={TrashCan} onClick={() => {
                      setSelectedRows(selected.map((r: any) => r.id));
                      setShowDeleteModal(true);
                    }}>
                      Delete
                    </Button>
                  </>
                )}
              </TableToolbarContent>
            </TableToolbar>
            <Table {...getTableProps()}>
              <TableHead>
                <TableRow>
                  <TableSelectAll {...getSelectionProps()} />
                  {headers.map(header => (
                    <TableHeader {...getHeaderProps({ header })} key={header.key}>
                      {header.header}
                    </TableHeader>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((row: any) => {
                  const exp = experiments.find(e => e.id === row.id);
                  const status = statusConfig[row.cells[4].value as keyof typeof statusConfig];
                  const StatusIcon = status.icon;

                  return (
                    <TableRow {...getRowProps({ row })} key={row.id}>
                      <TableSelectRow {...getSelectionProps({ row })} />
                      <TableCell>
                        <div className="experiment-cell">
                          <span className="experiment-cell__name">{row.cells[0].value}</span>
                          <span className="experiment-cell__desc">{exp?.description}</span>
                        </div>
                      </TableCell>
                      <TableCell>{row.cells[1].value}</TableCell>
                      <TableCell>
                        <div className="estimator-dots">
                          {(row.cells[2].value as string[]).slice(0, 4).map((estId: string) => (
                            <div
                              key={estId}
                              className="estimator-dot"
                              style={{ background: ESTIMATORS[estId as keyof typeof ESTIMATORS]?.color || '#000' }}
                              title={ESTIMATORS[estId as keyof typeof ESTIMATORS]?.name}
                            />
                          ))}
                        </div>
                      </TableCell>
                      <TableCell>{row.cells[3].value}</TableCell>
                      <TableCell>
                        <Tag type={status.kind} renderIcon={StatusIcon}>
                          {status.label}
                        </Tag>
                      </TableCell>
                      <TableCell>
                        <div className="action-buttons">
                          {row.cells[4].value === 'completed' && (
                            <>
                              <Button
                                kind="ghost"
                                size="sm"
                                hasIconOnly
                                iconDescription="View"
                                renderIcon={View}
                                onClick={() => navigate(`/results/${row.id}`)}
                              />
                              <Button kind="ghost" size="sm" hasIconOnly iconDescription="Download" renderIcon={Download} />
                            </>
                          )}
                          {row.cells[4].value === 'failed' && (
                            <Button kind="ghost" size="sm" hasIconOnly iconDescription="Retry" renderIcon={Renew} />
                          )}
                          {row.cells[4].value === 'draft' && (
                            <Button
                              kind="ghost"
                              size="sm"
                              hasIconOnly
                              iconDescription="Edit"
                              renderIcon={Edit}
                              onClick={() => navigate('/new')}
                            />
                          )}
                          <Button
                            kind="ghost"
                            size="sm"
                            hasIconOnly
                            iconDescription="Delete"
                            renderIcon={TrashCan}
                            onClick={() => {
                              setSelectedRows([row.id]);
                              setShowDeleteModal(true);
                            }}
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            <Pagination
              totalItems={experiments.length}
              pageSize={pageSize}
              pageSizes={[10, 20, 50]}
              page={page}
              onChange={({ page, pageSize }) => {
                setPage(page);
                setPageSize(pageSize);
              }}
            />
          </TableContainer>
        )}
      </DataTable>

      {/* Delete Modal */}
      <Modal
        open={showDeleteModal}
        danger
        modalHeading="Delete Experiments?"
        primaryButtonText="Delete"
        secondaryButtonText="Cancel"
        onRequestClose={() => setShowDeleteModal(false)}
        onRequestSubmit={handleDelete}
      >
        <p>
          This will permanently delete {selectedRows.length} experiment(s) and all associated data.
          This action cannot be undone.
        </p>
      </Modal>
    </div>
  );
}
