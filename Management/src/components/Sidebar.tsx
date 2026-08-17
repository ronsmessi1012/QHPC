import React, { useEffect, useState } from 'react';
import { api, type User } from '../utils/api';
import { Folder, Shield, LogOut, ChevronRight } from 'lucide-react';

interface SidebarProps {
  currentUser: User;
  onLogout: () => void;
  activeProjectId: string | null;
  onSelectProject: (id: string) => void;
  activeView: 'project' | 'admin';
  onChangeView: (view: 'project' | 'admin') => void;
  refreshTrigger: number;
}

interface ProjectMetadata {
  id: string;
  name: string;
  description: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentUser,
  onLogout,
  activeProjectId,
  onSelectProject,
  activeView,
  onChangeView,
  refreshTrigger
}) => {
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const res = await api.get('/api/projects');
        setProjects(res.projects);
      } catch (err) {
        console.error('Error fetching sidebar projects:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, [refreshTrigger, currentUser]);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">NotionLocal</div>
      </div>

      <div className="sidebar-user">
        <div className="user-avatar">
          {currentUser.name ? currentUser.name.slice(0, 2).toUpperCase() : 'U'}
        </div>
        <div className="user-details">
          <div className="user-name">{currentUser.name || currentUser.username}</div>
          <div className="user-role">{currentUser.role}</div>
        </div>
      </div>

      <div className="sidebar-menu">
        {currentUser.role === 'admin' && (
          <>
            <div className="sidebar-section-title">Administration</div>
            <div
              className={`sidebar-item ${activeView === 'admin' ? 'active' : ''}`}
              onClick={() => onChangeView('admin')}
            >
              <Shield size={18} />
              <span>Admin Panel</span>
              <ChevronRight size={14} style={{ marginLeft: 'auto', opacity: 0.5 }} />
            </div>
          </>
        )}

        <div className="sidebar-section-title">Projects</div>
        {loading ? (
          <div style={{ padding: '8px 12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Loading projects...
          </div>
        ) : projects.length === 0 ? (
          <div style={{ padding: '8px 12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            No projects assigned
          </div>
        ) : (
          projects.map((project) => (
            <div
              key={project.id}
              className={`sidebar-item ${
                activeView === 'project' && activeProjectId === project.id ? 'active' : ''
              }`}
              onClick={() => {
                onChangeView('project');
                onSelectProject(project.id);
              }}
              title={project.description}
            >
              <Folder size={18} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {project.name}
              </span>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <button className="btn btn-secondary" onClick={onLogout} style={{ width: '100%' }}>
          <LogOut size={16} />
          <span>Log Out</span>
        </button>
      </div>
    </aside>
  );
};
