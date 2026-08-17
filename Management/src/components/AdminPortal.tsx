import React, { useEffect, useState } from 'react';
import { api, type User } from '../utils/api';
import { UserPlus, Plus, Trash2, Users, X } from 'lucide-react';

interface Project {
  id: string;
  name: string;
  description: string;
  assignedUserIds: string[];
}

interface AdminPortalProps {
  onProjectCreatedOrModified: () => void;
}

export const AdminPortal: React.FC<AdminPortalProps> = ({ onProjectCreatedOrModified }) => {
  const [users, setUsers] = useState<User[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Modals state
  const [showUserModal, setShowUserModal] = useState(false);
  const [showProjectModal, setShowProjectModal] = useState(false);
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  // New User Form State
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newName, setNewName] = useState('');
  const [newRole, setNewRole] = useState<'admin' | 'member'>('member');
  const [userFormError, setUserFormError] = useState('');
  const [userLoading, setUserLoading] = useState(false);

  // New Project Form State
  const [newProjName, setNewProjName] = useState('');
  const [newProjDesc, setNewProjDesc] = useState('');
  const [newProjMembers, setNewProjMembers] = useState<string[]>(['u1']); // Include admin by default
  const [projFormError, setProjFormError] = useState('');
  const [projLoading, setProjLoading] = useState(false);

  // Fetch initial data
  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const usersRes = await api.get('/api/admin/users');
      const projectsRes = await api.get('/api/projects');
      setUsers(usersRes.users);
      setProjects(projectsRes.projects);
    } catch (err: any) {
      setError(err.message || 'Failed to load administrative data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Handle User creation
  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUsername || !newPassword || !newName) {
      setUserFormError('All fields are required');
      return;
    }
    setUserFormError('');
    setUserLoading(true);
    try {
      await api.post('/api/admin/users', {
        username: newUsername,
        password: newPassword,
        name: newName,
        role: newRole
      });
      // Reset form
      setNewUsername('');
      setNewPassword('');
      setNewName('');
      setNewRole('member');
      setShowUserModal(false);
      // Reload lists
      fetchData();
    } catch (err: any) {
      setUserFormError(err.message || 'Failed to create user');
    } finally {
      setUserLoading(false);
    }
  };

  // Handle Project creation
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjName) {
      setProjFormError('Project name is required');
      return;
    }
    setProjFormError('');
    setProjLoading(true);
    try {
      await api.post('/api/admin/projects', {
        name: newProjName,
        description: newProjDesc,
        assignedUserIds: newProjMembers
      });
      // Reset form
      setNewProjName('');
      setNewProjDesc('');
      setNewProjMembers(['u1']);
      setShowProjectModal(false);
      onProjectCreatedOrModified();
      fetchData();
    } catch (err: any) {
      setProjFormError(err.message || 'Failed to create project');
    } finally {
      setProjLoading(false);
    }
  };

  // Toggle member assignment for new project
  const toggleMemberForNewProject = (userId: string) => {
    if (newProjMembers.includes(userId)) {
      setNewProjMembers(newProjMembers.filter(id => id !== userId));
    } else {
      setNewProjMembers([...newProjMembers, userId]);
    }
  };

  // Open Assign Modal
  const openAssignModal = (project: Project) => {
    setSelectedProject(project);
    setShowAssignModal(true);
  };

  // Save Assignment Updates
  const handleSaveAssignments = async (userIds: string[]) => {
    if (!selectedProject) return;
    try {
      await api.put(`/api/projects/${selectedProject.id}/metadata`, {
        assignedUserIds: userIds
      });
      setShowAssignModal(false);
      setSelectedProject(null);
      onProjectCreatedOrModified();
      fetchData();
    } catch (err: any) {
      alert(err.message || 'Failed to save assignments');
    }
  };

  // Handle Project Delete
  const handleDeleteProject = async (projectId: string, projectName: string) => {
    if (!window.confirm(`Are you sure you want to permanently delete the project "${projectName}"?`)) {
      return;
    }
    try {
      await api.delete(`/api/admin/projects/${projectId}`);
      onProjectCreatedOrModified();
      fetchData();
    } catch (err: any) {
      alert(err.message || 'Failed to delete project');
    }
  };

  if (loading) {
    return <div style={{ color: 'var(--text-secondary)' }}>Loading admin portal...</div>;
  }

  return (
    <div className="admin-container">
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '8px' }}>Admin Dashboard</h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Manage workspaces, assign project access, and register project team mates.
        </p>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* --- PROJECTS MANAGEMENT SECTION --- */}
      <section className="admin-section">
        <div className="admin-section-header">
          <h2 className="admin-section-title">Projects Registry</h2>
          <button className="btn" onClick={() => setShowProjectModal(true)}>
            <Plus size={16} />
            <span>New Project</span>
          </button>
        </div>

        <div className="admin-grid">
          {projects.map((project) => {
            const assignedUsers = users.filter((u) => project.assignedUserIds.includes(u.id));
            return (
              <div key={project.id} className="admin-card">
                <div className="admin-card-header">
                  <div>
                    <h3 className="admin-card-title">{project.name}</h3>
                    <p className="admin-card-subtitle" style={{ marginTop: '4px' }}>
                      {project.description || 'No description provided'}
                    </p>
                  </div>
                </div>

                <div className="admin-card-users">
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Users size={14} />
                    <span>Assigned Mates ({project.assignedUserIds.length})</span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {assignedUsers.length === 0 ? (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>No mates assigned</span>
                    ) : (
                      assignedUsers.map((u) => (
                        <span
                          key={u.id}
                          className="admin-card-badge"
                          style={{ background: u.role === 'admin' ? 'var(--danger-light)' : 'var(--accent-light)' }}
                        >
                          {u.name}
                        </span>
                      ))
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                  <button
                    className="btn btn-secondary"
                    style={{ flex: 1, padding: '6px 12px', fontSize: '0.8rem' }}
                    onClick={() => openAssignModal(project)}
                  >
                    Manage Access
                  </button>
                  <button
                    className="btn btn-danger"
                    style={{ padding: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    onClick={() => handleDeleteProject(project.id, project.name)}
                    title="Delete Project"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* --- USER MANAGEMENT SECTION --- */}
      <section className="admin-section">
        <div className="admin-section-header">
          <h2 className="admin-section-title">Team Directory</h2>
          <button className="btn" onClick={() => setShowUserModal(true)}>
            <UserPlus size={16} />
            <span>Add Project Mate</span>
          </button>
        </div>

        <div className="admin-grid">
          {users.map((user) => (
            <div key={user.id} className="admin-card" style={{ flexDirection: 'row', alignItems: 'center', gap: '16px' }}>
              <div className="user-avatar" style={{ width: '40px', height: '40px', fontSize: '1rem', flexShrink: 0 }}>
                {user.name.slice(0, 2).toUpperCase()}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <h3 className="admin-card-title" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {user.name}
                </h3>
                <p className="admin-card-subtitle">@{user.username}</p>
              </div>
              <span
                className="admin-card-badge"
                style={{
                  background: user.role === 'admin' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(99, 102, 241, 0.15)',
                  color: user.role === 'admin' ? '#f87171' : '#a5b4fc',
                  border: user.role === 'admin' ? '1px solid rgba(239, 68, 68, 0.25)' : '1px solid rgba(99, 102, 241, 0.25)'
                }}
              >
                {user.role}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* --- ADD USER MODAL --- */}
      {showUserModal && (
        <div className="modal-overlay">
          <form className="modal-content" onSubmit={handleCreateUser}>
            <div className="modal-header">
              <h3 className="modal-title">Register Project Mate</h3>
              <button type="button" className="modal-close-btn" onClick={() => setShowUserModal(false)}>
                <X size={18} />
              </button>
            </div>

            {userFormError && <div className="error-message">{userFormError}</div>}

            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Alice Smith"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                disabled={userLoading}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. alice_smith"
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                disabled={userLoading}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Initial Password</label>
              <input
                type="password"
                className="form-input"
                placeholder="••••••••"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                disabled={userLoading}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Access Level</label>
              <select
                className="form-input"
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as 'admin' | 'member')}
                disabled={userLoading}
                style={{ backgroundColor: 'var(--bg-tertiary)' }}
              >
                <option value="member">Member (Access assigned projects only)</option>
                <option value="admin">Admin (Full project registry access)</option>
              </select>
            </div>

            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={() => setShowUserModal(false)} disabled={userLoading}>
                Cancel
              </button>
              <button type="submit" className="btn" disabled={userLoading}>
                {userLoading ? 'Creating...' : 'Register User'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* --- ADD PROJECT MODAL --- */}
      {showProjectModal && (
        <div className="modal-overlay">
          <form className="modal-content" onSubmit={handleCreateProject}>
            <div className="modal-header">
              <h3 className="modal-title">Create Project Workspace</h3>
              <button type="button" className="modal-close-btn" onClick={() => setShowProjectModal(false)}>
                <X size={18} />
              </button>
            </div>

            {projFormError && <div className="error-message">{projFormError}</div>}

            <div className="form-group">
              <label className="form-label">Project Name</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Apollo Missions"
                value={newProjName}
                onChange={(e) => setNewProjName(e.target.value)}
                disabled={projLoading}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="form-input"
                rows={3}
                placeholder="Brief summary of the project goals..."
                value={newProjDesc}
                onChange={(e) => setNewProjDesc(e.target.value)}
                disabled={projLoading}
                style={{ resize: 'vertical' }}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Assign Team Members</label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '160px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '10px', backgroundColor: 'rgba(0,0,0,0.1)' }}>
                {users.map((user) => (
                  <label key={user.id} className="user-assignment-row" style={{ cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={newProjMembers.includes(user.id)}
                      onChange={() => toggleMemberForNewProject(user.id)}
                      disabled={projLoading}
                      style={{ accentColor: 'var(--accent-color)', width: '15px', height: '15px' }}
                    />
                    <span style={{ fontSize: '0.85rem' }}>{user.name} (@{user.username})</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={() => setShowProjectModal(false)} disabled={projLoading}>
                Cancel
              </button>
              <button type="submit" className="btn" disabled={projLoading}>
                {projLoading ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* --- EDIT ASSIGNMENTS MODAL --- */}
      {showAssignModal && selectedProject && (
        <AssignModal
          project={selectedProject}
          users={users}
          onClose={() => {
            setShowAssignModal(false);
            setSelectedProject(null);
          }}
          onSave={handleSaveAssignments}
        />
      )}
    </div>
  );
};

// Modal helper to update project assignments
interface AssignModalProps {
  project: Project;
  users: User[];
  onClose: () => void;
  onSave: (userIds: string[]) => void;
}

const AssignModal: React.FC<AssignModalProps> = ({ project, users, onClose, onSave }) => {
  const [assignedIds, setAssignedIds] = useState<string[]>([...project.assignedUserIds]);

  const handleToggle = (userId: string) => {
    if (assignedIds.includes(userId)) {
      setAssignedIds(assignedIds.filter((id) => id !== userId));
    } else {
      setAssignedIds([...assignedIds, userId]);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h3 className="modal-title">Manage Project Access</h3>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '-8px' }}>
          Assign who can view and contribute to <strong>{project.name}</strong>. Mates not selected will not be able to see this workspace.
        </p>

        <div className="form-group">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '10px', backgroundColor: 'rgba(0,0,0,0.1)' }}>
            {users.map((user) => (
              <label key={user.id} className="user-assignment-row" style={{ cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={assignedIds.includes(user.id)}
                  onChange={() => handleToggle(user.id)}
                  style={{ accentColor: 'var(--accent-color)', width: '15px', height: '15px' }}
                />
                <span style={{ fontSize: '0.85rem' }}>{user.name} (@{user.username})</span>
              </label>
            ))}
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn" onClick={() => onSave(assignedIds)}>
            Save Assignments
          </button>
        </div>
      </div>
    </div>
  );
};
