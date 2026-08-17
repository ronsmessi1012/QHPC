import React, { useState, useEffect } from 'react';
import { api, type User } from '../utils/api';
import { Calendar, AlertCircle, Trash2, X, PlusCircle } from 'lucide-react';

interface Task {
  id: string;
  title: string;
  description: string;
  status: 'todo' | 'in-progress' | 'review' | 'done';
  priority: 'low' | 'medium' | 'high';
  dueDate: string;
  assignedUserId: string;
}

interface Project {
  id: string;
  name: string;
  assignedUserIds: string[];
  tasks: Task[];
}

interface KanbanBoardProps {
  projectId: string;
  refreshProject: () => void;
  project: Project;
}

export const KanbanBoard: React.FC<KanbanBoardProps> = ({ projectId, refreshProject, project }) => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [teamMembers, setTeamMembers] = useState<User[]>([]);
  
  // Modals state
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [editingTask, setEditingTask] = useState<Task | null>(null);

  // Form State
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<Task['status']>('todo');
  const [priority, setPriority] = useState<Task['priority']>('medium');
  const [assignedUserId, setAssignedUserId] = useState('');
  const [dueDate, setDueDate] = useState('');
  const [formError, setFormError] = useState('');
  const [saving, setSaving] = useState(false);

  // Sync tasks on project load/update
  useEffect(() => {
    setTasks(project.tasks || []);
  }, [project]);

  // Fetch team members list for assignments
  useEffect(() => {
    const fetchTeam = async () => {
      try {
        const res = await api.get('/api/admin/users');
        // Filter to only show users assigned to this project (or if current user is admin, they see all, but let's restrict to project members for relevance)
        const projectMates = res.users.filter((u: User) => project.assignedUserIds.includes(u.id));
        setTeamMembers(projectMates);
      } catch (err) {
        console.error('Failed to load team members:', err);
      }
    };
    fetchTeam();
  }, [projectId, project]);

  // Reset form helper
  const resetForm = () => {
    setTitle('');
    setDescription('');
    setStatus('todo');
    setPriority('medium');
    setAssignedUserId('');
    setDueDate('');
    setFormError('');
    setEditingTask(null);
  };

  // Open Create Modal
  const openCreateModal = (colStatus: Task['status']) => {
    resetForm();
    setStatus(colStatus);
    setShowTaskModal(true);
  };

  // Open Edit Modal
  const openEditModal = (task: Task) => {
    setEditingTask(task);
    setTitle(task.title);
    setDescription(task.description);
    setStatus(task.status);
    setPriority(task.priority);
    setAssignedUserId(task.assignedUserId);
    setDueDate(task.dueDate);
    setFormError('');
    setShowTaskModal(true);
  };

  // Save/Submit Form
  const handleSubmitTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title) {
      setFormError('Task title is required');
      return;
    }

    setSaving(true);
    setFormError('');

    const payload = {
      title,
      description,
      status,
      priority,
      assignedUserId,
      dueDate
    };

    try {
      if (editingTask) {
        // Edit Task API
        await api.put(`/api/projects/${projectId}/tasks/${editingTask.id}`, payload);
      } else {
        // Create Task API
        await api.post(`/api/projects/${projectId}/tasks`, payload);
      }
      setShowTaskModal(false);
      resetForm();
      refreshProject();
    } catch (err: any) {
      setFormError(err.message || 'Failed to save task');
    } finally {
      setSaving(false);
    }
  };

  // Delete Task
  const handleDeleteTask = async () => {
    if (!editingTask) return;
    if (!window.confirm('Are you sure you want to delete this task?')) return;

    setSaving(true);
    try {
      await api.delete(`/api/projects/${projectId}/tasks/${editingTask.id}`);
      setShowTaskModal(false);
      resetForm();
      refreshProject();
    } catch (err: any) {
      setFormError(err.message || 'Failed to delete task');
    } finally {
      setSaving(false);
    }
  };

  // Drag and Drop Logic
  const handleDragStart = (e: React.DragEvent, taskId: string) => {
    e.dataTransfer.setData('text/plain', taskId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = async (e: React.DragEvent, targetStatus: Task['status']) => {
    e.preventDefault();
    const taskId = e.dataTransfer.getData('text/plain');
    const task = tasks.find(t => t.id === taskId);
    
    if (task && task.status !== targetStatus) {
      // Optimistic Update
      const updatedTasks = tasks.map(t => t.id === taskId ? { ...t, status: targetStatus } : t);
      setTasks(updatedTasks);
      
      try {
        await api.put(`/api/projects/${projectId}/tasks/${taskId}`, {
          status: targetStatus
        });
        refreshProject();
      } catch (err) {
        console.error('Failed to drag-drop update status:', err);
        // Rollback
        setTasks(tasks);
      }
    }
  };

  const columns: { id: Task['status']; title: string; colorClass: string }[] = [
    { id: 'todo', title: 'To Do', colorClass: 'dot-todo' },
    { id: 'in-progress', title: 'In Progress', colorClass: 'dot-inprogress' },
    { id: 'review', title: 'In Review', colorClass: 'dot-review' },
    { id: 'done', title: 'Done', colorClass: 'dot-done' }
  ];

  return (
    <div className="board-container">
      {columns.map((column) => {
        const columnTasks = tasks.filter((t) => t.status === column.id);
        return (
          <div
            key={column.id}
            className="board-column"
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, column.id)}
          >
            <div className="board-column-header">
              <div className="column-title-container">
                <span className={`column-dot ${column.colorClass}`} />
                <span className="column-title">{column.title}</span>
              </div>
              <span className="column-count">{columnTasks.length}</span>
            </div>

            <div className="board-cards-list">
              {columnTasks.map((task) => {
                const assignee = teamMembers.find(m => m.id === task.assignedUserId);
                return (
                  <div
                    key={task.id}
                    className="board-card"
                    draggable
                    onDragStart={(e) => handleDragStart(e, task.id)}
                    onClick={() => openEditModal(task)}
                  >
                    <span className={`card-priority priority-${task.priority}`}>
                      {task.priority}
                    </span>
                    <h4 className="card-title">{task.title}</h4>
                    {task.description && (
                      <p className="card-description">{task.description}</p>
                    )}
                    
                    <div className="card-footer">
                      <div className="card-date">
                        <Calendar size={12} />
                        <span>{task.dueDate || 'No date'}</span>
                      </div>
                      
                      {assignee && (
                        <div 
                          className="card-assignee" 
                          title={assignee.name}
                        >
                          {assignee.name.slice(0, 2).toUpperCase()}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              <button 
                className="column-add-btn" 
                onClick={() => openCreateModal(column.id)}
              >
                <PlusCircle size={14} />
                <span>Add task</span>
              </button>
            </div>
          </div>
        );
      })}

      {/* --- TASK DETAILS MODAL (Create/Edit) --- */}
      {showTaskModal && (
        <div className="modal-overlay">
          <form className="modal-content" onSubmit={handleSubmitTask}>
            <div className="modal-header">
              <h3 className="modal-title">
                {editingTask ? 'Edit Task Cards' : 'Create Task'}
              </h3>
              <button 
                type="button" 
                className="modal-close-btn" 
                onClick={() => setShowTaskModal(false)}
              >
                <X size={18} />
              </button>
            </div>

            {formError && (
              <div className="error-message">
                <AlertCircle size={18} />
                <span>{formError}</span>
              </div>
            )}

            <div className="form-group">
              <label className="form-label">Task Title</label>
              <input
                type="text"
                className="form-input"
                placeholder="What needs to be done?"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                disabled={saving}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="form-input"
                rows={3}
                placeholder="Add more details about this task..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                disabled={saving}
                style={{ resize: 'vertical' }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
              <div className="form-group">
                <label className="form-label">Status</label>
                <select
                  className="form-input"
                  value={status}
                  onChange={(e) => setStatus(e.target.value as Task['status'])}
                  disabled={saving}
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <option value="todo">To Do</option>
                  <option value="in-progress">In Progress</option>
                  <option value="review">In Review</option>
                  <option value="done">Done</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Priority</label>
                <select
                  className="form-input"
                  value={priority}
                  onChange={(e) => setPriority(e.target.value as Task['priority'])}
                  disabled={saving}
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
              <div className="form-group">
                <label className="form-label">Assignee</label>
                <select
                  className="form-input"
                  value={assignedUserId}
                  onChange={(e) => setAssignedUserId(e.target.value)}
                  disabled={saving}
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <option value="">Unassigned</option>
                  {teamMembers.map((member) => (
                    <option key={member.id} value={member.id}>
                      {member.name} (@{member.username})
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Due Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={dueDate}
                  onChange={(e) => setDueDate(e.target.value)}
                  disabled={saving}
                  style={{ colorScheme: 'dark' }}
                />
              </div>
            </div>

            <div className="modal-footer">
              {editingTask && (
                <button
                  type="button"
                  className="btn btn-danger"
                  style={{ marginRight: 'auto' }}
                  onClick={handleDeleteTask}
                  disabled={saving}
                >
                  <Trash2 size={16} />
                  <span>Delete Task</span>
                </button>
              )}
              
              <button 
                type="button" 
                className="btn btn-secondary" 
                onClick={() => setShowTaskModal(false)}
                disabled={saving}
              >
                Cancel
              </button>
              
              <button 
                type="submit" 
                className="btn" 
                disabled={saving}
              >
                {saving ? 'Saving...' : editingTask ? 'Save Changes' : 'Create Task'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
