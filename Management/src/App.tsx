import { useState, useEffect } from 'react';
import { api, type User } from './utils/api';
import { Login } from './components/Login';
import { Sidebar } from './components/Sidebar';
import { AdminPortal } from './components/AdminPortal';
import { BlockEditor } from './components/BlockEditor';
import { KanbanBoard } from './components/KanbanBoard';
import { FileText, Kanban, Layers } from 'lucide-react';

interface Project {
  id: string;
  name: string;
  description: string;
  assignedUserIds: string[];
  pages: any[];
  tasks: any[];
}

function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [activeView, setActiveView] = useState<'project' | 'admin'>('project');
  const [workspaceTab, setWorkspaceTab] = useState<'docs' | 'board'>('docs');
  
  // Refresh triggers to force child updates
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [projectRefresh, setProjectRefresh] = useState(0);
  
  const [loading, setLoading] = useState(true);

  // Check login state on boot
  useEffect(() => {
    const initAuth = async () => {
      const storedUser = api.getUser();
      const storedToken = api.getToken();

      if (storedUser && storedToken) {
        try {
          // Verify token against /api/auth/me
          const res = await api.get('/api/auth/me');
          setCurrentUser(res.user);
        } catch (err) {
          console.error('Session expired, logging out:', err);
          api.logout();
          setCurrentUser(null);
        }
      }
      setLoading(false);
    };

    initAuth();
  }, []);

  // Fetch full project details when active ID changes or is refreshed
  useEffect(() => {
    const fetchProjectDetails = async () => {
      if (!activeProjectId) {
        setActiveProject(null);
        return;
      }

      try {
        const res = await api.get(`/api/projects/${activeProjectId}`);
        setActiveProject(res.project);
      } catch (err: any) {
        console.error('Failed to load project details:', err);
        // If forbidden or not found, clear active ID
        setActiveProjectId(null);
        setActiveProject(null);
        alert(err.message || 'Error accessing this project');
      }
    };

    fetchProjectDetails();
  }, [activeProjectId, projectRefresh]);

  const handleLoginSuccess = (user: User) => {
    setCurrentUser(user);
    setActiveProjectId(null);
    setActiveView(user.role === 'admin' ? 'admin' : 'project');
    setSidebarRefresh(prev => prev + 1);
  };

  const handleLogout = () => {
    api.logout();
    setCurrentUser(null);
    setActiveProjectId(null);
    setActiveProject(null);
  };

  const triggerProjectRefresh = () => {
    setProjectRefresh(prev => prev + 1);
  };

  const triggerSidebarRefresh = () => {
    setSidebarRefresh(prev => prev + 1);
  };

  if (loading) {
    return (
      <div style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: 'var(--bg-primary)',
        color: 'var(--text-secondary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'var(--font-sans)'
      }}>
        <div>Loading NotionLocal...</div>
      </div>
    );
  }

  // Render Login page if not authenticated
  if (!currentUser) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-container">
      {/* --- APP SIDEBAR --- */}
      <Sidebar
        currentUser={currentUser}
        onLogout={handleLogout}
        activeProjectId={activeProjectId}
        onSelectProject={(id) => {
          setActiveProjectId(id);
          setWorkspaceTab('docs'); // default tab
        }}
        activeView={activeView}
        onChangeView={setActiveView}
        refreshTrigger={sidebarRefresh}
      />

      {/* --- WORKSPACE VIEW --- */}
      <main className="main-viewport">
        {activeView === 'admin' && currentUser.role === 'admin' ? (
          <div className="main-content" style={{ maxWidth: '1100px' }}>
            <AdminPortal 
              onProjectCreatedOrModified={triggerSidebarRefresh}
            />
          </div>
        ) : activeView === 'project' && activeProject ? (
          <>
            {/* Workspace Header */}
            <header className="main-header">
              <div className="header-title-container">
                <span className="header-title">{activeProject.name}</span>
              </div>
              
              {/* Tab Selector: Documents vs Task Board */}
              <div className="header-tabs">
                <button
                  className={`tab-btn ${workspaceTab === 'docs' ? 'active' : ''}`}
                  onClick={() => setWorkspaceTab('docs')}
                >
                  <FileText size={15} />
                  <span>Document Editor</span>
                </button>
                <button
                  className={`tab-btn ${workspaceTab === 'board' ? 'active' : ''}`}
                  onClick={() => setWorkspaceTab('board')}
                >
                  <Kanban size={15} />
                  <span>Task Board</span>
                </button>
              </div>
            </header>

            {/* Workspace Content */}
            <div className={`main-content ${workspaceTab === 'board' ? 'board-view' : ''}`}>
              {workspaceTab === 'docs' ? (
                <BlockEditor
                  projectId={activeProject.id}
                  project={activeProject}
                  refreshProject={triggerProjectRefresh}
                />
              ) : (
                <KanbanBoard
                  projectId={activeProject.id}
                  project={activeProject}
                  refreshProject={triggerProjectRefresh}
                />
              )}
            </div>
          </>
        ) : (
          /* Empty State / Welcome Screen */
          <div className="empty-state">
            <div className="empty-state-icon">
              <Layers size={32} />
            </div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Select a project to start collaborating
            </h2>
            <p style={{ fontSize: '0.85rem', maxWidth: '340px' }}>
              {currentUser.role === 'admin' 
                ? 'Select an assigned project from the sidebar, or navigate to the Admin Panel to create workspaces and assign team members.'
                : 'Projects assigned to you by your Administrator will appear in the sidebar.'}
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
