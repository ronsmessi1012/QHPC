import express from 'express';
import cors from 'cors';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import {
  getUserByUsername,
  getUserById,
  createUser,
  getUsers,
  getProjectsForUser,
  getProjectById,
  createProject,
  updateProject,
  deleteProject
} from './db.js';

const app = express();
const PORT = process.env.PORT || 5001;
const JWT_SECRET = process.env.JWT_SECRET || 'local_notion_secret_key_12345!';

app.use(cors());
app.use(express.json());

// Authentication Middleware
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }
    
    // Fetch full user details to check role dynamically
    const fullUser = getUserById(user.id);
    if (!fullUser) {
      return res.status(403).json({ error: 'User no longer exists' });
    }
    
    req.user = {
      id: fullUser.id,
      username: fullUser.username,
      role: fullUser.role,
      name: fullUser.name
    };
    next();
  });
}

// Admin-only Middleware
function requireAdmin(req, res, next) {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Administrator access required' });
  }
  next();
}

// Project Authorization Middleware (User must be assigned to the project OR be an admin)
function requireProjectAccess(req, res, next) {
  const projectId = req.params.projectId || req.params.id;
  const project = getProjectById(projectId);

  if (!project) {
    return res.status(404).json({ error: 'Project not found' });
  }

  if (req.user.role !== 'admin' && !project.assignedUserIds.includes(req.user.id)) {
    return res.status(403).json({ error: 'Access denied: You are not assigned to this project' });
  }

  req.project = project;
  next();
}

// --- AUTHENTICATION ENDPOINTS ---

// Login
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  const user = getUserByUsername(username);
  if (!user) {
    return res.status(401).json({ error: 'Invalid username or password' });
  }

  const isPasswordValid = bcrypt.compareSync(password, user.passwordHash);
  if (!isPasswordValid) {
    return res.status(401).json({ error: 'Invalid username or password' });
  }

  // Generate JWT
  const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '7d' });

  res.json({
    token,
    user: {
      id: user.id,
      username: user.username,
      role: user.role,
      name: user.name
    }
  });
});

// Get Current User Profile
app.get('/api/auth/me', authenticateToken, (req, res) => {
  res.json({ user: req.user });
});


// --- ADMIN PORTAL ENDPOINTS ---

// Get All Users (Admin only)
app.get('/api/admin/users', authenticateToken, requireAdmin, (req, res) => {
  res.json({ users: getUsers() });
});

// Create/Register User (Admin only)
app.post('/api/admin/users', authenticateToken, requireAdmin, (req, res) => {
  const { username, password, role, name } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  const existingUser = getUserByUsername(username);
  if (existingUser) {
    return res.status(400).json({ error: 'Username is already taken' });
  }

  const passwordHash = bcrypt.hashSync(password, 10);
  const newUser = createUser(username, passwordHash, role, name);
  res.status(201).json({ user: newUser });
});

// Create Project (Admin only)
app.post('/api/admin/projects', authenticateToken, requireAdmin, (req, res) => {
  const { name, description, assignedUserIds } = req.body;

  if (!name) {
    return res.status(400).json({ error: 'Project name is required' });
  }

  const newProject = createProject(name, description, assignedUserIds);
  res.status(201).json({ project: newProject });
});

// Delete Project (Admin only)
app.delete('/api/admin/projects/:id', authenticateToken, requireAdmin, (req, res) => {
  const success = deleteProject(req.params.id);
  if (success) {
    res.json({ message: 'Project successfully deleted' });
  } else {
    res.status(404).json({ error: 'Project not found' });
  }
});


// --- PROJECTS ENDPOINTS ---

// Get Projects List (Assigned to user, or all if admin)
app.get('/api/projects', authenticateToken, (req, res) => {
  const projects = getProjectsForUser(req.user.id, req.user.role);
  // Strip block content and pages to make the list lightweight, returning only metadata
  const list = projects.map(({ id, name, description, assignedUserIds }) => ({
    id,
    name,
    description,
    assignedUserIds
  }));
  res.json({ projects: list });
});

// Get Project Details (Single project with pages and tasks)
app.get('/api/projects/:projectId', authenticateToken, requireProjectAccess, (req, res) => {
  res.json({ project: req.project });
});

// Update Project Metadata / Members (Admin only)
app.put('/api/projects/:projectId/metadata', authenticateToken, requireAdmin, requireProjectAccess, (req, res) => {
  const { name, description, assignedUserIds } = req.body;
  const project = req.project;

  if (name) project.name = name;
  if (description !== undefined) project.description = description;
  if (assignedUserIds) project.assignedUserIds = assignedUserIds;

  const updated = updateProject(project);
  res.json({ project: updated });
});


// --- PAGES/DOCUMENT MANAGEMENT ENDPOINTS ---

// Create Page in Project
app.post('/api/projects/:projectId/pages', authenticateToken, requireProjectAccess, (req, res) => {
  const { title } = req.body;
  const project = req.project;

  const newPage = {
    id: 'pg_' + Math.random().toString(36).substr(2, 9),
    title: title || 'Untitled Page',
    blocks: [
      { id: 'b_' + Math.random().toString(36).substr(2, 9), type: 'h1', content: title || 'Untitled Page' },
      { id: 'b_' + Math.random().toString(36).substr(2, 9), type: 'text', content: 'Type / to choose blocks...' }
    ]
  };

  if (!project.pages) project.pages = [];
  project.pages.push(newPage);
  updateProject(project);

  res.status(201).json({ page: newPage });
});

// Update Page Content (Title / Blocks)
app.put('/api/projects/:projectId/pages/:pageId', authenticateToken, requireProjectAccess, (req, res) => {
  const { title, blocks } = req.body;
  const project = req.project;
  const pageIndex = project.pages.findIndex(p => p.id === req.params.pageId);

  if (pageIndex === -1) {
    return res.status(404).json({ error: 'Page not found' });
  }

  if (title !== undefined) project.pages[pageIndex].title = title;
  if (blocks !== undefined) project.pages[pageIndex].blocks = blocks;

  updateProject(project);
  res.json({ page: project.pages[pageIndex] });
});

// Delete Page
app.delete('/api/projects/:projectId/pages/:pageId', authenticateToken, requireProjectAccess, (req, res) => {
  const project = req.project;
  
  if (project.pages.length <= 1) {
    return res.status(400).json({ error: 'Projects must have at least one page' });
  }

  const pageIndex = project.pages.findIndex(p => p.id === req.params.pageId);
  if (pageIndex === -1) {
    return res.status(404).json({ error: 'Page not found' });
  }

  project.pages.splice(pageIndex, 1);
  updateProject(project);
  res.json({ message: 'Page successfully deleted' });
});


// --- KANBAN TASK MANAGEMENT ENDPOINTS ---

// Create Task
app.post('/api/projects/:projectId/tasks', authenticateToken, requireProjectAccess, (req, res) => {
  const { title, description, status, priority, dueDate, assignedUserId } = req.body;
  const project = req.project;

  if (!title) {
    return res.status(400).json({ error: 'Task title is required' });
  }

  const newTask = {
    id: 't_' + Math.random().toString(36).substr(2, 9),
    title,
    description: description || '',
    status: status || 'todo',
    priority: priority || 'medium',
    dueDate: dueDate || '',
    assignedUserId: assignedUserId || ''
  };

  if (!project.tasks) project.tasks = [];
  project.tasks.push(newTask);
  updateProject(project);

  res.status(201).json({ task: newTask });
});

// Update Task
app.put('/api/projects/:projectId/tasks/:taskId', authenticateToken, requireProjectAccess, (req, res) => {
  const { title, description, status, priority, dueDate, assignedUserId } = req.body;
  const project = req.project;
  const taskIndex = project.tasks.findIndex(t => t.id === req.params.taskId);

  if (taskIndex === -1) {
    return res.status(404).json({ error: 'Task not found' });
  }

  const task = project.tasks[taskIndex];
  if (title !== undefined) task.title = title;
  if (description !== undefined) task.description = description;
  if (status !== undefined) task.status = status;
  if (priority !== undefined) task.priority = priority;
  if (dueDate !== undefined) task.dueDate = dueDate;
  if (assignedUserId !== undefined) task.assignedUserId = assignedUserId;

  updateProject(project);
  res.json({ task });
});

// Delete Task
app.delete('/api/projects/:projectId/tasks/:taskId', authenticateToken, requireProjectAccess, (req, res) => {
  const project = req.project;
  const taskIndex = project.tasks.findIndex(t => t.id === req.params.taskId);

  if (taskIndex === -1) {
    return res.status(404).json({ error: 'Task not found' });
  }

  project.tasks.splice(taskIndex, 1);
  updateProject(project);
  res.json({ message: 'Task successfully deleted' });
});


// Start Server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running locally on http://0.0.0.0:${PORT}`);
});
