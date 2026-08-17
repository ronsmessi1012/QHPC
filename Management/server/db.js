import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DB_PATH = path.join(__dirname, 'db.json');

// Helper to read the database
export function readDB() {
  try {
    if (!fs.existsSync(DB_PATH)) {
      return { users: [], projects: [] };
    }
    const data = fs.readFileSync(DB_PATH, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error reading database:', error);
    return { users: [], projects: [] };
  }
}

// Helper to write the database
export function writeDB(data) {
  try {
    fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2), 'utf8');
    return true;
  } catch (error) {
    console.error('Error writing database:', error);
    return false;
  }
}

// User helper methods
export function getUsers() {
  const db = readDB();
  // Don't return password hashes to the client
  return db.users.map(({ passwordHash, ...user }) => user);
}

export function getUserById(id) {
  const db = readDB();
  return db.users.find(u => u.id === id);
}

export function getUserByUsername(username) {
  const db = readDB();
  return db.users.find(u => u.username.toLowerCase() === username.toLowerCase());
}

export function createUser(username, passwordHash, role, name) {
  const db = readDB();
  const newUser = {
    id: 'u_' + Math.random().toString(36).substr(2, 9),
    username,
    passwordHash,
    role: role || 'member',
    name: name || username
  };
  db.users.push(newUser);
  writeDB(db);
  const { passwordHash: _, ...safeUser } = newUser;
  return safeUser;
}

// Project helper methods
export function getProjects() {
  const db = readDB();
  return db.projects || [];
}

export function getProjectById(id) {
  const db = readDB();
  return db.projects.find(p => p.id === id);
}

export function createProject(name, description, assignedUserIds = ['u1']) {
  const db = readDB();
  const newProject = {
    id: 'p_' + Math.random().toString(36).substr(2, 9),
    name,
    description: description || '',
    assignedUserIds,
    pages: [
      {
        id: 'pg_' + Math.random().toString(36).substr(2, 9),
        title: 'Main Page',
        blocks: [
          { id: 'b_init', type: 'h1', content: `Welcome to ${name} 👋` },
          { id: 'b_init_text', type: 'text', content: 'Create pages, document ideas, and assign tasks to your team mates here.' }
        ]
      }
    ],
    tasks: []
  };
  
  if (!db.projects) db.projects = [];
  db.projects.push(newProject);
  writeDB(db);
  return newProject;
}

export function updateProject(updatedProject) {
  const db = readDB();
  const index = db.projects.findIndex(p => p.id === updatedProject.id);
  if (index !== -1) {
    db.projects[index] = {
      ...db.projects[index],
      ...updatedProject
    };
    writeDB(db);
    return db.projects[index];
  }
  return null;
}

export function deleteProject(projectId) {
  const db = readDB();
  const filtered = db.projects.filter(p => p.id !== projectId);
  if (filtered.length !== db.projects.length) {
    db.projects = filtered;
    writeDB(db);
    return true;
  }
  return false;
}

export function getProjectsForUser(userId, role) {
  const db = readDB();
  if (role === 'admin') {
    return db.projects || [];
  }
  return (db.projects || []).filter(p => p.assignedUserIds.includes(userId));
}
