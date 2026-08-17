import React, { useState } from 'react';
import { api, type User } from '../utils/api';
import { Lock, User as UserIcon, LogIn, AlertCircle } from 'lucide-react';

interface LoginProps {
  onLoginSuccess: (user: User) => void;
}

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Please fill in all fields');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const res = await api.post('/api/auth/login', { username, password });
      api.setToken(res.token);
      api.setUser(res.user);
      onLoginSuccess(res.user);
    } catch (err: any) {
      setError(err.message || 'Invalid username or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <form className="login-card" onSubmit={handleSubmit}>
        <div>
          <h1 className="login-logo">NotionLocal</h1>
          <p className="login-subtitle">Collaborate securely with your project mates</p>
        </div>

        {error && (
          <div className="error-message">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <div className="form-group">
          <label className="form-label" htmlFor="username">
            Username
          </label>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <UserIcon 
              size={18} 
              style={{ 
                position: 'absolute', 
                left: '12px', 
                color: 'var(--text-muted)' 
              }} 
            />
            <input
              id="username"
              type="text"
              className="form-input"
              style={{ paddingLeft: '38px', width: '100%' }}
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
              autoComplete="username"
            />
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="password">
            Password
          </label>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Lock 
              size={18} 
              style={{ 
                position: 'absolute', 
                left: '12px', 
                color: 'var(--text-muted)' 
              }} 
            />
            <input
              id="password"
              type="password"
              className="form-input"
              style={{ paddingLeft: '38px', width: '100%' }}
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              autoComplete="current-password"
            />
          </div>
        </div>

        <button type="submit" className="btn" style={{ marginTop: '8px' }} disabled={loading}>
          {loading ? (
            <span>Signing in...</span>
          ) : (
            <>
              <span>Sign In</span>
              <LogIn size={18} />
            </>
          )}
        </button>
      </form>
    </div>
  );
};
