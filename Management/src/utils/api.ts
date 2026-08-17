const TOKEN_KEY = 'notion_local_token';
const USER_KEY = 'notion_local_user';

export interface User {
  id: string;
  username: string;
  role: 'admin' | 'member';
  name: string;
}

export const api = {
  setToken(token: string) {
    localStorage.setItem(TOKEN_KEY, token);
  },

  getToken() {
    return localStorage.getItem(TOKEN_KEY);
  },

  setUser(user: User) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },

  getUser(): User | null {
    const user = localStorage.getItem(USER_KEY);
    return user ? JSON.parse(user) : null;
  },

  logout() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  },

  async request(url: string, method: string, data?: any) {
    const token = this.getToken();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const config: RequestInit = {
      method,
      headers,
    };

    if (data) {
      config.body = JSON.stringify(data);
    }

    const response = await fetch(url, config);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP error! Status: ${response.status}`);
    }

    return result;
  },

  get(url: string) {
    return this.request(url, 'GET');
  },

  post(url: string, data: any) {
    return this.request(url, 'POST', data);
  },

  put(url: string, data: any) {
    return this.request(url, 'PUT', data);
  },

  delete(url: string) {
    return this.request(url, 'DELETE');
  }
};
