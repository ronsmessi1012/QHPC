import React, { useState, useEffect, useRef } from 'react';
import { api } from '../utils/api';
import { Plus, Trash2, Heading1, Heading2, Heading3, Type, CheckSquare, List, Code, Info, ChevronUp, ChevronDown } from 'lucide-react';

interface Block {
  id: string;
  type: 'text' | 'h1' | 'h2' | 'h3' | 'todo' | 'bullet' | 'code' | 'callout';
  content: string;
  checked?: boolean;
}

interface Page {
  id: string;
  title: string;
  blocks: Block[];
}

interface Project {
  id: string;
  name: string;
  pages: Page[];
}

interface BlockEditorProps {
  projectId: string;
  refreshProject: () => void;
  project: Project;
}

export const BlockEditor: React.FC<BlockEditorProps> = ({ projectId, refreshProject, project }) => {
  const [activePageId, setActivePageId] = useState<string>('');
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [pageTitle, setPageTitle] = useState('');
  const [focusedBlockId, setFocusedBlockId] = useState<string | null>(null);

  // Slash commands popover state
  const [slashMenuOpen, setSlashMenuOpen] = useState(false);
  const [slashBlockId, setSlashBlockId] = useState<string | null>(null);
  const [slashCoords, setSlashCoords] = useState({ top: 0, left: 0 });

  // Refs for tracking block element focus
  const blockRefs = useRef<{ [key: string]: HTMLTextAreaElement | null }>({});

  const pages = project.pages || [];
  const activePage = pages.find((p) => p.id === activePageId) || pages[0];

  // Set active page on load or project switch
  useEffect(() => {
    if (activePage) {
      setActivePageId(activePage.id);
      setBlocks(activePage.blocks || []);
      setPageTitle(activePage.title || 'Untitled');
    }
  }, [projectId, activePageId, project]);

  // Sync state if project refreshes from backend
  useEffect(() => {
    if (activePage) {
      setBlocks(activePage.blocks || []);
      setPageTitle(activePage.title || 'Untitled');
    }
  }, [project]);

  // Handle focus changes
  useEffect(() => {
    if (focusedBlockId && blockRefs.current[focusedBlockId]) {
      const textarea = blockRefs.current[focusedBlockId];
      if (textarea) {
        textarea.focus();
        // Move cursor to the end of the text
        const len = textarea.value.length;
        textarea.setSelectionRange(len, len);
      }
      setFocusedBlockId(null);
    }
  }, [focusedBlockId, blocks]);

  // Autosave API Helper
  const savePageData = async (updatedTitle: string, updatedBlocks: Block[]) => {
    if (!activePage) return;
    try {
      await api.put(`/api/projects/${projectId}/pages/${activePage.id}`, {
        title: updatedTitle,
        blocks: updatedBlocks
      });
      // Silent refresh to update parent state
      refreshProject();
    } catch (err) {
      console.error('Failed to save page:', err);
    }
  };

  // Handle Title change
  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setPageTitle(val);
    savePageData(val, blocks);
  };

  // Add Page
  const handleAddPage = async () => {
    try {
      const res = await api.post(`/api/projects/${projectId}/pages`, {
        title: 'New Page'
      });
      refreshProject();
      setActivePageId(res.page.id);
    } catch (err) {
      console.error('Error adding page:', err);
    }
  };

  // Delete Page
  const handleDeletePage = async (pageId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (pages.length <= 1) {
      alert('Projects must have at least one page!');
      return;
    }
    if (!window.confirm('Delete this page and all its content?')) {
      return;
    }
    try {
      await api.delete(`/api/projects/${projectId}/pages/${pageId}`);
      if (activePageId === pageId) {
        // Switch to the first other page
        const otherPage = pages.find((p) => p.id !== pageId);
        if (otherPage) setActivePageId(otherPage.id);
      }
      refreshProject();
    } catch (err) {
      console.error('Error deleting page:', err);
    }
  };

  // Block Content Change
  const handleBlockChange = (blockId: string, value: string) => {
    const updated = blocks.map((b) => {
      if (b.id === blockId) {
        // Trigger slash commands menu if user typing "/"
        if (value.endsWith('/')) {
          setSlashBlockId(blockId);
          triggerSlashMenu(blockId);
        } else if (slashMenuOpen && slashBlockId === blockId && !value.includes('/')) {
          setSlashMenuOpen(false);
        }
        return { ...b, content: value };
      }
      return b;
    });
    setBlocks(updated);
    savePageData(pageTitle, updated);
  };

  // Trigger Slash Commands Menu
  const triggerSlashMenu = (blockId: string) => {
    const el = document.getElementById(`editor-block-${blockId}`);
    if (el) {
      const rect = el.getBoundingClientRect();
      setSlashCoords({
        top: rect.bottom + window.scrollY - 10,
        left: rect.left + window.scrollX + 24
      });
      setSlashMenuOpen(true);
    }
  };

  // Apply Slash Command Type
  const applyBlockType = (blockId: string, type: Block['type']) => {
    const updated = blocks.map((b) => {
      if (b.id === blockId) {
        // Remove the slash char from content
        const cleanContent = b.content.replace(/\/$/, '');
        return { ...b, type, content: cleanContent, checked: type === 'todo' ? false : undefined };
      }
      return b;
    });
    setBlocks(updated);
    setSlashMenuOpen(false);
    setSlashBlockId(null);
    setFocusedBlockId(blockId);
    savePageData(pageTitle, updated);
  };

  // Checkbox toggle for Todo blocks
  const handleTodoToggle = (blockId: string) => {
    const updated = blocks.map((b) => {
      if (b.id === blockId) {
        return { ...b, checked: !b.checked };
      }
      return b;
    });
    setBlocks(updated);
    savePageData(pageTitle, updated);
  };

  // Block keydown handlers (Enter, Backspace, Arrow keys navigation)
  const handleBlockKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>, index: number, blockId: string) => {
    const block = blocks[index];

    if (e.key === 'Enter') {
      e.preventDefault();
      // Add new text block below
      const newBlock: Block = {
        id: 'b_' + Math.random().toString(36).substr(2, 9),
        type: 'text',
        content: ''
      };
      const updated = [...blocks];
      updated.splice(index + 1, 0, newBlock);
      setBlocks(updated);
      setFocusedBlockId(newBlock.id);
      savePageData(pageTitle, updated);
    } else if (e.key === 'Backspace' && block.content === '') {
      e.preventDefault();
      if (blocks.length === 1) return; // Keep at least 1 block

      const updated = blocks.filter((b) => b.id !== blockId);
      setBlocks(updated);
      
      // Focus previous block
      const prevBlock = blocks[index - 1] || blocks[index + 1];
      if (prevBlock) {
        setFocusedBlockId(prevBlock.id);
      }
      savePageData(pageTitle, updated);
    } else if (e.key === 'ArrowUp' && index > 0) {
      e.preventDefault();
      const prevBlock = blocks[index - 1];
      setFocusedBlockId(prevBlock.id);
    } else if (e.key === 'ArrowDown' && index < blocks.length - 1) {
      e.preventDefault();
      const nextBlock = blocks[index + 1];
      setFocusedBlockId(nextBlock.id);
    }
  };

  // Reorder Block Up
  const moveBlockUp = (index: number) => {
    if (index === 0) return;
    const updated = [...blocks];
    const temp = updated[index];
    updated[index] = updated[index - 1];
    updated[index - 1] = temp;
    setBlocks(updated);
    savePageData(pageTitle, updated);
  };

  // Reorder Block Down
  const moveBlockDown = (index: number) => {
    if (index === blocks.length - 1) return;
    const updated = [...blocks];
    const temp = updated[index];
    updated[index] = updated[index + 1];
    updated[index + 1] = temp;
    setBlocks(updated);
    savePageData(pageTitle, updated);
  };

  // Remove Block manually
  const deleteBlock = (blockId: string) => {
    if (blocks.length === 1) return;
    const updated = blocks.filter((b) => b.id !== blockId);
    setBlocks(updated);
    savePageData(pageTitle, updated);
  };

  // Auto-grow Textarea height
  const adjustHeight = (el: HTMLTextAreaElement | null) => {
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${el.scrollHeight}px`;
    }
  };

  if (!activePage) {
    return <div style={{ color: 'var(--text-secondary)' }}>No active pages found in this project.</div>;
  }

  const slashMenuItems: { type: Block['type']; title: string; desc: string; icon: React.ReactNode }[] = [
    { type: 'text', title: 'Text', desc: 'Start writing with plain text.', icon: <Type size={16} /> },
    { type: 'h1', title: 'Heading 1', desc: 'Big section heading.', icon: <Heading1 size={16} /> },
    { type: 'h2', title: 'Heading 2', desc: 'Medium section heading.', icon: <Heading2 size={16} /> },
    { type: 'h3', title: 'Heading 3', desc: 'Small section heading.', icon: <Heading3 size={16} /> },
    { type: 'todo', title: 'To-do List', desc: 'Track tasks with checkboxes.', icon: <CheckSquare size={16} /> },
    { type: 'bullet', title: 'Bulleted List', desc: 'Create a simple bulleted list.', icon: <List size={16} /> },
    { type: 'code', title: 'Code Block', desc: 'Write code snippets.', icon: <Code size={16} /> },
    { type: 'callout', title: 'Callout Box', desc: 'Make writing stand out.', icon: <Info size={16} /> }
  ];

  return (
    <div style={{ display: 'flex', height: '100%', gap: '30px' }}>
      
      {/* --- PAGE SUB-SIDEBAR --- */}
      <div style={{ width: '200px', flexShrink: 0, borderRight: '1px solid var(--border-color)', paddingRight: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Pages</span>
          <button 
            className="modal-close-btn" 
            style={{ padding: '2px', border: '1px solid var(--border-color)' }}
            onClick={handleAddPage}
            title="Create New Page"
          >
            <Plus size={14} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', overflowY: 'auto', flex: 1 }}>
          {pages.map((p) => (
            <div
              key={p.id}
              onClick={() => setActivePageId(p.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                backgroundColor: activePageId === p.id ? 'var(--bg-hover)' : 'transparent',
                color: activePageId === p.id ? 'var(--text-primary)' : 'var(--text-secondary)'
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginRight: '6px' }}>
                {p.title || 'Untitled'}
              </span>
              {pages.length > 1 && (
                <button
                  className="modal-close-btn"
                  style={{ opacity: activePageId === p.id ? 0.6 : 0, padding: '2px' }}
                  onClick={(e) => handleDeletePage(p.id, e)}
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* --- MAIN EDITOR SURFACE --- */}
      <div className="editor-container" style={{ flex: 1 }}>
        <input
          type="text"
          className="editor-title"
          placeholder="Untitled Page"
          value={pageTitle}
          onChange={handleTitleChange}
        />

        <div className="editor-blocks">
          {blocks.map((block, idx) => (
            <div
              key={block.id}
              id={`editor-block-${block.id}`}
              className="editor-block-wrapper"
              style={{ paddingLeft: block.type === 'bullet' ? '12px' : '0px' }}
            >
              {/* Drag/Options Handle */}
              <div className="block-handle" style={{ display: 'flex', flexDirection: 'row', gap: '2px', left: '-42px' }}>
                <button className="modal-close-btn" onClick={() => moveBlockUp(idx)} title="Move Up" style={{ padding: '1px' }}>
                  <ChevronUp size={10} />
                </button>
                <button className="modal-close-btn" onClick={() => moveBlockDown(idx)} title="Move Down" style={{ padding: '1px' }}>
                  <ChevronDown size={10} />
                </button>
                <button className="modal-close-btn" onClick={() => deleteBlock(block.id)} title="Delete Block" style={{ padding: '1px' }}>
                  <Trash2 size={10} style={{ color: 'var(--danger)' }} />
                </button>
              </div>

              {/* Checkbox for Todo List type */}
              {block.type === 'todo' && (
                <div
                  className={`block-todo-checkbox ${block.checked ? 'checked' : ''}`}
                  onClick={() => handleTodoToggle(block.id)}
                  style={{ marginTop: '7px', marginRight: '8px' }}
                >
                  {block.checked && (
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
              )}

              {/* Bullet point indicator */}
              {block.type === 'bullet' && (
                <div className="block-bullet-dot" style={{ marginTop: '13px', marginRight: '8px' }} />
              )}

              {/* Render input area */}
              <div className="block-input-container">
                <textarea
                  ref={(el) => {
                    blockRefs.current[block.id] = el;
                    adjustHeight(el);
                  }}
                  value={block.content}
                  onChange={(e) => handleBlockChange(block.id, e.target.value)}
                  onKeyDown={(e) => handleBlockKeyDown(e, idx, block.id)}
                  rows={1}
                  placeholder={
                    block.type === 'h1'
                      ? 'Heading 1'
                      : block.type === 'h2'
                      ? 'Heading 2'
                      : block.type === 'h3'
                      ? 'Heading 3'
                      : 'Type / to change block type...'
                  }
                  className={`block-input block-${block.type} ${
                    block.type === 'todo' && block.checked ? 'block-todo-text checked' : ''
                  }`}
                  style={{
                    fontStyle: block.type === 'callout' ? 'italic' : 'normal',
                    padding: block.type === 'code' ? '12px' : '4px 6px'
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* --- SLASH POPUP COMMAND MENU --- */}
        {slashMenuOpen && slashBlockId && (
          <div
            className="slash-popup"
            style={{
              top: `${slashCoords.top}px`,
              left: `${slashCoords.left}px`
            }}
          >
            {slashMenuItems.map((item) => (
              <div
                key={item.type}
                className="slash-item"
                onClick={() => applyBlockType(slashBlockId, item.type)}
              >
                <div className="slash-item-icon">{item.icon}</div>
                <div className="slash-item-details">
                  <span className="slash-item-title">{item.title}</span>
                  <span className="slash-item-desc">{item.desc}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
