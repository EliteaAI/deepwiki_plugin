import { useState, useCallback, useRef, useEffect, memo, useMemo } from 'react';
import {
  Box,
  Chip,
  Typography,
  TextField,
  IconButton,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
  Collapse,
  Drawer,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  Send as SendIcon,
  QuestionAnswer as AskIcon,
  Science as ResearchIcon,
  Close as CloseIcon,
  Chat as ChatIcon,
  ExpandLess as ExpandIcon,
  Code as CodeIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  ContentCopy as CopyIcon,
  Refresh as RegenerateIcon,
  DeleteOutline as ClearIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { v4 as uuidv4 } from 'uuid';
import MermaidDiagram from './MermaidDiagram';
import { useManualSocket, sioEvents, SocketMessageType, getSocketId, getSocket } from '../hooks/useSocket';

const DEFAULT_DRAWER_WIDTH = 480;
const MIN_DRAWER_WIDTH = 350;
const MAX_DRAWER_WIDTH = 800;

const MAX_THINKING_STEPS_PER_RUN = 200;
const SCROLL_NEAR_BOTTOM_THRESHOLD_PX = 120;

/**
 * Vertical tab button shown when chat is collapsed.
 * Positioned on the right edge of the screen.
 */
const ChatTab = memo(function ChatTab({ onClick, disabled }) {
  return (
    <Tooltip title={disabled ? "Generate wiki first to enable chat" : "Open Chat"} placement="left">
      <Box
        onClick={disabled ? undefined : onClick}
        sx={{
          position: 'fixed',
          right: 0,
          top: '50%',
          transform: 'translateY(-50%)',
          bgcolor: disabled ? 'action.disabled' : 'primary.main',
          color: disabled ? 'text.disabled' : 'primary.contrastText',
          borderRadius: '8px 0 0 8px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '16px 8px',
          gap: '8px',
          boxShadow: 3,
          transition: 'all 0.2s ease',
          zIndex: 1200,
          opacity: disabled ? 0.6 : 1,
          '&:hover': disabled ? {} : {
            bgcolor: 'primary.dark',
            paddingRight: '12px',
          },
        }}
      >
        <ChatIcon fontSize="small" />
        <Typography
          variant="caption"
          sx={{
            writingMode: 'vertical-rl',
            textOrientation: 'mixed',
            transform: 'rotate(180deg)',
            fontWeight: 600,
            letterSpacing: '0.5px',
          }}
        >
          Chat
        </Typography>
        <ExpandIcon
          fontSize="small"
          sx={{ transform: 'rotate(-90deg)' }}
        />
      </Box>
    </Tooltip>
  );
});

/**
 * Todos panel - Copilot-style todo widget
 * Collapsed: Shows "Todos (n/n)" on top, current item full-width below with pulsing dot
 * Expanded: Full list with checkmarks
 */
const TodosPanel = memo(function TodosPanel({ todos, expanded, onToggle, onDismiss }) {
  const hasTodos = todos && todos.length > 0;
  
  if (!hasTodos) return null;

  // Calculate progress
  const completedTodos = todos.filter(t => t.status === 'completed').length;
  const inProgressTodo = todos.find(t => t.status === 'in-progress');
  const totalTodos = todos.length;

  // Get current active item for collapsed display
  const currentItem = inProgressTodo || todos.find(t => t.status !== 'completed');

  return (
    <Box
      sx={{
        borderTop: 1,
        borderColor: 'divider',
        bgcolor: 'background.default',
      }}
    >
      {/* Header row with label and controls */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1.5,
          py: 0.5,
        }}
      >
        {/* Toggle and label */}
        <Box
          onClick={onToggle}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            cursor: 'pointer',
            '&:hover': { opacity: 0.8 },
          }}
        >
          <ExpandMoreIcon
            fontSize="small"
            sx={{
              transform: expanded ? 'rotate(0deg)' : 'rotate(-90deg)',
              transition: 'transform 0.2s',
              color: 'text.secondary',
              fontSize: 16,
            }}
          />
          <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ fontSize: '11px' }}>
            Todos ({completedTodos}/{totalTodos})
          </Typography>
        </Box>

        {/* Dismiss button */}
        <IconButton
          size="small"
          onClick={onDismiss}
          sx={{
            p: 0.25,
            color: 'text.secondary',
            '&:hover': { color: 'text.primary' },
          }}
        >
          <CloseIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Box>

      {/* Collapsed: Show current item full width */}
      {!expanded && currentItem && (
        <Box
          onClick={onToggle}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 1.5,
            pb: 0.75,
            cursor: 'pointer',
          }}
        >
          {/* Pulsing green dot for in-progress */}
          <Typography
            component="span"
            sx={{
              fontSize: '12px',
              color: '#4CAF50',
              fontWeight: 600,
              ...(inProgressTodo && {
                animation: 'todoPulse 1.5s ease-in-out infinite',
              }),
            }}
          >
            ●
          </Typography>
          <Typography
            variant="caption"
            sx={{
              flex: 1,
              fontSize: '12px',
              color: 'text.primary',
              lineHeight: 1.3,
            }}
          >
            {currentItem.title}
          </Typography>
        </Box>
      )}

      {/* Expanded: Full todo list */}
      <Collapse in={expanded}>
        <Box sx={{ px: 1.5, pb: 1, maxHeight: 200, overflow: 'auto' }}>
          {todos.map((todo, i) => {
            const status = todo.status || 'not-started';
            const isCompleted = status === 'completed';
            const isInProgress = status === 'in-progress';
            
            return (
              <Box 
                key={todo.id || i}
                sx={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  gap: 0.75,
                  py: 0.4,
                }}
              >
                <Typography
                  component="span"
                  sx={{ 
                    fontSize: '12px',
                    lineHeight: 1.4,
                    color: isCompleted ? '#4CAF50' : isInProgress ? '#4CAF50' : 'text.secondary',
                    fontWeight: 600,
                    minWidth: '14px',
                    ...(isInProgress && { animation: 'todoPulse 1.5s ease-in-out infinite' }),
                  }}
                >
                  {isCompleted ? '✓' : isInProgress ? '●' : '○'}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ 
                    fontSize: '11px', 
                    lineHeight: 1.4,
                    color: isCompleted ? 'text.secondary' : 'text.primary',
                    textDecoration: isCompleted ? 'line-through' : 'none',
                    opacity: isCompleted ? 0.7 : 1,
                  }}
                >
                  {todo.title}
                </Typography>
              </Box>
            );
          })}
        </Box>
      </Collapse>
      
      <style>{`@keyframes todoPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }`}</style>
    </Box>
  );
});

/**
 * Tool call card - Collapsible, shows tool name + combined input/output
 */
const ToolCallCard = memo(function ToolCallCard({ step, mode }) {
  const [expanded, setExpanded] = useState(false);
  
  const data = step.data || step;
  const toolName = data.tool || data.toolName || 'tool';
  const input = data.input || '';
  const output = data.output || '';
  const isCompleted = step.event === 'tool_end' || data.status === 'completed';
  const hasContent = (input && input.length > 0) || (output && output.length > 0);
  
  // Format tool name
  const displayName = toolName
    .replace(/_/g, ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');

  return (
    <Box
      sx={{
        mb: 0.75,
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        onClick={() => hasContent && setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.75,
          py: 0.6,
          px: 1,
          cursor: hasContent ? 'pointer' : 'default',
          '&:hover': hasContent ? { bgcolor: 'action.hover' } : {},
        }}
      >
        {isCompleted ? (
          <CheckCircleIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CircularProgress size={12} thickness={4} />
        )}
        <Typography variant="body2" sx={{ flex: 1, fontSize: '12px', color: 'text.primary' }}>
          {displayName}
        </Typography>
        {hasContent && (
          <ExpandMoreIcon
            sx={{
              fontSize: 16,
              color: 'text.secondary',
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
            }}
          />
        )}
      </Box>
      
      {/* Expanded content - shows both input and output */}
      <Collapse in={expanded}>
        <Box sx={{ px: 1, pb: 0.75, borderTop: '1px solid', borderColor: 'divider' }}>
          {input && (
            <Box sx={{ mt: 0.5 }}>
              <Typography variant="caption" sx={{ fontSize: '10px', color: 'text.secondary', fontWeight: 600 }}>
                Input:
              </Typography>
              <Typography
                component="pre"
                sx={{
                  fontSize: '10px',
                  lineHeight: 1.4,
                  color: 'text.secondary',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  m: 0,
                  mt: 0.25,
                  p: 0.5,
                  bgcolor: mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.04)',
                  borderRadius: 0.5,
                  maxHeight: 80,
                  overflow: 'auto',
                }}
              >
                {typeof input === 'object' ? JSON.stringify(input, null, 2) : String(input).slice(0, 500)}
              </Typography>
            </Box>
          )}
          {output && (
            <Box sx={{ mt: 0.5 }}>
              <Typography variant="caption" sx={{ fontSize: '10px', color: 'text.secondary', fontWeight: 600 }}>
                Output:
              </Typography>
              <Typography
                component="pre"
                sx={{
                  fontSize: '10px',
                  lineHeight: 1.4,
                  color: 'text.secondary',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  m: 0,
                  mt: 0.25,
                  p: 0.5,
                  bgcolor: mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.04)',
                  borderRadius: 0.5,
                  maxHeight: 100,
                  overflow: 'auto',
                }}
              >
                {typeof output === 'object' ? JSON.stringify(output, null, 2) : String(output).slice(0, 500)}
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
});

/**
 * Log message - Non-intrusive bullet point style (like screenshot 4)
 */
const LogMessage = memo(function LogMessage({ message }) {
  const [expanded, setExpanded] = useState(false);

  if (!message) return null;

  const str = String(message);
  const lines = str.split('\n');
  const firstLine = lines[0] || '';
  const isMultiline = lines.length > 1;
  const isLong = str.length > 180;
  const canExpand = isMultiline || isLong;
  const collapsedText = firstLine.length > 160 ? `${firstLine.slice(0, 160)}…` : firstLine;

  const handleToggle = () => {
    if (!canExpand) return;
    setExpanded(prev => !prev);
  };

  if (!canExpand) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.75, py: 0.25 }}>
        <Typography
          component="span"
          sx={{ fontSize: '10px', color: '#26A69A', mt: 0.3, flexShrink: 0 }}
        >
          ●
        </Typography>
        <Typography
          variant="caption"
          sx={{
            fontSize: '11px',
            color: 'text.secondary',
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {str}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ mb: 0.5 }}>
      <Box
        onClick={handleToggle}
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 0.75,
          py: 0.25,
          cursor: 'pointer',
          userSelect: 'none',
          '&:hover': { opacity: 0.85 },
        }}
      >
        <Typography
          component="span"
          sx={{ fontSize: '10px', color: '#26A69A', mt: 0.3, flexShrink: 0 }}
        >
          ●
        </Typography>
        <Typography
          variant="caption"
          sx={{
            flex: 1,
            fontSize: '11px',
            color: 'text.secondary',
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {collapsedText}
        </Typography>
        <ExpandMoreIcon
          sx={{
            fontSize: 16,
            color: 'text.secondary',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s',
            mt: 0.15,
          }}
        />
      </Box>
      <Collapse in={expanded}>
        <Typography
          component="pre"
          sx={{
            fontSize: '11px',
            lineHeight: 1.5,
            color: 'text.secondary',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            m: 0,
            mt: 0.5,
            ml: 1.75,
            p: 0.75,
            bgcolor: 'action.hover',
            borderRadius: 1,
          }}
        >
          {str}
        </Typography>
      </Collapse>
    </Box>
  );
});

/**
 * Thinking step message - Routes to appropriate component based on type
 */
const ThinkingStepMessage = memo(function ThinkingStepMessage({ step, mode }) {
  const eventType = step.event || step.type;
  const data = step.data || step;
  const message = step.message || data.message || data.description || '';
  
  // Tool calls use the card component
  const isToolEvent = eventType === 'tool_start' || eventType === 'tool_end';
  
  // LLM thinking - show as chip with spinner
  const isLlmThinking = eventType === 'llm_thinking';
  
  // Log/status events use simple bullet format
  const isLogEvent = eventType === 'log' || eventType === 'status';
  
  if (isToolEvent) {
    return <ToolCallCard step={step} mode={mode} />;
  }
  
  if (isLlmThinking) {
    // LLM generating chip with spinner
    return (
      <Box
        sx={{
          mb: 0.75,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 0.75,
          py: 0.5,
          px: 1,
          borderRadius: 1,
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
        }}
      >
        <CircularProgress size={12} thickness={4} />
        <Typography variant="body2" sx={{ fontSize: '12px', color: 'text.primary' }}>
          {message || 'Thinking...'}
        </Typography>
      </Box>
    );
  }
  
  if (isLogEvent) {
    return <LogMessage message={message} />;
  }
  
  // Plain thinking message with AI icon (strategic thinking)
  if (!message) return null;
  
  return (
    <Box
      sx={{
        mb: 0.75,
        display: 'flex',
        alignItems: 'flex-start',
        gap: 0.75,
        py: 0.25,
      }}
    >
      {/* AI thinking icon */}
      <Box
        sx={{
          width: 16,
          height: 16,
          borderRadius: '50%',
          bgcolor: 'primary.main',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          mt: 0.25,
        }}
      >
        <Typography sx={{ fontSize: '9px', color: 'primary.contrastText' }}>✦</Typography>
      </Box>
      
      {/* Message text */}
      <Typography
        variant="body2"
        sx={{
          fontSize: '12px',
          color: 'text.primary',
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {message}
      </Typography>
    </Box>
  );
});

/**
 * Thinking steps block - rendered in the chat timeline after the user message.
 * Collapsed: header only. Expanded: full list of steps.
 */
const ThinkingStepsBlock = memo(function ThinkingStepsBlock({ block, mode, onToggle }) {
  const { expanded = false, status, steps } = block || {};
  const isRunning = status === 'running';
  const hasSteps = Array.isArray(steps) && steps.length > 0;

  return (
    <Box data-thinking-block-id={block?.id} sx={{ mb: 2, display: 'flex', justifyContent: 'flex-start' }}>
      <Paper
        elevation={0}
        sx={{
          p: 0,
          maxWidth: '100%',
          width: '100%',
          bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          overflow: 'hidden',
        }}
      >
        {/* Header row */}
        <Box
          onClick={onToggle}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            px: 1.5,
            py: 0.75,
            cursor: 'pointer',
            userSelect: 'none',
            '&:hover': { bgcolor: 'action.hover' },
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <ExpandMoreIcon
              fontSize="small"
              sx={{
                transform: expanded ? 'rotate(0deg)' : 'rotate(-90deg)',
                transition: 'transform 0.2s',
                color: 'text.secondary',
                fontSize: 18,
              }}
            />
            <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ fontSize: '12px' }}>
              Thinking steps
            </Typography>
          </Box>

          {isRunning && <CircularProgress size={14} thickness={4} />}
        </Box>

        {/* Expanded content */}
        <Collapse in={expanded}>
          <Box sx={{ px: 1.5, pb: 1.25, pt: 0.5 }}>
            {hasSteps ? (
              steps.map((step, idx) => (
                <ThinkingStepMessage key={step.id || idx} step={step} mode={mode} />
              ))
            ) : (
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '11px' }}>
                No steps yet.
              </Typography>
            )}
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
});

/**
 * Message bubble component with markdown and mermaid diagram support
 */
const MessageBubble = memo(function MessageBubble({ 
  message, 
  mode, 
  onQuickFixDiagram, 
  onRegenerate, 
  isLastAssistant,
  activeCapability,
  messageIdx,
  fixingDiagram, // { messageIdx, blockIndex } or null
}) {
  const isUser = message.role === 'user';
  const isError = message.isError;
  const messageCapability = message.capability;
  const [copyState, setCopyState] = useState(null); // null | 'success' | 'error'

  const copyToClipboard = useCallback(async (text) => {
    // Prefer modern Clipboard API when available (requires secure context)
    try {
      if (navigator?.clipboard?.writeText && window?.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch {
      // Fall back to legacy method below
    }

    // Legacy fallback: works on many non-secure contexts (HTTP)
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.top = '0';
      textarea.style.left = '0';
      textarea.style.width = '1px';
      textarea.style.height = '1px';
      textarea.style.opacity = '0';

      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();

      const ok = document.execCommand('copy');
      document.body.removeChild(textarea);
      return ok;
    } catch {
      return false;
    }
  }, []);

  const handleCopy = useCallback(async () => {
    const ok = await copyToClipboard(message.content);
    setCopyState(ok ? 'success' : 'error');
    setTimeout(() => setCopyState(null), 2000);
  }, [copyToClipboard, message.content]);

  // Custom code renderer that handles mermaid diagrams
  const codeComponent = useCallback((() => {
    let mermaidBlockIndex = 0;
    
    return ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      const codeString = String(children).replace(/\n$/, '');

      const renderInlineCode = () => {
        return (
          <code
            style={{
              backgroundColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)',
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              padding: '2px 6px',
              borderRadius: '4px',
              fontSize: '0.9em',
              fontFamily: '"Fira Code", "Courier New", monospace',
              whiteSpace: 'pre-wrap',
              display: 'inline',
              verticalAlign: 'baseline',
            }}
            {...props}
          >
            {children}
          </code>
        );
      };
      
      // Render Mermaid diagrams
      if (language === 'mermaid') {
        const currentBlockIndex = mermaidBlockIndex++;
        const isFixingThis = fixingDiagram?.messageIdx === messageIdx && 
                            fixingDiagram?.blockIndex === currentBlockIndex;
        return (
          <Box
            data-mermaid-anchor="true"
            data-message-idx={String(messageIdx)}
            data-block-idx={String(currentBlockIndex)}
            sx={{ scrollMarginTop: '5rem' }}
          >
            <MermaidDiagram 
              chart={codeString} 
              mode={mode}
              blockIndex={currentBlockIndex}
              onQuickFix={onQuickFixDiagram ? (errorInfo) => onQuickFixDiagram(messageIdx, errorInfo) : undefined}
              isFixing={isFixingThis}
            />
          </Box>
        );
      }
      
      // Regular code block
      if (!inline) {
        const isSingleLine = !codeString.includes('\n');

        // Match DeepWiki main view behavior: single-line code blocks stay inline
        if (isSingleLine) {
          return renderInlineCode();
        }

        return (
          <Box
            component="pre"
            sx={{
              bgcolor: mode === 'dark' ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.05)',
              p: 1.5,
              borderRadius: 1,
              overflow: 'auto',
              fontSize: '0.75rem',
              my: 1,
            }}
          >
            <code
              style={{
                fontFamily: '"Fira Code", "Courier New", monospace',
                fontSize: '0.75rem',
              }}
              {...props}
            >
              {children}
            </code>
          </Box>
        );
      }
      
      // Inline code
      return renderInlineCode();
    };
  })(), [mode, messageIdx, onQuickFixDiagram, fixingDiagram]);

  return (
    <Box
      sx={{
        mb: 2,
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <Paper
        elevation={1}
        sx={{
          p: 1.5,
          // User messages: constrained width, right-aligned
          // Assistant messages: full width for better content display
          maxWidth: isUser ? '85%' : '100%',
          width: isUser ? 'auto' : '100%',
          bgcolor: isUser
            ? 'primary.dark'
            : isError
              ? 'error.dark'
              : mode === 'dark'
                ? 'background.secondary'
                : 'grey.100',
          borderRadius: 2,
          fontSize: '0.875rem',
          lineHeight: 1.6,
          '& p': { 
            m: 0, 
            mb: 1, 
            '&:last-child': { mb: 0 },
          },
          '& h1, & h2, & h3, & h4': {
            fontSize: '1rem',
            fontWeight: 600,
            mt: 1.5,
            mb: 0.5,
            '&:first-of-type': { mt: 0 },
          },
          '& ul, & ol': {
            pl: 2.5,
            my: 1,
            '& li': {
              mb: 0.5,
              pl: 0.5,
            },
          },
          '& ul': {
            listStyleType: 'disc',
          },
          // Table styles
          '& table': {
            borderCollapse: 'collapse',
            width: '100%',
            my: 1,
            fontSize: '0.8rem',
          },
          '& th, & td': {
            border: '1px solid',
            borderColor: 'divider',
            p: 1,
            textAlign: 'left',
          },
          '& th': {
            bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
            fontWeight: 600,
          },
        }}
      >
        {messageCapability && (
          <Box
            sx={{
              display: 'flex',
              justifyContent: isUser ? 'flex-end' : 'flex-start',
              mb: 0.75,
              userSelect: 'none',
            }}
          >
            <Chip
              size="small"
              variant="outlined"
              label={messageCapability === 'research' ? 'RESEARCH' : 'ASK'}
              sx={{
                height: 20,
                opacity: 0.8,
                userSelect: 'none',
                pointerEvents: 'none',
                '& .MuiChip-label': {
                  px: 0.75,
                  fontSize: '0.65rem',
                  fontWeight: 600,
                  letterSpacing: '0.04em',
                },
              }}
            />
          </Box>
        )}

        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code: codeComponent,
            // Override pre to just pass through children (code component handles styling)
            pre: ({ children }) => <>{children}</>,
          }}
        >
          {message.content}
        </ReactMarkdown>

        {/* Sources - Show file paths */}
        {message.sources?.length > 0 && (
          <Box sx={{ mt: 1.5, pt: 1, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Sources:
            </Typography>
            <Stack spacing={0.5} sx={{ mt: 0.5 }}>
              {message.sources.slice(0, 8).map((src, i) => (
                <Box 
                  key={i} 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    gap: 0.5,
                    fontSize: '0.7rem',
                    color: 'text.secondary',
                  }}
                >
                  <CodeIcon sx={{ fontSize: '12px', opacity: 0.7 }} />
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      fontFamily: 'monospace',
                      fontSize: '0.7rem',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    [{src.index}] {src.source}
                    {src.symbol && <span style={{ opacity: 0.7 }}> :: {src.symbol}</span>}
                  </Typography>
                </Box>
              ))}
              {message.sources.length > 8 && (
                <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.65rem' }}>
                  +{message.sources.length - 8} more sources
                </Typography>
              )}
            </Stack>
          </Box>
        )}

        {/* Action buttons for assistant messages */}
        {!isUser && (
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: 0.5,
              mt: 1,
              pt: 1,
              borderTop: '1px solid',
              borderColor: 'divider',
            }}
          >
            {isLastAssistant && onRegenerate && (
              <Tooltip title={`Regenerate (${activeCapability === 'research' ? 'Research' : 'Ask'})`}>
                <IconButton
                  size="small"
                  onClick={onRegenerate}
                  sx={{
                    color: 'text.secondary',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  <RegenerateIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip
              title={
                copyState === 'success'
                  ? 'Copied markdown!'
                  : copyState === 'error'
                    ? 'Copy failed'
                    : 'Copy markdown'
              }
            >
              <IconButton
                size="small"
                onClick={handleCopy}
                sx={{
                  color:
                    copyState === 'success'
                      ? 'success.main'
                      : copyState === 'error'
                        ? 'error.main'
                        : 'text.secondary',
                  '&:hover': {
                    color:
                      copyState === 'success'
                        ? 'success.main'
                        : copyState === 'error'
                          ? 'error.main'
                          : 'primary.main',
                  },
                }}
              >
                <CopyIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Tooltip>
          </Box>
        )}
      </Paper>
    </Box>
  );
});

/**
 * Main ChatDrawer component - slides from right edge
 * Uses Socket.IO with test_toolkit_tool event (same as generate_wiki)
 */
const ChatDrawer = memo(function ChatDrawer({
  projectId,
  toolkitId,
  toolkit,  // Full toolkit object with settings
  wikiGenerated,
  repoIdentifierOverride,
  analysisKeyOverride,
  open,
  onClose,
  onOpen,
  mode = 'dark',
  authHeaders = {},
}) {
  // Storage key for persisting messages
  const storageKey = useMemo(
    () => `deepwiki-chat-${projectId}-${toolkitId}`,
    [projectId, toolkitId]
  );

  // Storage key for the last capability used to generate a response
  // (derived from actual requests, not from UI toggle flips)
  const capabilityStorageKey = useMemo(
    () => `deepwiki-chat-capability-${projectId}-${toolkitId}`,
    [projectId, toolkitId]
  );

  // Load persisted messages on mount
  const loadPersistedMessages = useCallback(() => {
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        return Array.isArray(parsed) ? parsed : [];
      }
    } catch (err) {
      console.warn('[ChatDrawer] Failed to load persisted messages:', err);
    }
    return [];
  }, [storageKey]);

  const [chatMode, setChatMode] = useState(() => {
    try {
      const stored = localStorage.getItem(capabilityStorageKey);
      return stored === 'research' ? 'research' : 'ask';
    } catch {
      return 'ask';
    }
  }); // 'ask' | 'research'
  const [messages, setMessages] = useState(loadPersistedMessages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [error, setError] = useState(null);
  const [pendingAnswer, setPendingAnswer] = useState('');
  const [drawerWidth, setDrawerWidth] = useState(DEFAULT_DRAWER_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const [researchTodos, setResearchTodos] = useState([]); // For deep research todo tracking
  const [fixingDiagram, setFixingDiagram] = useState(null); // { messageIdx, blockIndex }
  const pendingScrollToDiagramRef = useRef(null); // { messageIdx, blockIndex }
  const pendingCapabilityRef = useRef(null); // 'ask' | 'research' | null
  const capabilityStorageKeyRef = useRef(capabilityStorageKey);
  const messagesRef = useRef(messages);
  const currentThinkingBlockIdRef = useRef(null);
  const prevMessagesLengthRef = useRef(messages.length);
  const suppressNextAutoScrollRef = useRef(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  
  // Track current request
  const currentStreamIdRef = useRef(null);
  const currentMessageIdRef = useRef(null);
  const pendingQuestionRef = useRef(null);

  useEffect(() => {
    capabilityStorageKeyRef.current = capabilityStorageKey;
  }, [capabilityStorageKey]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const persistLastCapability = useCallback((capability) => {
    try {
      if (capability === 'ask' || capability === 'research') {
        localStorage.setItem(capabilityStorageKeyRef.current, capability);
      }
    } catch (err) {
      console.warn('[ChatDrawer] Failed to persist capability:', err);
    }
  }, []);

  const loadPersistedCapability = useCallback(() => {
    try {
      const stored = localStorage.getItem(capabilityStorageKeyRef.current);
      return stored === 'research' ? 'research' : stored === 'ask' ? 'ask' : null;
    } catch {
      return null;
    }
  }, []);

  const getLastAssistantCapability = useCallback((msgs) => {
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i];
      if (m?.role === 'assistant' && (m.capability === 'ask' || m.capability === 'research')) {
        return m.capability;
      }
    }
    return null;
  }, []);

  // When the drawer opens, restore capability from last generated response
  const prevOpenRef = useRef(open);
  useEffect(() => {
    if (open && !prevOpenRef.current) {
      const last = getLastAssistantCapability(messagesRef.current);
      const persisted = loadPersistedCapability();
      const next = last || persisted;
      if (next && next !== chatMode) {
        setChatMode(next);
      }
    }
    prevOpenRef.current = open;
  }, [open, chatMode, getLastAssistantCapability, loadPersistedCapability]);

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        // Persist chat timeline items (limit to last 50 for storage)
        // Include thinking_steps blocks even though they don't have `content`.
        const messagesToStore = messages
          .filter(m => m?.type === 'thinking_steps' || (!m?.isError && m?.content))
          .slice(-50)
          .map(m => {
            if (m?.type !== 'thinking_steps') return m;

            const steps = Array.isArray(m.steps) ? m.steps.slice(-MAX_THINKING_STEPS_PER_RUN) : [];
            return {
              ...m,
              steps,
            };
          });
        localStorage.setItem(storageKey, JSON.stringify(messagesToStore));
      } catch (err) {
        console.warn('[ChatDrawer] Failed to persist messages:', err);
      }
    }
  }, [messages, storageKey]);

  // Handle resize
  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    setIsResizing(true);
    
    const startX = e.clientX;
    const startWidth = drawerWidth;
    
    const handleMouseMove = (moveEvent) => {
      const diff = startX - moveEvent.clientX;
      const newWidth = Math.min(MAX_DRAWER_WIDTH, Math.max(MIN_DRAWER_WIDTH, startWidth + diff));
      setDrawerWidth(newWidth);
    };
    
    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [drawerWidth]);

  // Handle socket messages for Ask tool
  const handleAskSocketMessage = useCallback((data) => {
    const { content, type: socketMessageType, response_metadata } = data || {};

    // Check if this message is for our current stream
    const messageStreamId = response_metadata?.stream_id;
    if (messageStreamId && currentStreamIdRef.current && messageStreamId !== currentStreamIdRef.current) {
      return; // Ignore messages from other streams
    }

    console.log('[ChatDrawer] Socket message:', socketMessageType, data);

    switch (socketMessageType) {
      case SocketMessageType.AgentThinkingStep:
      case SocketMessageType.AgentThinkingStepUpdate: {
        // Handle structured events from response_metadata.message
        const messageContent = response_metadata?.message;
        
        // Try to parse as JSON first (structured event)
        let parsedEvent = null;
        if (messageContent) {
          try {
            parsedEvent = typeof messageContent === 'string' ? JSON.parse(messageContent) : messageContent;
          } catch (e) {
            // Not JSON, treat as plain message
            parsedEvent = null;
          }
        }
        
        const activeThinkingBlockId = currentThinkingBlockIdRef.current;

        const updateThinkingBlock = (updater) => {
          if (!activeThinkingBlockId) return;
          setMessages(prev =>
            prev.map(m => {
              if (m?.type === 'thinking_steps' && m?.id === activeThinkingBlockId) {
                return updater(m);
              }
              return m;
            })
          );
        };

        const appendStep = (newStep) => {
          updateThinkingBlock(block => {
            const prevSteps = Array.isArray(block.steps) ? block.steps : [];
            const nextSteps = [...prevSteps, newStep].slice(-MAX_THINKING_STEPS_PER_RUN);
            return { ...block, steps: nextSteps };
          });
        };

        if (parsedEvent && parsedEvent.event) {
          // Handle structured events
          const { event, data: eventData } = parsedEvent;
          
          switch (event) {
            case 'todo_update': {
              // Update todos from structured event
              const todos = eventData?.items || eventData?.todos || eventData || [];
              setResearchTodos(Array.isArray(todos) ? todos : []);
              break;
            }
            case 'tool_start': {
              // Tool starting - add with in-progress status
              appendStep({
                id: eventData?.id || `tool-${Date.now()}`,
                event,
                data: {
                  ...eventData,
                  tool: eventData?.tool || eventData?.toolName,
                  input: eventData?.input || eventData?.args || '',
                  output: '',
                },
                message: eventData?.description || `Calling: ${eventData?.tool || 'tool'}`,
              });
              break;
            }
            case 'tool_end': {
              // Tool completed - merge output into existing tool_start card
              updateThinkingBlock(block => {
                const prevSteps = Array.isArray(block.steps) ? block.steps : [];
                const toolId = eventData?.id;
                const existingIdx = prevSteps.findIndex(s => s.id === toolId || s.data?.id === toolId);

                if (existingIdx >= 0) {
                  const updated = [...prevSteps];
                  const existing = updated[existingIdx];
                  updated[existingIdx] = {
                    ...existing,
                    event: 'tool_end',
                    data: {
                      ...existing.data,
                      ...eventData,
                      input: existing.data?.input || eventData?.input || '',
                      output: eventData?.output || eventData?.result || '',
                      status: 'completed',
                    },
                  };
                  return { ...block, steps: updated.slice(-MAX_THINKING_STEPS_PER_RUN) };
                }

                return {
                  ...block,
                  steps: [...prevSteps, {
                    id: toolId || `tool-${Date.now()}`,
                    event: 'tool_end',
                    data: {
                      ...eventData,
                      tool: eventData?.tool || eventData?.toolName,
                      output: eventData?.output || eventData?.result || '',
                      status: 'completed',
                    },
                    message: eventData?.description || 'Tool completed',
                  }].slice(-MAX_THINKING_STEPS_PER_RUN),
                };
              });
              break;
            }
            case 'thinking': {
              // Plain thinking step with AI icon
              appendStep({
                id: eventData?.id || `think-${Date.now()}`,
                event,
                data: eventData,
                message: eventData?.message || eventData?.title || '',
              });
              break;
            }
            case 'llm_thinking': {
              // LLM is generating - show as a chip with spinner
              // Replace any previous llm_thinking step (only show latest)
              updateThinkingBlock(block => {
                const prevSteps = Array.isArray(block.steps) ? block.steps : [];
                const filtered = prevSteps.filter(s => s.event !== 'llm_thinking');
                return {
                  ...block,
                  steps: [...filtered, {
                    id: eventData?.id || `llm-${Date.now()}`,
                    event,
                    data: eventData,
                    message: eventData?.message || 'Thinking...',
                  }].slice(-MAX_THINKING_STEPS_PER_RUN),
                };
              });
              break;
            }
            case 'status':
            case 'log': {
              // Add status/log as plain thinking step
              appendStep({
                id: `${event}-${Date.now()}`,
                event,
                data: eventData,
                message: eventData?.message || '',
              });
              break;
            }
            default: {
              // Unknown structured event, add as generic step
              appendStep({
                id: `${event}-${Date.now()}`,
                event,
                data: eventData,
                message: eventData?.message || eventData?.title || JSON.stringify(eventData),
              });
            }
          }
        } else {
          // Plain text message (legacy format)
          const stepMessage = messageContent || 
            (content && typeof content === 'object' ? content.message : content) || 
            'Processing...';
          appendStep({
            id: `step-${Date.now()}`,
            event: 'log',
            message: stepMessage,
            type: socketMessageType,
          });
        }
        break;
      }

      // Handle todo_update events from deep research (direct socket type)
      case 'todo_update': {
        const todos = content?.todos || content || [];
        setResearchTodos(Array.isArray(todos) ? todos : []);
        break;
      }

      case SocketMessageType.AgentResponse: {
        // Ask completed - extract answer from content
        console.log('[ChatDrawer] Ask complete, content:', content);
        
        let answer = '';
        let sources = [];
        let isError = false;
        
        // Content could be:
        // 1. A string (direct answer)
        // 2. An array of result objects from platform: [{ object_type, result_target, data }]
        // 3. An object with answer/result fields
        if (typeof content === 'string') {
          // Try to parse if it looks like JSON
          if (content.startsWith('[') || content.startsWith('{')) {
            try {
              const parsed = JSON.parse(content);
              if (Array.isArray(parsed)) {
                // Platform result format: array of result objects
                const messageObj = parsed.find(obj => obj.object_type === 'message');
                if (messageObj) {
                  answer = messageObj.data || '';
                  // Only mark as error if success is explicitly false
                  isError = messageObj.success === false;
                }
              } else {
                answer = parsed.answer || parsed.result || parsed.message || parsed.data || content;
                sources = parsed.sources || [];
                isError = parsed.success === false;
              }
            } catch {
              answer = content;
            }
          } else {
            answer = content;
          }
        } else if (Array.isArray(content)) {
          // Platform result format: array of result objects
          const messageObj = content.find(obj => obj.object_type === 'message');
          if (messageObj) {
            answer = messageObj.data || '';
            // Only mark as error if success is explicitly false
            isError = messageObj.success === false;
          }
        } else if (typeof content === 'object' && content !== null) {
          // Try to extract answer from various possible fields
          answer = content.answer || content.result || content.message || content.data || JSON.stringify(content);
          sources = content.sources || [];
        }

        const capabilityUsed = pendingCapabilityRef.current || 'ask';

        // Mark thinking block completed for this request
        if (currentThinkingBlockIdRef.current) {
          const finishedId = currentThinkingBlockIdRef.current;
          setMessages(prev =>
            prev.map(m => {
              if (m?.type === 'thinking_steps' && m?.id === finishedId) {
                return { ...m, status: 'completed' };
              }
              return m;
            })
          );
          currentThinkingBlockIdRef.current = null;
        }

        // Add assistant response
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: answer || 'Response received',
            sources,
            isError,
            capability: capabilityUsed,
          },
        ]);

        // Sync toggle + persist based on what actually generated the last response
        persistLastCapability(capabilityUsed);
        setChatMode(capabilityUsed);
        pendingCapabilityRef.current = null;
        
        setIsLoading(false);
        currentStreamIdRef.current = null;
        currentMessageIdRef.current = null;
        socketUnsubscribeRef.current?.();
        break;
      }

      case SocketMessageType.Chunk:
      case SocketMessageType.AIMessageChunk:
      case SocketMessageType.AgentLlmChunk: {
        // Streaming chunk - accumulate
        if (content && typeof content === 'string') {
          setPendingAnswer(prev => prev + content);
        }
        break;
      }

      case SocketMessageType.AgentToolError:
      case SocketMessageType.AgentException:
      case SocketMessageType.Error:
      case SocketMessageType.LlmError: {
        const errorMsg = typeof content === 'string' 
          ? content 
          : content?.message || content?.error || 'An error occurred';
        
        console.error('[ChatDrawer] Error:', errorMsg);
        const capabilityUsed = pendingCapabilityRef.current || 'ask';

        if (currentThinkingBlockIdRef.current) {
          const finishedId = currentThinkingBlockIdRef.current;
          setMessages(prev =>
            prev.map(m => {
              if (m?.type === 'thinking_steps' && m?.id === finishedId) {
                return { ...m, status: 'completed' };
              }
              return m;
            })
          );
          currentThinkingBlockIdRef.current = null;
        }
        setError(errorMsg);
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: `Sorry, I encountered an error: ${errorMsg}`,
            isError: true,
            capability: capabilityUsed,
          },
        ]);

        persistLastCapability(capabilityUsed);
        setChatMode(capabilityUsed);
        pendingCapabilityRef.current = null;
        
        setIsLoading(false);
        currentStreamIdRef.current = null;
        currentMessageIdRef.current = null;
        socketUnsubscribeRef.current?.();
        break;
      }

      default:
        console.log('[ChatDrawer] Unhandled message type:', socketMessageType);
    }
  }, []);

  // Socket setup for test_toolkit_tool event (same as generate_wiki)
  const { subscribe: socketSubscribe, unsubscribe: socketUnsubscribe, emit: socketEmit } = 
    useManualSocket(sioEvents.test_toolkit_tool, handleAskSocketMessage);

  // Store unsubscribe ref for cleanup
  const socketUnsubscribeRef = useRef(socketUnsubscribe);
  useEffect(() => {
    socketUnsubscribeRef.current = socketUnsubscribe;
  }, [socketUnsubscribe]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      socketUnsubscribeRef.current?.();
    };
  }, []);

  // Scroll to bottom on new messages
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    const pending = pendingScrollToDiagramRef.current;

    if (pending && messagesContainerRef.current) {
      const { messageIdx, blockIndex } = pending;
      const selector = `[data-mermaid-anchor="true"][data-message-idx="${messageIdx}"][data-block-idx="${blockIndex}"]`;

      const doScroll = () => {
        const el = messagesContainerRef.current?.querySelector(selector);
        el?.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
      };

      // Scroll immediately and once again after Mermaid re-renders
      doScroll();
      setTimeout(doScroll, 250);

      pendingScrollToDiagramRef.current = null;
      return;
    }

    if (suppressNextAutoScrollRef.current) {
      suppressNextAutoScrollRef.current = false;
      prevMessagesLengthRef.current = messages.length;
      return;
    }

    const prevLen = prevMessagesLengthRef.current;
    const nextLen = messages.length;
    prevMessagesLengthRef.current = nextLen;

    // Avoid jumping when only toggling UI state (e.g. expanding/collapsing thinking steps).
    // Auto-scroll only when new messages are appended, or when streaming while already near bottom.
    if (nextLen > prevLen) {
      scrollToBottom();
      return;
    }

    if (isLoading && messagesContainerRef.current) {
      const el = messagesContainerRef.current;
      const distanceToBottom = el.scrollHeight - (el.scrollTop + el.clientHeight);
      if (distanceToBottom < SCROLL_NEAR_BOTTOM_THRESHOLD_PX) {
        scrollToBottom();
      }
    }
  }, [messages, isLoading, scrollToBottom]);

  // Handle mode toggle
  const handleModeChange = useCallback((event, newMode) => {
    if (newMode) {
      setChatMode(newMode);
    }
  }, []);

  // Send message via Socket.IO (same pattern as generate_wiki)
  const handleSend = useCallback(() => {
    if (!input.trim() || isLoading || !toolkit || !projectId || !toolkitId) return;

    const question = input.trim();
    const capability = chatMode;
    pendingCapabilityRef.current = capability;
    pendingQuestionRef.current = question;

    // Add user message + thinking steps block to UI
    const thinkingBlockId = uuidv4();
    currentThinkingBlockIdRef.current = thinkingBlockId;

    const userMessage = { role: 'user', content: question, capability };
    const thinkingBlock = {
      type: 'thinking_steps',
      id: thinkingBlockId,
      status: 'running',
      expanded: false,
      steps: [],
      capability,
    };

    setMessages(prev => [...prev, userMessage, thinkingBlock]);
    setInput('');
    setIsLoading(true);
    setError(null);
    setPendingAnswer('');
    setResearchTodos([]); // Reset todos for research mode

    try {
      const originalSettings = toolkit.settings || {};
      const llmModel = originalSettings.toolkit_configuration_llm_model;
      const maxTokens = originalSettings.toolkit_configuration_max_tokens;

      if (!llmModel) {
        throw new Error('Toolkit settings missing llm_model. Configure it in toolkit settings first.');
      }

      const toolkitName = toolkit.toolkit_name || toolkit.name || String(toolkitId);

      // Generate unique IDs for tracking
      const streamId = uuidv4();
      const messageId = uuidv4();
      currentStreamIdRef.current = streamId;
      currentMessageIdRef.current = messageId;

      console.log('[ChatDrawer] Sending ask via Socket.IO:', { streamId, messageId, question });

      // Subscribe to socket events before emitting
      socketSubscribe();

      // Determine tool based on chat mode
      const toolName = chatMode === 'research' ? 'deep_research' : 'ask';

      const buildChatHistory = (msgs) => {
        const base = Array.isArray(msgs) ? msgs : [];
        return base
          .filter(m => (m?.role === 'user' || m?.role === 'assistant') && typeof m?.content === 'string')
          .slice(-6)
          .map(m => ({ role: m.role, content: m.content }));
      };

      // Build socket payload (matching generate_wiki pattern)
      const payload = {
        project_id: Number(projectId),
        stream_id: streamId,
        message_id: messageId,
        toolkit_config: {
          type: toolkit.type,
          toolkit_name: toolkitName,
          toolkit_id: Number(toolkitId),
          settings: originalSettings,
        },
        tool_name: toolName,
        tool_params: { 
          question,
          chat_history: buildChatHistory(messagesRef.current),
          ...(repoIdentifierOverride ? { repo_identifier_override: repoIdentifierOverride } : {}),
          ...(analysisKeyOverride ? { analysis_key_override: analysisKeyOverride } : {}),
          ...(chatMode === 'research' ? { research_type: 'general', enable_subagents: true } : {}),
        },
        llm_model: llmModel,
        llm_settings: {
          max_tokens: maxTokens || 4096,
          model_name: llmModel,
        },
      };

      // Emit the socket event
      const emitResult = socketEmit(payload);
      console.log('[ChatDrawer] Socket emit result:', emitResult);

      if (!emitResult) {
        throw new Error('Failed to emit socket event - socket may not be connected');
      }

    } catch (err) {
      console.error('[ChatDrawer] Error sending ask:', err);
      setError(err.message);
      // Remove the running thinking block if the request failed before streaming started.
      setMessages(prev => prev.filter(m => !(m?.type === 'thinking_steps' && m?.id === thinkingBlockId)));
      currentThinkingBlockIdRef.current = null;
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${err.message}`,
          isError: true,
          capability,
        },
      ]);
      persistLastCapability(capability);
      setChatMode(capability);
      pendingCapabilityRef.current = null;
      setIsLoading(false);
    }
  }, [analysisKeyOverride, chatMode, input, isLoading, messages, persistLastCapability, projectId, repoIdentifierOverride, socketEmit, socketSubscribe, toolkit, toolkitId]);

  // Handle regenerate - re-send the last user message
  const handleRegenerate = useCallback(() => {
    if (isLoading || messages.length < 2) return;

    // Find last user message
    let lastUserMessage = null;
    let lastUserIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserMessage = messages[i];
        lastUserIndex = i;
        break;
      }
    }

    if (!lastUserMessage) return;

    // Remove all messages after (and including) the last user message, then re-add user message
    const messagesBeforeUser = messages.slice(0, lastUserIndex);
    
    // Set input to the last question and trigger send
    setMessages(messagesBeforeUser);
    setInput(lastUserMessage.content);
    
    // Use setTimeout to allow state update before sending
    setTimeout(() => {
      // Directly call handleSend logic with the question
      const question = lastUserMessage.content;
      const capability = chatMode;
      pendingCapabilityRef.current = capability;
      pendingQuestionRef.current = question;

      // Add user message to UI
      const thinkingBlockId = uuidv4();
      currentThinkingBlockIdRef.current = thinkingBlockId;

      const userMessage = { role: 'user', content: question, capability };
      const thinkingBlock = {
        type: 'thinking_steps',
        id: thinkingBlockId,
        status: 'running',
        expanded: false,
        steps: [],
        capability,
      };

      setMessages(prev => [...prev, userMessage, thinkingBlock]);
      setInput('');
      setIsLoading(true);
      setError(null);
      setPendingAnswer('');
      setResearchTodos([]);

      try {
        const originalSettings = toolkit.settings || {};
        const llmModel = originalSettings.toolkit_configuration_llm_model;
        const maxTokens = originalSettings.toolkit_configuration_max_tokens;

        if (!llmModel) {
          throw new Error('Toolkit settings missing llm_model.');
        }

        const toolkitName = toolkit.toolkit_name || toolkit.name || String(toolkitId);
        const streamId = uuidv4();
        const messageId = uuidv4();
        currentStreamIdRef.current = streamId;
        currentMessageIdRef.current = messageId;

        socketSubscribe();

        const toolName = chatMode === 'research' ? 'deep_research' : 'ask';

        const buildChatHistory = (msgs) => {
          const base = Array.isArray(msgs) ? msgs : [];
          return base
            .filter(m => (m?.role === 'user' || m?.role === 'assistant') && typeof m?.content === 'string')
            .slice(-6)
            .map(m => ({ role: m.role, content: m.content }));
        };

        const payload = {
          project_id: Number(projectId),
          stream_id: streamId,
          message_id: messageId,
          toolkit_config: {
            type: toolkit.type,
            toolkit_name: toolkitName,
            toolkit_id: Number(toolkitId),
            settings: originalSettings,
          },
          tool_name: toolName,
          tool_params: { 
            question,
            chat_history: buildChatHistory(messagesBeforeUser),
            ...(chatMode === 'research' ? { research_type: 'general', enable_subagents: true } : {}),
          },
          llm_model: llmModel,
          llm_settings: {
            max_tokens: maxTokens || 4096,
            model_name: llmModel,
          },
        };

        socketEmit(payload);
      } catch (err) {
        console.error('[ChatDrawer] Error regenerating:', err);
        setError(err.message);
        setMessages(prev => prev.filter(m => !(m?.type === 'thinking_steps' && m?.id === thinkingBlockId)));
        currentThinkingBlockIdRef.current = null;
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: `Error: ${err.message}`, isError: true, capability },
        ]);
        persistLastCapability(capability);
        setChatMode(capability);
        pendingCapabilityRef.current = null;
        setIsLoading(false);
      }
    }, 0);
  }, [isLoading, messages, toolkit, projectId, toolkitId, socketSubscribe, socketEmit, chatMode, persistLastCapability]);

  const handleToggleThinkingBlock = useCallback((blockId) => {
    if (!blockId) return;

    suppressNextAutoScrollRef.current = true;

    const container = messagesContainerRef.current;
    const selector = `[data-thinking-block-id="${blockId}"]`;
    const anchorEl = container?.querySelector?.(selector);

    const containerTop = container?.getBoundingClientRect?.().top;
    const beforeOffset =
      container && anchorEl && typeof containerTop === 'number'
        ? anchorEl.getBoundingClientRect().top - containerTop
        : null;

    setMessages(prev =>
      prev.map(m => {
        if (m?.type === 'thinking_steps' && m?.id === blockId) {
          return { ...m, expanded: !m.expanded };
        }
        return m;
      })
    );

    if (!container || beforeOffset === null || typeof containerTop !== 'number') return;

    requestAnimationFrame(() => {
      const nextAnchor = container.querySelector(selector);
      if (!nextAnchor) return;
      const afterOffset = nextAnchor.getBoundingClientRect().top - containerTop;
      container.scrollTop += afterOffset - beforeOffset;
    });
  }, []);

  // Helper: Extract mermaid code from LLM response
  const extractMermaidFromResponse = useCallback((response) => {
    // Look for mermaid code block
    const mermaidMatch = response.match(/```mermaid\n([\s\S]*?)\n```/);
    if (mermaidMatch) {
      return mermaidMatch[1].trim();
    }
    // Try without newlines
    const altMatch = response.match(/```mermaid([\s\S]*?)```/);
    if (altMatch) {
      return altMatch[1].trim();
    }
    // If no code block, assume the whole response is the diagram
    return response.trim();
  }, []);

  // Helper: Replace mermaid block in message content
  const replaceMermaidBlock = useCallback((content, blockIndex, newMermaidCode) => {
    const mermaidRegex = /```mermaid\n[\s\S]*?\n```/g;
    let currentIndex = 0;
    return content.replace(mermaidRegex, (match) => {
      if (currentIndex === blockIndex) {
        currentIndex++;
        return '```mermaid\n' + newMermaidCode + '\n```';
      }
      currentIndex++;
      return match;
    });
  }, []);

  // Handle quick fix for mermaid diagrams - fixes in place
  const handleQuickFixDiagram = useCallback(async (messageIdx, errorInfo) => {
    if (fixingDiagram || !toolkit || !projectId) return;

    setFixingDiagram({ messageIdx, blockIndex: errorInfo.blockIndex });
    pendingScrollToDiagramRef.current = {
      messageIdx: String(messageIdx),
      blockIndex: String(errorInfo.blockIndex),
    };

    try {
      const originalSettings = toolkit.settings || {};
      const llmModel = originalSettings.toolkit_configuration_llm_model;
      
      if (!llmModel) {
        throw new Error('No LLM model configured');
      }

      // Comprehensive prompt that teaches syntax rules for different diagram types
      const fixPrompt = `You are a diagram syntax fixer. Analyze and fix the diagram below.

## STEP 1: IDENTIFY DIAGRAM TYPE
First, identify what type of diagram this is:
- classDiagram (Mermaid)
- erDiagram (Mermaid)
- flowchart/graph (Mermaid)
- sequenceDiagram (Mermaid)
- Other (PlantUML, etc.)

## STEP 2: CHECK AND FIX COMMON ISSUES

### A) BRACKET AND QUOTE ERRORS
Look for mismatched brackets and quotes.

❌ WRONG:
  ["text[]]
  ["text"]"]
  (value())
  
✅ CORRECT:
  ["text[]"]
  ["text"]
  (value)

### B) RELATIONSHIP SYNTAX FOR classDiagram
Use cardinality in quotes with simple relationship symbols:

❌ WRONG (ER diagram syntax - not for classDiagram):
  ClassA ||--o{ ClassB : "label"
  ClassA }o--o{ ClassB : "label"

✅ CORRECT (classDiagram syntax):
  ClassA "1" --o "*" ClassB : label
  ClassA "1" --> "*" ClassB : label
  ClassA "0..1" --o "1..*" ClassB : label
  ClassA <|-- ClassB
  ClassA *-- ClassB
  ClassA o-- ClassB

Cardinality values: "1", "*", "0..1", "1..*", "0..*", "n"

### C) RELATIONSHIP SYNTAX FOR erDiagram
Use crow's foot notation WITHOUT quotes around cardinality:

❌ WRONG (classDiagram syntax - not for erDiagram):
  ENTITY1 "1" --o "*" ENTITY2 : label

✅ CORRECT (erDiagram syntax):
  ENTITY1 ||--o{ ENTITY2 : relationship_name
  ENTITY1 ||--|| ENTITY2 : relationship_name
  ENTITY1 }o--o{ ENTITY2 : relationship_name

Symbols: || (exactly one), o| (zero or one), }| (one or more), }o (zero or more)

### D) NODE DEFINITIONS FOR flowchart/graph

❌ WRONG:
  A["text[]] --> B
  A --> |"label"| B["missing quote]

✅ CORRECT:
  A["text[]"] --> B
  A --> |"label"| B["complete quote"]

### E) SUBGRAPH SYNTAX

❌ WRONG:
  subgraph Name[Label
  subgraph Name["Label"

✅ CORRECT:
  subgraph Name["Label"]
  end

### F) NOTES FOR classDiagram

❌ WRONG:
  note for ClassName "Unclosed note
  note for ClassName Unquoted note

✅ CORRECT:
  note for ClassName "Note text here"
  note "General note"

### G) PARTICIPANT SYNTAX FOR sequenceDiagram
Participants must use simple text for display names. Do NOT use brackets or parentheses in participant definitions.

❌ WRONG:
  participant API as API["api/v2/endpoint.py::ClassName.method"]
  participant DB as DB["Database session"]

✅ CORRECT:
  participant API as API: ClassName.method
  participant DB as Database Session

### H) RESERVED KEYWORDS AS PARTICIPANT IDs FOR sequenceDiagram
Do NOT use reserved keywords as participant IDs.

RESERVED KEYWORDS (case-insensitive):
  alt, else, opt, loop, par, and, critical, break, end, rect, note, over, participant, actor

❌ WRONG:
  participant OPT as Options Generator
  participant ALT as Alternative Service
  participant END as End Handler

✅ CORRECT (rename to avoid keywords):
  participant OPTGEN as Options Generator
  participant ALTSVC as Alternative Service
  participant ENDH as End Handler

### I) SPECIAL CHARACTERS IN sequenceDiagram MESSAGES
Avoid these characters in message text: < > { } [ ] "
They break parsing. Simplify or remove them.

❌ WRONG:
  Client->>API: PUT /... (body: {"toolkit_id": <id>})
  API-->>Client: 404 "Version doesn't exist"
  C->>API: GET /api/v2/configurations/{project_id}

✅ CORRECT:
  Client->>API: PUT request with toolkit_id
  API-->>Client: 404 Version does not exist
  C->>API: GET /api/v2/configurations/project_id

### J) ALT/OPT/LOOP BLOCKS IN sequenceDiagram
Every alt, opt, loop, par, critical block MUST have a matching "end".
Nested blocks each need their OWN "end" statement.

❌ WRONG (missing end for outer alt):
  alt condition1
      A->>B: message1
  else condition2
      B->>C: message2
      alt nested_condition
          C->>D: message3
      end

✅ CORRECT:
  alt condition1
      A->>B: message1
  else condition2
      B->>C: message2
      alt nested_condition
          C->>D: message3
      end
  end

## STEP 3: VALIDATION CHECKLIST
Before returning, verify:

For ALL diagrams:
[ ] All brackets [] {} () are properly paired
[ ] All quotes "" are properly closed
[ ] No HTML tags in text

For sequenceDiagram:
[ ] No participant ID is a reserved keyword
[ ] All participants use simple text (no brackets [] or parentheses ())
[ ] All participant IDs in messages match their declarations exactly
[ ] All messages avoid special characters < > { } [ ] "
[ ] Every alt/opt/loop/par has a matching "end"

For classDiagram:
[ ] Relationships use correct syntax with quoted cardinality
[ ] Notes are properly quoted

For flowchart/graph:
[ ] All subgraphs have matching "end" statements
[ ] Node labels are properly quoted

## STEP 4: OUTPUT FORMAT
Return ONLY the complete fixed diagram in a mermaid code block. No explanations.

---

ERROR MESSAGE:
${errorInfo.message}

DIAGRAM TO FIX:
\`\`\`mermaid
${errorInfo.mermaidCode}
\`\`\``;

      const url = `/api/v2/elitea_core/predict_llm/prompt_lib/${projectId}`;

      const socketId = getSocketId();
      if (!socketId) {
        throw new Error('Socket not connected. Please refresh the page.');
      }

      const streamId = uuidv4();
      const messageId = uuidv4();

      const requestBody = {
        user_input: fixPrompt,
        chat_history: [],
        sid: socketId,
        await_task_timeout: 0, // Non-blocking streaming mode
        stream_id: streamId,
        message_id: messageId,
        llm_settings: {
          model_name: llmModel,
          max_tokens: 4096,
          temperature: 1.0,
        },
      };

      // Use streaming mode: collect response via socket.io events
      const llmResponse = await new Promise((resolve, reject) => {
        const socket = getSocket();
        let collectedContent = '';
        let resolved = false;

        const cleanup = () => {
          if (!resolved) resolved = true;
          socket.off(sioEvents.application_predict, handler);
        };

        const timeoutId = setTimeout(() => {
          cleanup();
          reject(new Error('LLM request timed out after 90 seconds'));
        }, 90000);

        let fallbackTimeoutId = null;

        const handler = (data) => {
          if (data.stream_id !== streamId) return;

          const toStr = (c) => {
            if (c == null) return '';
            return typeof c === 'string' ? c : JSON.stringify(c);
          };

          switch (data.type) {
            case SocketMessageType.Chunk:
            case SocketMessageType.AIMessageChunk:
            case SocketMessageType.AgentLlmChunk:
              if (data.content) collectedContent += toStr(data.content);
              break;
            case SocketMessageType.AgentResponse: {
              // Final complete response — authoritative, replaces accumulated chunks
              clearTimeout(timeoutId);
              if (fallbackTimeoutId) clearTimeout(fallbackTimeoutId);
              cleanup();
              const final = toStr(data.content);
              resolve(final || collectedContent);
              break;
            }
            case SocketMessageType.AgentLlmEnd:
              // AgentLlmEnd fires BEFORE AgentResponse in the platform event sequence.
              // Don't resolve here — wait for AgentResponse which has the complete content.
              // Start a short fallback timeout in case AgentResponse never arrives.
              fallbackTimeoutId = setTimeout(() => {
                clearTimeout(timeoutId);
                cleanup();
                resolve(collectedContent);
              }, 10000);
              break;
            case SocketMessageType.Error:
            case SocketMessageType.LlmError:
              clearTimeout(timeoutId);
              if (fallbackTimeoutId) clearTimeout(fallbackTimeoutId);
              cleanup();
              reject(new Error(data.content?.error || toStr(data.content) || 'LLM error'));
              break;
            default:
              break;
          }
        };

        socket.on(sioEvents.application_predict, handler);

        fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders },
          body: JSON.stringify(requestBody),
        }).then(async (response) => {
          if (!response.ok) {
            const text = await response.text();
            clearTimeout(timeoutId);
            cleanup();
            reject(new Error(`LLM request failed: ${text}`));
          }
        }).catch((err) => {
          clearTimeout(timeoutId);
          cleanup();
          reject(err);
        });
      });

      if (!llmResponse) {
        throw new Error('Empty response from LLM');
      }

      const fixedCode = extractMermaidFromResponse(llmResponse);
      
      // Update message content in place
      setMessages(prev => prev.map((msg, idx) => {
        if (idx === messageIdx) {
          return {
            ...msg,
            content: replaceMermaidBlock(msg.content, errorInfo.blockIndex, fixedCode),
          };
        }
        return msg;
      }));

    } catch (err) {
      console.error('[ChatDrawer] Error fixing diagram:', err);
      setError(`Fix failed: ${err.message}`);
    } finally {
      setFixingDiagram(null);
    }
  }, [fixingDiagram, toolkit, projectId, extractMermaidFromResponse, replaceMermaidBlock]);

  // Handle clear chat - removes all messages and clears localStorage
  const handleClearChat = useCallback(() => {
    setMessages([]);
    currentThinkingBlockIdRef.current = null;
    try {
      localStorage.removeItem(storageKey);
      localStorage.removeItem(capabilityStorageKey);
    } catch (err) {
      console.warn('[ChatDrawer] Failed to clear persisted messages:', err);
    }
  }, [storageKey, capabilityStorageKey]);

  // Handle Enter key
  const handleKeyPress = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  // Show collapsed tab when drawer is closed
  if (!open) {
    return <ChatTab onClick={onOpen} disabled={!wikiGenerated} />;
  }

  return (
    <Drawer
      variant="persistent"
      anchor="right"
      open={open}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          borderLeft: '1px solid',
          borderColor: 'divider',
          bgcolor: mode === 'dark' ? 'background.default' : 'background.paper',
        },
      }}
    >
      {/* Resize Handle */}
      <Box
        onMouseDown={handleResizeStart}
        sx={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: 6,
          cursor: 'col-resize',
          bgcolor: isResizing ? 'primary.main' : 'transparent',
          transition: 'background-color 0.2s',
          zIndex: 1,
          '&:hover': {
            bgcolor: 'primary.light',
            opacity: 0.5,
          },
        }}
      />

      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography
          variant="h6"
          sx={{ display: 'flex', alignItems: 'center', gap: 1, fontSize: '1rem' }}
        >
          <ChatIcon fontSize="small" /> Chat
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {messages.length > 0 && (
            <Tooltip title="Clear chat history">
              <IconButton onClick={handleClearChat} size="small">
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <IconButton onClick={onClose} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Wiki Required Warning */}
      {!wikiGenerated ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Alert severity="info" sx={{ textAlign: 'left' }}>
            Generate a wiki first to enable Chat features. The chat uses the wiki's
            index to answer your questions.
          </Alert>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100% - 64px)' }}>
          {/* Mode Toggle */}
          <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
            <ToggleButtonGroup
              value={chatMode}
              exclusive
              onChange={handleModeChange}
              size="small"
              fullWidth
            >
              <ToggleButton value="ask" sx={{ py: 0.5 }}>
                <AskIcon sx={{ mr: 0.5, fontSize: '18px' }} />
                <Typography variant="caption">Ask</Typography>
              </ToggleButton>
              <ToggleButton value="research" sx={{ py: 0.5 }}>
                <ResearchIcon sx={{ mr: 0.5, fontSize: '18px' }} />
                <Typography variant="caption">Research</Typography>
              </ToggleButton>
            </ToggleButtonGroup>
            <Typography
              variant="caption"
              sx={{ mt: 0.5, display: 'block', color: 'text.secondary', fontSize: '11px' }}
            >
              {chatMode === 'ask'
                ? 'Quick Q&A about the repository'
                : 'Deep analysis with planning and multi-step investigation'}
            </Typography>
          </Box>

          {/* Messages Area */}
          <Box ref={messagesContainerRef} sx={{ flex: 1, overflow: 'auto', p: 1.5 }}>
            {messages.length === 0 && (
              <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                <ChatIcon sx={{ fontSize: 40, opacity: 0.3, mb: 1 }} />
                <Typography variant="body2" sx={{ fontSize: '13px' }}>
                  Ask any question about this repository
                </Typography>
                <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block' }}>
                  Examples:
                </Typography>
                <Typography variant="caption" color="text.disabled">
                  • How does authentication work?
                </Typography>
                <Typography variant="caption" color="text.disabled" display="block">
                  • What are the main components?
                </Typography>
              </Box>
            )}

            {messages.map((msg, idx) => {
              if (msg?.type === 'thinking_steps') {
                return (
                  <ThinkingStepsBlock
                    key={msg.id || idx}
                    block={msg}
                    mode={mode}
                    onToggle={() => handleToggleThinkingBlock(msg.id)}
                  />
                );
              }

              const isLastAssistant = msg?.role === 'assistant' && idx === messages.length - 1;
              return (
                <MessageBubble
                  key={idx}
                  message={msg}
                  mode={mode}
                  messageIdx={idx}
                  activeCapability={chatMode}
                  isLastAssistant={isLastAssistant}
                  onRegenerate={isLastAssistant && !isLoading ? handleRegenerate : undefined}
                  onQuickFixDiagram={handleQuickFixDiagram}
                  fixingDiagram={fixingDiagram}
                />
              );
            })}

            {/* Always show spinner at bottom while loading */}
            {isLoading && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, ml: 1 }}>
                <CircularProgress size={16} />
                <Typography variant="caption" color="text.secondary">
                  Searching and analyzing...
                </Typography>
              </Box>
            )}

            <div ref={messagesEndRef} />
          </Box>

          {/* Todos Panel - Only shows todos for research mode */}
          <TodosPanel
            todos={researchTodos}
            expanded={showThinking}
            onToggle={() => setShowThinking(!showThinking)}
            onDismiss={() => setResearchTodos([])}
          />

          {/* Error Display */}
          {error && (
            <Box sx={{ px: 1.5, pt: 1 }}>
              <Alert 
                severity="error" 
                onClose={() => setError(null)}
                sx={{ fontSize: '0.75rem' }}
              >
                {error}
              </Alert>
            </Box>
          )}

          {/* Input Area */}
          <Box sx={{ p: 1.5, borderTop: 1, borderColor: 'divider' }}>
            <TextField
              fullWidth
              multiline
              maxRows={3}
              size="small"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about the code..."
              disabled={isLoading}
              sx={{
                '& .MuiInputBase-root': {
                  fontSize: '14px',
                  pr: 0.5,
                },
              }}
              InputProps={{
                endAdornment: (
                  <IconButton
                    onClick={handleSend}
                    disabled={isLoading || !input.trim()}
                    color="primary"
                    size="small"
                  >
                    <SendIcon fontSize="small" />
                  </IconButton>
                ),
              }}
            />
          </Box>
        </Box>
      )}
    </Drawer>
  );
});

ChatDrawer.displayName = 'ChatDrawer';

export default ChatDrawer;
export { DEFAULT_DRAWER_WIDTH as DRAWER_WIDTH };
