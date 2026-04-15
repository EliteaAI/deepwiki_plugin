/**
 * ThinkingStepsPanel - Display progress/thinking steps during wiki generation
 * Shows real-time progress updates from the backend
 */
import { useState, useEffect, useRef, memo } from 'react';
import {
  Box,
  Typography,
  Paper,
  Collapse,
  IconButton,
  LinearProgress,
  Chip,
  Stack,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as PendingIcon,
  Error as ErrorIcon,
  Psychology as ThinkingIcon,
  UnfoldMore as UnfoldMoreIcon,
  UnfoldLess as UnfoldLessIcon,
} from '@mui/icons-material';

// Phase icons and colors based on message content
const PHASE_CONFIG = {
  indexing: { icon: '📁', color: '#4CAF50', label: 'Indexing' },
  parsing: { icon: '🔍', color: '#2196F3', label: 'Parsing' },
  analyzing: { icon: '🧠', color: '#9C27B0', label: 'Analyzing' },
  generating: { icon: '✍️', color: '#FF9800', label: 'Generating' },
  writing: { icon: '📝', color: '#E91E63', label: 'Writing' },
  retrieval: { icon: '🔎', color: '#00BCD4', label: 'Retrieval' },
  filter: { icon: '🔧', color: '#795548', label: 'Filter' },
  building: { icon: '🏗️', color: '#3F51B5', label: 'Building' },
  mermaid: { icon: '📊', color: '#009688', label: 'Diagrams' },
  worker: { icon: '⚙️', color: '#607D8B', label: 'Worker' },
  info: { icon: 'ℹ️', color: '#2196F3', label: 'Info' },
  success: { icon: '✅', color: '#4CAF50', label: 'Success' },
  default: { icon: '💭', color: '#607D8B', label: 'Processing' },
};

/**
 * Detect phase from message content or step type
 */
function detectPhase(message, stepType) {
  // Check explicit type first (for reconnect notices, success messages, etc.)
  if (stepType === 'info') return 'info';
  if (stepType === 'success') return 'success';
  
  if (!message) return 'default';
  const lower = message.toLowerCase();
  
  if (lower.includes('[retrieval]')) return 'retrieval';
  if (lower.includes('[arch_filter]') || lower.includes('filter')) return 'filter';
  if (lower.includes('building llm') || lower.includes('building')) return 'building';
  if (lower.includes('mermaid')) return 'mermaid';
  if (lower.includes('worker') || lower.includes('slot')) return 'worker';
  if (lower.includes('indexing') || lower.includes('index')) return 'indexing';
  if (lower.includes('parsing') || lower.includes('parse')) return 'parsing';
  if (lower.includes('analyzing') || lower.includes('analyz')) return 'analyzing';
  if (lower.includes('generating') || lower.includes('generat')) return 'generating';
  if (lower.includes('writing') || lower.includes('write')) return 'writing';
  
  return 'default';
}

/**
 * Get first line from message (no truncation)
 */
function getFirstLine(message) {
  if (!message) return '';
  return message.split('\n')[0].trim();
}

/**
 * Single thinking step item - expandable chip-like design
 */
const ThinkingStep = memo(({ step, index, isLatest }) => {
  const message = step.message || step.data?.message || '';
  const phase = detectPhase(message, step.type);
  const config = PHASE_CONFIG[phase] || PHASE_CONFIG.default;
  const [expanded, setExpanded] = useState(false);
  
  const firstLine = getFirstLine(message);
  const hasMoreContent = message.includes('\n') && message.trim() !== firstLine;

  // Special styling for reconnect notices
  const isReconnectNotice = step.isReconnectNotice;

  return (
    <Box
      sx={{
        borderRadius: 1.5,
        bgcolor: isLatest ? 'action.selected' : 'transparent',
        border: expanded ? '1px solid' : '1px solid transparent',
        borderColor: expanded ? 'divider' : 'transparent',
        transition: 'all 0.2s ease',
        overflow: 'hidden',
        '&:hover': {
          bgcolor: 'action.hover',
          borderColor: 'divider',
        },
      }}
    >
      {/* Compact preview row */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          py: 0.75,
          px: 1.5,
          cursor: hasMoreContent ? 'pointer' : 'default',
        }}
        onClick={() => hasMoreContent && setExpanded(!expanded)}
      >
        {/* Phase icon */}
        <Box
          sx={{
            width: 24,
            height: 24,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: `${config.color}20`,
            flexShrink: 0,
            fontSize: '12px',
          }}
        >
          {config.icon}
        </Box>

        {/* Phase label chip */}
        <Chip
          label={config.label}
          size="small"
          sx={{
            height: 18,
            fontSize: '0.65rem',
            bgcolor: `${config.color}25`,
            color: config.color,
            fontWeight: 600,
            flexShrink: 0,
            '& .MuiChip-label': { px: 1 },
          }}
        />

        {/* Preview text - full first line, no truncation */}
        <Typography
          variant="body2"
          sx={{
            flex: 1,
            color: 'text.primary',
            fontSize: '0.75rem',
            lineHeight: 1.4,
          }}
        >
          {firstLine}
        </Typography>

        {/* Current indicator */}
        {isLatest && (
          <Chip
            label="●"
            size="small"
            color="primary"
            sx={{ 
              height: 16, 
              width: 16,
              minWidth: 16,
              fontSize: '0.5rem',
              '& .MuiChip-label': { px: 0 },
            }}
          />
        )}

        {/* Expand toggle */}
        {hasMoreContent && (
          <IconButton 
            size="small" 
            sx={{ 
              p: 0.25,
              color: 'text.secondary',
            }}
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            {expanded ? <UnfoldLessIcon sx={{ fontSize: 16 }} /> : <UnfoldMoreIcon sx={{ fontSize: 16 }} />}
          </IconButton>
        )}
      </Box>

      {/* Expanded full content */}
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 1.5,
            pb: 1.5,
            pt: 0.5,
            borderTop: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography
            variant="body2"
            component="pre"
            sx={{
              color: 'text.secondary',
              fontSize: '0.7rem',
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'monospace',
              m: 0,
              p: 1,
              bgcolor: 'background.default',
              borderRadius: 1,
              maxHeight: 200,
              overflow: 'auto',
            }}
          >
            {message}
          </Typography>
        </Box>
      </Collapse>
    </Box>
  );
});

ThinkingStep.displayName = 'ThinkingStep';

/**
 * Main ThinkingStepsPanel component
 */
function ThinkingStepsPanel({ 
  steps = [], 
  status = 'idle', // idle, running, completed, error
  statusMessage = '',
  elapsedTime = 0,
  mode = 'light',
}) {
  const [expanded, setExpanded] = useState(true);
  const [expandAll, setExpandAll] = useState(false);
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new steps arrive
  useEffect(() => {
    if (scrollRef.current && expanded) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [steps, expanded]);

  const isRunning = status === 'running';
  const isCompleted = status === 'completed';
  const isError = status === 'error';

  const colors = mode === 'dark' 
    ? {
        bg: 'rgba(41, 184, 245, 0.08)',
        border: 'rgba(41, 184, 245, 0.3)',
        headerBg: 'rgba(41, 184, 245, 0.12)',
      }
    : {
        bg: 'rgba(99, 144, 254, 0.06)',
        border: 'rgba(99, 144, 254, 0.25)',
        headerBg: 'rgba(99, 144, 254, 0.1)',
      };

  return (
    <Paper
      elevation={0}
      sx={{
        borderBottom: '1px solid',
        borderColor: colors.border,
        borderRadius: 0,
        overflow: 'hidden',
        bgcolor: colors.bg,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1.5,
          bgcolor: colors.headerBg,
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          {isRunning && <ThinkingIcon sx={{ color: 'primary.main', animation: 'pulse 2s infinite' }} />}
          {isCompleted && <CheckCircleIcon sx={{ color: 'success.main' }} />}
          {isError && <ErrorIcon sx={{ color: 'error.main' }} />}
          {!isRunning && !isCompleted && !isError && <PendingIcon sx={{ color: 'text.secondary' }} />}
          
          <Typography variant="subtitle2" fontWeight={600}>
            {isRunning ? 'Generating Wiki...' : isCompleted ? 'Generation Complete' : isError ? 'Generation Failed' : 'Progress'}
          </Typography>

          {steps.length > 0 && (
            <Chip
              label={`${steps.length} step${steps.length !== 1 ? 's' : ''}`}
              size="small"
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {elapsedTime > 0 && (
            <Typography variant="caption" color="text.secondary">
              {Math.floor(elapsedTime / 60)}:{String(elapsedTime % 60).padStart(2, '0')}
            </Typography>
          )}
          <IconButton size="small">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>
      </Box>

      {/* Progress bar */}
      {isRunning && <LinearProgress sx={{ height: 2 }} />}

      {/* Content */}
      <Collapse in={expanded}>
        <Box
          ref={scrollRef}
          sx={{
            maxHeight: 400,
            overflowY: 'auto',
            p: 1,
          }}
        >
          {/* Error message only */}
          {isError && statusMessage && (
            <Typography
              variant="body2"
              sx={{
                px: 1.5,
                py: 1,
                color: 'error.main',
                fontStyle: 'italic',
                borderBottom: steps.length > 0 ? '1px solid' : 'none',
                borderColor: 'divider',
                mb: steps.length > 0 ? 1 : 0,
              }}
            >
              {statusMessage}
            </Typography>
          )}

          {/* Steps list */}
          {steps.length > 0 ? (
            <Stack spacing={0.25}>
              {steps.map((step, index) => (
                <ThinkingStep
                  key={step.id || index}
                  step={step}
                  index={index}
                  isLatest={index === steps.length - 1 && isRunning}
                />
              ))}
            </Stack>
          ) : (
            <Typography
              variant="body2"
              sx={{ px: 1.5, py: 2, color: 'text.secondary', textAlign: 'center' }}
            >
              {isRunning ? 'Waiting for progress updates...' : 'No progress information available'}
            </Typography>
          )}
        </Box>
      </Collapse>

      {/* Add keyframes for pulse animation */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `}
      </style>
    </Paper>
  );
}

ThinkingStepsPanel.displayName = 'ThinkingStepsPanel';

export default memo(ThinkingStepsPanel);
