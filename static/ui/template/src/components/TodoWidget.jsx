/**
 * TodoWidget - Beautiful todo list visualization for Deep Research
 * Shows research plan with visual status indicators
 */
import { memo } from 'react';
import { Box, Typography, Collapse, Stack, Chip } from '@mui/material';

/**
 * Status configuration with icons and colors
 */
const STATUS_CONFIG = {
  'not-started': {
    icon: '○',
    color: '#9E9E9E',  // Gray
    label: 'Queued',
    bgColor: 'rgba(158, 158, 158, 0.1)',
  },
  'in-progress': {
    icon: '●',
    color: '#4CAF50',  // Green
    label: 'In Progress',
    bgColor: 'rgba(76, 175, 80, 0.15)',
    pulse: true,
  },
  'completed': {
    icon: '✓',
    color: '#4CAF50',  // Green
    label: 'Completed',
    bgColor: 'rgba(76, 175, 80, 0.1)',
  },
  'failed': {
    icon: '✗',
    color: '#F44336',  // Red
    label: 'Failed',
    bgColor: 'rgba(244, 67, 54, 0.1)',
  },
  'skipped': {
    icon: '⊘',
    color: '#FF9800',  // Yellow/Orange
    label: 'Skipped',
    bgColor: 'rgba(255, 152, 0, 0.1)',
  },
};

/**
 * Single todo item with visual status indicator
 */
const TodoItem = memo(function TodoItem({ todo, isCompact = false }) {
  const status = todo.status || 'not-started';
  const config = STATUS_CONFIG[status] || STATUS_CONFIG['not-started'];
  
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 1,
        py: isCompact ? 0.5 : 0.75,
        px: isCompact ? 0 : 1,
        borderRadius: 1,
        bgcolor: isCompact ? 'transparent' : config.bgColor,
        transition: 'all 0.2s ease',
      }}
    >
      {/* Status indicator */}
      <Typography
        component="span"
        sx={{
          fontSize: isCompact ? '14px' : '16px',
          lineHeight: 1.4,
          color: config.color,
          fontWeight: 600,
          minWidth: isCompact ? '16px' : '20px',
          textAlign: 'center',
          ...(config.pulse && {
            animation: 'todoPulse 1.5s ease-in-out infinite',
          }),
        }}
      >
        {config.icon}
      </Typography>

      {/* Content */}
      <Box sx={{ flex: 1, minWidth: 0 }}>
        {/* Title */}
        <Typography
          variant="body2"
          sx={{
            fontSize: isCompact ? '12px' : '13px',
            fontWeight: status === 'in-progress' ? 600 : 400,
            color: status === 'completed' ? 'text.secondary' : 'text.primary',
            textDecoration: status === 'completed' ? 'line-through' : 'none',
            opacity: status === 'completed' ? 0.75 : 1,
            lineHeight: 1.4,
          }}
        >
          {todo.title}
        </Typography>

        {/* Description (only show in non-compact mode) */}
        {!isCompact && todo.description && (
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              fontSize: '11px',
              color: 'text.secondary',
              mt: 0.25,
              lineHeight: 1.3,
              opacity: status === 'completed' ? 0.6 : 0.8,
            }}
          >
            {todo.description}
          </Typography>
        )}
      </Box>

      {/* Status label (optional, for non-compact mode) */}
      {!isCompact && status === 'in-progress' && (
        <Chip
          label={config.label}
          size="small"
          sx={{
            height: 18,
            fontSize: '10px',
            bgcolor: config.bgColor,
            color: config.color,
            fontWeight: 600,
            '& .MuiChip-label': { px: 0.75 },
          }}
        />
      )}
    </Box>
  );
});

TodoItem.displayName = 'TodoItem';

/**
 * Full TodoWidget component - displays a list of todos with progress
 */
const TodoWidget = memo(function TodoWidget({ 
  todos = [], 
  isCompact = false,
  showProgress = true,
}) {
  if (!todos || todos.length === 0) {
    return null;
  }

  // Calculate progress
  const completed = todos.filter(t => t.status === 'completed').length;
  const inProgress = todos.filter(t => t.status === 'in-progress').length;
  const total = todos.length;
  const progressPercent = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <Box
      sx={{
        borderRadius: isCompact ? 0 : 1.5,
        bgcolor: isCompact ? 'transparent' : 'background.paper',
        border: isCompact ? 'none' : '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      {/* Header with progress (non-compact mode) */}
      {!isCompact && showProgress && (
        <Box
          sx={{
            px: 1.5,
            py: 1,
            bgcolor: 'action.hover',
            borderBottom: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Typography
            variant="subtitle2"
            sx={{
              fontSize: '12px',
              fontWeight: 600,
              color: 'text.primary',
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
            }}
          >
            📋 Research Plan
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {inProgress > 0 && (
              <Chip
                label={`${inProgress} active`}
                size="small"
                sx={{
                  height: 18,
                  fontSize: '10px',
                  bgcolor: 'rgba(76, 175, 80, 0.15)',
                  color: '#4CAF50',
                  fontWeight: 600,
                }}
              />
            )}
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
              {completed}/{total}
            </Typography>
          </Box>
        </Box>
      )}

      {/* Progress bar (non-compact mode) */}
      {!isCompact && showProgress && (
        <Box
          sx={{
            height: 3,
            bgcolor: 'action.disabledBackground',
            position: 'relative',
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: `${progressPercent}%`,
              bgcolor: completed === total ? 'success.main' : 'primary.main',
              transition: 'width 0.3s ease',
            }}
          />
        </Box>
      )}

      {/* Todo list */}
      <Stack 
        spacing={isCompact ? 0 : 0.5} 
        sx={{ 
          p: isCompact ? 0 : 1,
          maxHeight: isCompact ? 'none' : 300,
          overflowY: 'auto',
        }}
      >
        {todos.map((todo, index) => (
          <TodoItem
            key={todo.id || index}
            todo={todo}
            isCompact={isCompact}
          />
        ))}
      </Stack>

      {/* Keyframes for pulse animation */}
      <style>
        {`
          @keyframes todoPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `}
      </style>
    </Box>
  );
});

TodoWidget.displayName = 'TodoWidget';

export default TodoWidget;
export { TodoItem, STATUS_CONFIG };
