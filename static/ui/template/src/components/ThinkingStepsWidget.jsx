/**
 * ThinkingStepsWidget - Collapsible thinking steps visualization
 * Shows tool calls, status updates, and thinking steps in GitHub Copilot style
 */
import { useState, memo, useRef, useEffect } from 'react';
import { Box, Typography, Collapse, IconButton, Chip, Stack } from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckIcon,
  RadioButtonUnchecked as PendingIcon,
  Error as ErrorIcon,
  Build as ToolIcon,
  Psychology as ThinkingIcon,
  Search as SearchIcon,
  AutoFixHigh as OptimizeIcon,
  Article as ArticleIcon,
  Code as CodeIcon,
} from '@mui/icons-material';

/**
 * Event type configuration
 */
const EVENT_TYPE_CONFIG = {
  tool_start: {
    icon: ToolIcon,
    color: '#2196F3',  // Blue
    defaultTitle: 'Tool Call',
  },
  tool_end: {
    icon: CheckIcon,
    color: '#4CAF50',  // Green
    defaultTitle: 'Tool Complete',
  },
  status: {
    icon: ThinkingIcon,
    color: '#9C27B0',  // Purple
    defaultTitle: 'Status',
  },
  log: {
    icon: ArticleIcon,
    color: '#607D8B',  // Blue Gray
    defaultTitle: 'Log',
  },
  thinking: {
    icon: ThinkingIcon,
    color: '#FF9800',  // Orange
    defaultTitle: 'Thinking',
  },
  query_optimization: {
    icon: OptimizeIcon,
    color: '#00BCD4',  // Cyan
    defaultTitle: 'Query Optimization',
  },
  answer_generation: {
    icon: ArticleIcon,
    color: '#4CAF50',  // Green
    defaultTitle: 'Generating Answer',
  },
  search: {
    icon: SearchIcon,
    color: '#2196F3',  // Blue
    defaultTitle: 'Searching',
  },
  read_file: {
    icon: CodeIcon,
    color: '#795548',  // Brown
    defaultTitle: 'Reading File',
  },
};

/**
 * Get tool-specific icon based on tool name
 */
function getToolIcon(toolName) {
  if (!toolName) return ToolIcon;
  const lower = toolName.toLowerCase();
  
  if (lower.includes('search') || lower.includes('retriev')) return SearchIcon;
  if (lower.includes('read') || lower.includes('file')) return CodeIcon;
  if (lower.includes('optim')) return OptimizeIcon;
  if (lower.includes('think') || lower.includes('plan')) return ThinkingIcon;
  
  return ToolIcon;
}

/**
 * Format tool name for display
 */
function formatToolName(name) {
  if (!name) return 'Processing';
  return name
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

/**
 * Single thinking step item - collapsible with expand/collapse
 */
const ThinkingStepItem = memo(function ThinkingStepItem({ 
  step, 
  isLatest = false,
  defaultExpanded = false,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  
  // Parse step data
  const eventType = step.event || step.type || 'log';
  const data = step.data || step;
  const config = EVENT_TYPE_CONFIG[eventType] || EVENT_TYPE_CONFIG.log;
  
  // Extract display info
  const toolName = data.tool || data.toolName || '';
  const title = step.title || data.message || formatToolName(toolName) || config.defaultTitle;
  const status = data.status || (eventType === 'tool_end' ? 'completed' : 'in_progress');
  const content = data.content || data.input || data.output || data.description || '';
  const hasContent = content && content.length > 0;
  
  // Get appropriate icon
  const IconComponent = getToolIcon(toolName) || config.icon;
  
  // Status styling
  const isCompleted = status === 'completed' || eventType === 'tool_end';
  const isError = status === 'error' || status === 'failed';
  const isInProgress = status === 'in_progress' || status === 'in-progress' || eventType === 'tool_start';

  return (
    <Box
      sx={{
        borderRadius: 1,
        border: '1px solid',
        borderColor: expanded ? 'divider' : 'transparent',
        bgcolor: isLatest ? 'action.selected' : 'transparent',
        transition: 'all 0.2s ease',
        overflow: 'hidden',
        '&:hover': {
          borderColor: 'divider',
          bgcolor: 'action.hover',
        },
      }}
    >
      {/* Header row - always visible */}
      <Box
        onClick={() => hasContent && setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          py: 0.75,
          px: 1,
          cursor: hasContent ? 'pointer' : 'default',
        }}
      >
        {/* Status icon */}
        <Box
          sx={{
            width: 24,
            height: 24,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: `${config.color}15`,
            flexShrink: 0,
          }}
        >
          {isError ? (
            <ErrorIcon sx={{ fontSize: 14, color: 'error.main' }} />
          ) : isCompleted ? (
            <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
          ) : (
            <IconComponent 
              sx={{ 
                fontSize: 14, 
                color: config.color,
                ...(isInProgress && isLatest && {
                  animation: 'stepPulse 1.5s ease-in-out infinite',
                }),
              }} 
            />
          )}
        </Box>

        {/* Title */}
        <Typography
          variant="body2"
          sx={{
            flex: 1,
            fontSize: '12px',
            fontWeight: isLatest && isInProgress ? 600 : 400,
            color: isCompleted ? 'text.secondary' : 'text.primary',
            lineHeight: 1.4,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {title}
        </Typography>

        {/* Status chip for in-progress items */}
        {isInProgress && isLatest && (
          <Chip
            label="●"
            size="small"
            color="primary"
            sx={{
              height: 16,
              width: 16,
              minWidth: 16,
              '& .MuiChip-label': { px: 0, fontSize: '8px' },
            }}
          />
        )}

        {/* Expand toggle */}
        {hasContent && (
          <IconButton
            size="small"
            sx={{
              p: 0.25,
              color: 'text.secondary',
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
            }}
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            <ExpandMoreIcon sx={{ fontSize: 16 }} />
          </IconButton>
        )}
      </Box>

      {/* Expanded content */}
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 1.5,
            pb: 1,
            pt: 0.5,
            borderTop: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography
            variant="body2"
            component="pre"
            sx={{
              fontSize: '11px',
              lineHeight: 1.5,
              color: 'text.secondary',
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              m: 0,
              p: 1,
              bgcolor: 'background.default',
              borderRadius: 0.5,
              maxHeight: 200,
              overflow: 'auto',
            }}
          >
            {typeof content === 'object' ? JSON.stringify(content, null, 2) : content}
          </Typography>
        </Box>
      </Collapse>
    </Box>
  );
});

ThinkingStepItem.displayName = 'ThinkingStepItem';

/**
 * Plain text thinking step (no collapsible, just text)
 * For strategic thinking / planning messages
 */
const PlainThinkingStep = memo(function PlainThinkingStep({ step, isLatest }) {
  const message = step.message || step.data?.message || step.content || '';
  
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 0.75,
        py: 0.5,
        px: 1,
        borderRadius: 0.5,
        bgcolor: isLatest ? 'action.selected' : 'transparent',
      }}
    >
      {/* Thinking icon */}
      <ThinkingIcon 
        sx={{ 
          fontSize: 14, 
          color: 'text.secondary',
          mt: 0.25,
          ...(isLatest && {
            animation: 'stepPulse 1.5s ease-in-out infinite',
          }),
        }} 
      />
      
      {/* Text */}
      <Typography
        variant="body2"
        sx={{
          fontSize: '12px',
          color: 'text.primary',
          lineHeight: 1.5,
        }}
      >
        {message}
      </Typography>
    </Box>
  );
});

PlainThinkingStep.displayName = 'PlainThinkingStep';

/**
 * Main ThinkingStepsWidget - renders a list of thinking steps
 */
const ThinkingStepsWidget = memo(function ThinkingStepsWidget({
  steps = [],
  isCompact = false,
  maxHeight = 200,
}) {
  const scrollRef = useRef(null);

  // Auto-scroll to bottom on new steps
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [steps.length]);

  if (!steps || steps.length === 0) {
    return null;
  }

  return (
    <Box
      ref={scrollRef}
      sx={{
        maxHeight,
        overflowY: 'auto',
        overflowX: 'hidden',
      }}
    >
      <Stack spacing={0.25}>
        {steps.map((step, index) => {
          const isLatest = index === steps.length - 1;
          const eventType = step.event || step.type;
          
          // Use plain text for simple log/status messages without tool info
          const isPlainMessage = 
            !eventType || 
            eventType === 'log' || 
            (eventType === 'status' && !step.data?.tool);
          
          if (isPlainMessage && !step.data?.content) {
            return (
              <PlainThinkingStep
                key={step.id || index}
                step={step}
                isLatest={isLatest}
              />
            );
          }
          
          return (
            <ThinkingStepItem
              key={step.id || index}
              step={step}
              isLatest={isLatest}
              defaultExpanded={false}
            />
          );
        })}
      </Stack>

      {/* Keyframes for pulse animation */}
      <style>
        {`
          @keyframes stepPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
        `}
      </style>
    </Box>
  );
});

ThinkingStepsWidget.displayName = 'ThinkingStepsWidget';

export default ThinkingStepsWidget;
export { ThinkingStepItem, PlainThinkingStep, EVENT_TYPE_CONFIG };
