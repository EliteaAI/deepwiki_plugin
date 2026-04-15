import { memo } from 'react';
import { Box, Typography, Chip, Tooltip, CircularProgress } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

/**
 * SlotsIndicator - Shows cluster-wide slot availability for wiki generation
 * 
 * @param {Object} props
 * @param {import('../hooks/useSlots').SlotsStatus | null} props.slots - Current slots status
 * @param {boolean} props.loading - Whether slots are being loaded
 * @param {string | null} props.error - Error message if fetch failed
 * @param {boolean} [props.compact=false] - Use compact display mode
 */
const SlotsIndicator = memo(props => {
  const { slots, loading, error, compact = false } = props;

  if (loading && !slots) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, opacity: 0.7 }}>
        <CircularProgress size={12} />
        <Typography variant="caption">Checking slots...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Tooltip title={error}>
        <Chip
          size="small"
          label="Slots unknown"
          color="default"
          variant="outlined"
          sx={{ height: 20, fontSize: '0.7rem' }}
        />
      </Tooltip>
    );
  }

  if (!slots) {
    return null;
  }

  const { available, total, active, mode, canStart } = slots;
  const isJobsMode = mode === 'jobs';
  
  // Color based on availability
  const getColor = () => {
    if (!canStart) return 'error';
    if (available <= 1) return 'warning';
    return 'success';
  };

  const tooltipContent = (
    <Box sx={{ p: 0.5 }}>
      <Typography variant="caption" component="div">
        <strong>Cluster Slot Availability</strong>
      </Typography>
      <Typography variant="caption" component="div" sx={{ mt: 0.5 }}>
        Available: {available} / {total}
      </Typography>
      <Typography variant="caption" component="div">
        Active jobs: {active}
      </Typography>
      <Typography variant="caption" component="div" sx={{ mt: 0.5, opacity: 0.7 }}>
        Mode: {isJobsMode ? 'K8s Jobs (cluster-wide)' : 'Subprocess (per-pod)'}
      </Typography>
      {!canStart && (
        <Typography variant="caption" component="div" sx={{ mt: 0.5, color: 'error.main' }}>
          All slots are currently in use. Please wait for a generation to complete.
        </Typography>
      )}
    </Box>
  );

  if (compact) {
    return (
      <Tooltip title={tooltipContent} arrow placement="top">
        <Chip
          size="small"
          label={`${available}/${total}`}
          color={getColor()}
          variant={canStart ? 'outlined' : 'filled'}
          icon={<InfoOutlinedIcon sx={{ fontSize: 14 }} />}
          sx={{ height: 20, fontSize: '0.7rem', cursor: 'help' }}
        />
      </Tooltip>
    );
  }

  return (
    <Tooltip title={tooltipContent} arrow placement="top">
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          p: 1,
          borderRadius: 1,
          bgcolor: canStart ? 'action.hover' : 'error.dark',
          cursor: 'help',
        }}
      >
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            bgcolor: canStart ? 'success.main' : 'error.main',
            animation: active > 0 ? 'pulse 2s infinite' : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 1 },
              '50%': { opacity: 0.5 },
              '100%': { opacity: 1 },
            },
          }}
        />
        <Typography variant="caption" sx={{ fontWeight: 500 }}>
          {available}/{total} slots available
        </Typography>
      </Box>
    </Tooltip>
  );
});

SlotsIndicator.displayName = 'SlotsIndicator';

export default SlotsIndicator;
