import { useEffect, useRef, useState } from 'react';
import { Box, Alert, IconButton, Tooltip, Dialog, DialogContent, Button, CircularProgress, Typography } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import DownloadIcon from '@mui/icons-material/Download';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import CloseIcon from '@mui/icons-material/Close';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import EditIcon from '@mui/icons-material/Edit';
import mermaid from 'mermaid';

/**
 * Parse error message to extract line number
 * @param {string} errorMessage - Error message like "Lexical error on line 40"
 * @returns {number|null} - Line number or null if not found
 */
const parseErrorLineNumber = (errorMessage) => {
  const match = errorMessage?.match(/line\s+(\d+)/i);
  return match ? parseInt(match[1], 10) : null;
};

const MermaidDiagram = ({ 
  chart, 
  mode = 'dark',
  blockIndex = 0,        // Which mermaid block this is (0-indexed)
  onQuickFix,            // Callback for quick fix: (errorInfo) => void
  onNavigateToError,     // Callback for edit navigation: (errorInfo) => void
  isFixing = false,      // Show loading state while fixing
}) => {
  const containerRef = useRef(null);
  const modalContainerRef = useRef(null);
  const svgRef = useRef(null);
  const [error, setError] = useState(null);
  const [svg, setSvg] = useState(null);
  const [svgDimensions, setSvgDimensions] = useState({ width: 0, height: 0 });
  
  // Main view state - start at 1.5x scale for better visibility
  const [scale, setScale] = useState(1.5);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Modal view state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [modalScale, setModalScale] = useState(1.5);
  const [modalPosition, setModalPosition] = useState({ x: 0, y: 0 });
  const [isModalDragging, setIsModalDragging] = useState(false);
  const [modalDragStart, setModalDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Update theme based on mode
    mermaid.initialize({
      startOnLoad: false,
      theme: mode === 'dark' ? 'dark' : 'default',
      securityLevel: 'loose',
      fontFamily: '"Montserrat", Roboto, Arial, sans-serif',
      fontSize: 14,
      logLevel: 'fatal', // Suppress console errors
      themeVariables: mode === 'dark' ? {
        primaryColor: '#6ae8fa',
        primaryTextColor: '#fff',
        primaryBorderColor: '#3B3E46',
        lineColor: '#6ae8fa',
        secondaryColor: '#181F2A',
        tertiaryColor: '#0E131D',
      } : {
        primaryColor: '#C428DD',
        primaryTextColor: '#000',
        primaryBorderColor: '#E1E5E9',
        lineColor: '#C428DD',
        secondaryColor: '#FFFFFF',
        tertiaryColor: '#F8FCFF',
      },
    });

    const renderDiagram = async () => {
      if (!chart) return;

      try {
        setError(null);
        setSvg(null);

        // Validate first to avoid Mermaid injecting error SVGs
        await mermaid.parse(chart);

        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg: renderedSvg } = await mermaid.render(id, chart);
        setSvg(renderedSvg);
        
        // Extract SVG dimensions
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(renderedSvg, 'image/svg+xml');
        const svgElement = svgDoc.querySelector('svg');
        if (svgElement) {
          const viewBox = svgElement.getAttribute('viewBox');
          if (viewBox) {
            const [, , width, height] = viewBox.split(' ').map(Number);
            setSvgDimensions({ width, height });
          }
        }
      } catch (err) {
        // Keep full error message for LLM (includes context with arrows)
        const fullErrorMessage = err.message || 'Failed to render diagram';
        // Extract first line for display, but keep full error available
        const shortError = fullErrorMessage.split('\n')[0];
        setSvg(null);
        // Store both: short for display, full for LLM
        setError({ 
          display: shortError, 
          full: fullErrorMessage 
        });
      }
    };

    renderDiagram();
  }, [chart, mode]);

  // Main view handlers
  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };

  const handleFitToScreen = () => {
    const container = containerRef.current;
    if (container && svgDimensions.width && svgDimensions.height) {
      const { clientWidth, clientHeight } = container;
      const availableWidth = Math.max(clientWidth - 32, 100);
      const availableHeight = Math.max(clientHeight - 32, 200);
      const fitScale = Math.min(
        availableWidth / svgDimensions.width,
        availableHeight / svgDimensions.height,
        3,
      );
      const clamped = Math.min(Math.max(fitScale, 0.5), 3);
      setScale(clamped);
    } else {
      setScale(1.5);
    }
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setScale(prev => Math.min(Math.max(prev + delta, 0.5), 3));
    }
  };

  // Modal view handlers
  const handleModalZoomIn = () => {
    setModalScale(prev => Math.min(prev + 0.2, 5));
  };

  const handleModalZoomOut = () => {
    setModalScale(prev => Math.max(prev - 0.2, 0.3));
  };

  const handleModalFitToScreen = () => {
    const container = modalContainerRef.current;
    if (container && svgDimensions.width && svgDimensions.height) {
      const { clientWidth, clientHeight } = container;
      const availableWidth = Math.max(clientWidth - 64, 100);
      const availableHeight = Math.max(clientHeight - 64, 200);
      const fitScale = Math.min(
        availableWidth / svgDimensions.width,
        availableHeight / svgDimensions.height,
        4,
      );
      const clamped = Math.min(Math.max(fitScale, 0.3), 4);
      setModalScale(clamped);
    } else {
      setModalScale(1.5);
    }
    setModalPosition({ x: 0, y: 0 });
  };

  const handleOpenFullscreen = () => {
    setIsFullscreen(true);
    setModalPosition({ x: 0, y: 0 });
    
    // Calculate optimal scale to fit viewport
    if (svgDimensions.width && svgDimensions.height) {
      const viewportWidth = window.innerWidth * 0.9; // 90% of viewport for padding
      const viewportHeight = window.innerHeight * 0.9;
      const scaleX = viewportWidth / svgDimensions.width;
      const scaleY = viewportHeight / svgDimensions.height;
      const optimalScale = Math.min(scaleX, scaleY, 3); // Cap at 3x
      setModalScale(Math.max(optimalScale, 0.5)); // Minimum 0.5x
    } else {
      setModalScale(1.5);
    }
  };

  const handleCloseFullscreen = () => {
    setIsFullscreen(false);
  };

  const handleModalMouseDown = (e) => {
    if (e.button === 0) {
      setIsModalDragging(true);
      setModalDragStart({
        x: e.clientX - modalPosition.x,
        y: e.clientY - modalPosition.y,
      });
    }
  };

  const handleModalMouseMove = (e) => {
    if (isModalDragging) {
      setModalPosition({
        x: e.clientX - modalDragStart.x,
        y: e.clientY - modalDragStart.y,
      });
    }
  };

  const handleModalMouseUp = () => {
    setIsModalDragging(false);
  };

  const handleModalWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setModalScale(prev => Math.min(Math.max(prev + delta, 0.3), 5));
    }
  };

  const handleDownload = () => {
    if (!svg) return;
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mermaid-diagram-${Date.now()}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
      return () => container.removeEventListener('wheel', handleWheel);
    }
  }, []);

  // Auto-fit the diagram to the available width/height on initial render
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !svgDimensions.width || !svgDimensions.height) return;

    const { clientWidth, clientHeight } = container;
    const availableWidth = Math.max(clientWidth - 32, 100); // account for padding and buttons
    const availableHeight = Math.max(clientHeight - 32, 200);
    const fitScale = Math.min(
      availableWidth / svgDimensions.width,
      availableHeight / svgDimensions.height,
      1.5,
    );
    const clamped = Math.min(Math.max(fitScale, 0.5), 2);
    setScale(clamped);
    setPosition({ x: 0, y: 0 });
  }, [svgDimensions.width, svgDimensions.height]);

  useEffect(() => {
    const container = modalContainerRef.current;
    if (container && isFullscreen) {
      container.addEventListener('wheel', handleModalWheel, { passive: false });
      return () => container.removeEventListener('wheel', handleModalWheel);
    }
  }, [isFullscreen]);

  if (error) {
    // Handle both old string format and new object format
    const errorDisplay = typeof error === 'string' ? error : error.display;
    const errorFull = typeof error === 'string' ? error : error.full;
    
    const errorInfo = {
      message: errorFull,  // Pass full error with context to LLM
      mermaidCode: chart,
      blockIndex,
      lineNumber: parseErrorLineNumber(errorDisplay),
    };

    return (
      <Alert 
        severity="error" 
        sx={{ 
          my: 2,
          '& .MuiAlert-message': {
            width: '100%',
          }
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, width: '100%' }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
              Failed to render Mermaid diagram
            </Typography>
            <Typography 
              variant="body2" 
              component="pre"
              sx={{ 
                fontFamily: 'monospace',
                fontSize: '11px',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                m: 0,
                opacity: 0.9,
              }}
            >
              {errorFull}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
            {isFixing ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={18} />
                <Typography variant="body2" color="text.secondary">Fixing...</Typography>
              </Box>
            ) : (
              <>
                {onQuickFix && (
                  <Button 
                    size="small" 
                    variant="contained"
                    startIcon={<AutoFixHighIcon />}
                    onClick={() => onQuickFix(errorInfo)}
                    sx={{ 
                      textTransform: 'none',
                      fontSize: '12px',
                      py: 0.5,
                    }}
                  >
                    Quick Fix
                  </Button>
                )}
                {onNavigateToError && (
                  <Button 
                    size="small" 
                    variant="outlined"
                    startIcon={<EditIcon />}
                    onClick={() => onNavigateToError(errorInfo)}
                    sx={{ 
                      textTransform: 'none',
                      fontSize: '12px',
                      py: 0.5,
                    }}
                  >
                    Edit
                  </Button>
                )}
              </>
            )}
          </Box>
        </Box>
      </Alert>
    );
  }

  return (
    <Box
      sx={{
        position: 'relative',
        my: 3,
        borderRadius: 2,
        bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      {/* Control Panel */}
      <Box
        sx={{
          position: 'absolute',
          top: 12,
          right: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 0.5,
          zIndex: 10,
          bgcolor: mode === 'dark' ? 'rgba(14, 19, 29, 0.9)' : 'rgba(255, 255, 255, 0.9)',
          borderRadius: 1,
          padding: 0.5,
          border: '1px solid',
          borderColor: 'divider',
          backdropFilter: 'blur(8px)',
        }}
      >
        <Tooltip title="Zoom In (Ctrl + Scroll)" placement="left">
          <IconButton
            size="small"
            onClick={handleZoomIn}
            disabled={scale >= 3}
            sx={{
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              '&:hover': { 
                bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
              },
            }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom Out (Ctrl + Scroll)" placement="left">
          <IconButton
            size="small"
            onClick={handleZoomOut}
            disabled={scale <= 0.5}
            sx={{
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              '&:hover': { 
                bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
              },
            }}
          >
            <RemoveIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Reset View" placement="left">
          <IconButton
            size="small"
            onClick={handleFitToScreen}
            sx={{
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              '&:hover': { 
                bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
              },
            }}
          >
            <FitScreenIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Download SVG" placement="left">
          <IconButton
            size="small"
            onClick={handleDownload}
            sx={{
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              '&:hover': { 
                bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
              },
            }}
          >
            <DownloadIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Expand Fullscreen" placement="left">
          <IconButton
            size="small"
            onClick={handleOpenFullscreen}
            sx={{
              color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
              '&:hover': { 
                bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
              },
            }}
          >
            <FullscreenIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Diagram Canvas */}
      <Box
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        sx={{
          width: '100%',
          height: 500,
          overflow: 'hidden',
          cursor: isDragging ? 'grabbing' : 'grab',
          userSelect: 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 2,
          '& > .mermaid-diagram-inner': {
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          },
          '& svg': {
            display: 'block',
            maxWidth: '100%',
            height: 'auto',
          },
        }}
      >
        <Box
          className="mermaid-diagram-inner"
          dangerouslySetInnerHTML={{ __html: svg }}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
          }}
        />
      </Box>

      {/* Fullscreen Modal */}
      <Dialog
        open={isFullscreen}
        onClose={handleCloseFullscreen}
        maxWidth={false}
        fullScreen
        sx={{
          '& .MuiDialog-paper': {
            bgcolor: mode === 'dark' ? '#0E131D' : '#F8FCFF',
            m: 0,
          },
        }}
      >
        <DialogContent
          sx={{
            p: 0,
            position: 'relative',
            overflow: 'hidden',
            height: '100vh',
          }}
        >
          {/* Modal Close Button */}
          <IconButton
            onClick={handleCloseFullscreen}
            sx={{
              position: 'absolute',
              top: 16,
              left: 16,
              zIndex: 1300,
              bgcolor: mode === 'dark' ? 'rgba(14, 19, 29, 0.9)' : 'rgba(255, 255, 255, 0.9)',
              border: '1px solid',
              borderColor: 'divider',
              backdropFilter: 'blur(8px)',
              '&:hover': {
                bgcolor: mode === 'dark' ? 'rgba(14, 19, 29, 1)' : 'rgba(255, 255, 255, 1)',
              },
            }}
          >
            <CloseIcon sx={{ color: mode === 'dark' ? '#6ae8fa' : '#C428DD' }} />
          </IconButton>

          {/* Modal Control Panel */}
          <Box
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 0.5,
              zIndex: 1300,
              bgcolor: mode === 'dark' ? 'rgba(14, 19, 29, 0.9)' : 'rgba(255, 255, 255, 0.9)',
              borderRadius: 1,
              padding: 0.5,
              border: '1px solid',
              borderColor: 'divider',
              backdropFilter: 'blur(8px)',
            }}
          >
            <Tooltip title="Zoom In (Ctrl + Scroll)" placement="left">
              <IconButton
                size="small"
                onClick={handleModalZoomIn}
                disabled={modalScale >= 5}
                sx={{
                  color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
                  '&:hover': { 
                    bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
                  },
                }}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out (Ctrl + Scroll)" placement="left">
              <IconButton
                size="small"
                onClick={handleModalZoomOut}
                disabled={modalScale <= 0.3}
                sx={{
                  color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
                  '&:hover': { 
                    bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
                  },
                }}
              >
                <RemoveIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Reset View" placement="left">
              <IconButton
                size="small"
                onClick={handleModalFitToScreen}
                sx={{
                  color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
                  '&:hover': { 
                    bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
                  },
                }}
              >
                <FitScreenIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download SVG" placement="left">
              <IconButton
                size="small"
                onClick={handleDownload}
                sx={{
                  color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
                  '&:hover': { 
                    bgcolor: mode === 'dark' ? 'rgba(106, 232, 250, 0.1)' : 'rgba(196, 40, 221, 0.1)' 
                  },
                }}
              >
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Modal Diagram Canvas */}
          <Box
            ref={modalContainerRef}
            onMouseDown={handleModalMouseDown}
            onMouseMove={handleModalMouseMove}
            onMouseUp={handleModalMouseUp}
            onMouseLeave={handleModalMouseUp}
            sx={{
              width: '100%',
              height: '100%',
              overflow: 'hidden',
              cursor: isModalDragging ? 'grabbing' : 'grab',
              userSelect: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              p: 4,
              '& > .mermaid-diagram-inner': {
                transform: `translate(${modalPosition.x}px, ${modalPosition.y}px) scale(${modalScale})`,
                transformOrigin: 'center center',
                transition: isModalDragging ? 'none' : 'transform 0.1s ease-out',
              },
              '& svg': {
                display: 'block',
                maxWidth: '100%',
                height: 'auto',
              },
            }}
          >
            <Box
              className="mermaid-diagram-inner"
              dangerouslySetInnerHTML={{ __html: svg }}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
              }}
            />
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default MermaidDiagram;
