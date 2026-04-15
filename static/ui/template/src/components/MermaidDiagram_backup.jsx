import { useEffect, useRef, useState } from 'react';
import { Box, Alert, IconButton, Tooltip, Dialog, DialogContent, IconButton as CloseButton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import DownloadIcon from '@mui/icons-material/Download';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import CloseIcon from '@mui/icons-material/Close';
import mermaid from 'mermaid';

// Initialize Mermaid with configuration
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: '"Montserrat", Roboto, Arial, sans-serif',
  fontSize: 14,
  themeVariables: {
    primaryColor: '#6ae8fa',
    primaryTextColor: '#fff',
    primaryBorderColor: '#3B3E46',
    lineColor: '#6ae8fa',
    secondaryColor: '#181F2A',
    tertiaryColor: '#0E131D',
  },
});

const MermaidDiagram = ({ chart, mode = 'dark' }) => {
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const [error, setError] = useState(null);
  const [svg, setSvg] = useState(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [modalScale, setModalScale] = useState(1);
  const [modalPosition, setModalPosition] = useState({ x: 0, y: 0 });
  const [isModalDragging, setIsModalDragging] = useState(false);
  const [modalDragStart, setModalDragStart] = useState({ x: 0, y: 0 });
  const modalContainerRef = useRef(null);

  useEffect(() => {
    // Update theme based on mode
    mermaid.initialize({
      startOnLoad: false,
      theme: mode === 'dark' ? 'dark' : 'default',
      securityLevel: 'loose',
      fontFamily: '"Montserrat", Roboto, Arial, sans-serif',
      fontSize: 14,
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
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg: renderedSvg } = await mermaid.render(id, chart);
        setSvg(renderedSvg);
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        setError(err.message || 'Failed to render diagram');
      }
    };

    renderDiagram();
  }, [chart, mode]);

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };

  const handleFitToScreen = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleModalFitToScreen = () => {
    setModalScale(1);
    setModalPosition({ x: 0, y: 0 });
  };

  const handleModalZoomIn = () => {
    setModalScale(prev => Math.min(prev + 0.2, 5));
  };

  const handleModalZoomOut = () => {
    setModalScale(prev => Math.max(prev - 0.2, 0.3));
  };

  const handleOpenFullscreen = () => {
    setIsFullscreen(true);
    setModalScale(1);
    setModalPosition({ x: 0, y: 0 });
  };

  const handleCloseFullscreen = () => {
    setIsFullscreen(false);
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

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
      return () => container.removeEventListener('wheel', handleWheel);
    }
  }, []);

  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        Failed to render Mermaid diagram: {error}
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
      </Box>IconButton>
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

          dangerouslySetInnerHTML={{ __html: svg }}
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
          <CloseButton
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
          </CloseButton>

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
            }}
          >
            <Box
              sx={{
                transform: `translate(${modalPosition.x}px, ${modalPosition.y}px) scale(${modalScale})`,
                transformOrigin: 'center center',
                transition: isModalDragging ? 'none' : 'transform 0.1s ease-out',
                display: 'inline-block',
                p: 4,
                '& svg': {
                  display: 'block',
                  maxWidth: 'none',
                  height: 'auto',
                },
              }}
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default MermaidDiagram;
          width: '100%',
          height: '500px',
          overflow: 'hidden',
          cursor: isDragging ? 'grabbing' : 'grab',
          userSelect: 'none',
          position: 'relative',
        }}
      >
        <Box
          ref={contentRef}
          sx={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: '0 0',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
            display: 'inline-block',
            p: 2,
            '& svg': {
              display: 'block',
              maxWidth: 'none',
              height: 'auto',
            },
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </Box>
    </Box>
  );
};

export default MermaidDiagram;
