import { memo, useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { v4 as uuidv4 } from 'uuid';
import * as Diff from 'diff';
import { getEnvVar } from './utils/env';
import ThinkingStepsPanel from './components/ThinkingStepsPanel';
import ChatDrawer, { DRAWER_WIDTH } from './components/ChatDrawer';
import SlotsIndicator from './components/SlotsIndicator';
import useSlots from './hooks/useSlots';
import { useManualSocket, sioEvents, SocketMessageType, emitSocketEvent, getSocket, getSocketId } from './hooks/useSocket';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Button,
  TextField,
  MenuItem,
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  LinearProgress,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Collapse,
  Drawer,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  Grid,
  Tab,
  Tabs,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import {
  AutoAwesome as GenerateIcon,
  ExpandMore as ExpandMoreIcon,
  ChevronRight as ChevronRightIcon,
  Description as DescriptionIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Close as CloseIcon,
  Settings as SettingsIcon,
  Delete as DeleteIcon,
  Undo as UndoIcon,
  Visibility as VisibilityIcon,
  CompareArrows as CompareArrowsIcon,
  Refresh as RefreshIcon,
  Stop as StopIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import MermaidDiagram from './components/MermaidDiagram';
import CodeBlock from './components/CodeBlock';
import CodeMirror from '@uiw/react-codemirror';
import { markdown as markdownLang } from '@codemirror/lang-markdown';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView } from '@codemirror/view';

const SIDEBAR_WIDTH = 280;

// EliteA Design Tokens
const designTokens = {
  typography: {
    fontFamily: '"Montserrat", Roboto, Arial, sans-serif',
    headingMedium: {
      fontWeight: 600,
      fontSize: '16px',
      lineHeight: '24px',
    },
    headingSmall: {
      fontWeight: 600,
      fontSize: '14px',
      lineHeight: '24px',
    },
    bodyMedium: {
      fontWeight: 400,
      fontSize: '14px',
      lineHeight: '24px',
    },
    bodySmall: {
      fontWeight: 400,
      fontSize: '12px',
      lineHeight: '16px',
    },
    labelMedium: {
      fontWeight: 500,
      fontSize: '14px',
      lineHeight: '24px',
    },
  },
  colors: {
    dark: {
      primary: '#6ae8fa',
      background: {
        default: '#0E131D',
        secondary: '#181F2A',
        hover: 'rgba(255, 255, 255, 0.06)',
        selected: 'rgba(41, 184, 245, 0.15)',
      },
      text: {
        primary: 'rgba(255, 255, 255, 0.9)',
        secondary: 'rgba(255, 255, 255, 0.7)',
        tertiary: 'rgba(255, 255, 255, 0.5)',
      },
      border: '#3B3E46',
    },
    light: {
      primary: '#C428DD',
      background: {
        default: '#F8FCFF',
        secondary: '#FFFFFF',
        hover: 'rgba(61, 68, 86, 0.06)',
        selected: 'rgba(99, 144, 254, 0.12)',
      },
      text: {
        primary: 'rgba(0, 0, 0, 0.87)',
        secondary: 'rgba(0, 0, 0, 0.6)',
        tertiary: 'rgba(0, 0, 0, 0.38)',
      },
      border: '#E1E5E9',
    },
  },
};

// Sanitize bucket name to match allowed characters
function sanitizeBucketName(name) {
  // Allowed characters: lowercase letters, numbers, hyphens, underscores
  // Must start with letter or number, 3-63 characters
  // Note: S3/MinIO allows underscores in bucket names
  return name
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-')
    .replace(/^-+|-+$/g, '')
    .substring(0, 63);
}

// Validate bucket name - returns error message or null if valid
function validateBucketName(name) {
  if (!name) return 'Bucket name is required';
  if (!/^[a-z][a-z0-9-]*$/.test(name)) {
    return 'Bucket name must start with a lowercase letter and contain only lowercase letters, numbers, and hyphens';
  }
  if (name.length < 3) return 'Bucket name must be at least 3 characters';
  if (name.length > 63) return 'Bucket name must be at most 63 characters';
  if (name.includes('--')) return 'Bucket name cannot contain consecutive hyphens';
  if (name.endsWith('-')) return 'Bucket name cannot end with a hyphen';
  return null;
}

/**
 * Normalize a repository string to the wiki_id format prefix.
 * wiki_id format: {owner}--{repo}--{branch}
 * This returns just {owner}--{repo} prefix for matching against wiki_id.
 * 
 * @param {string} repo - Repository string like "owner/repo" or "fmtlib/fmt"
 * @returns {string|null} - Normalized prefix like "owner--repo" or null if invalid
 */
function normalizeRepoToWikiIdPrefix(repo) {
  if (!repo || typeof repo !== 'string') return null;
  
  // Split by '/' to get owner and repo name
  const parts = repo.split('/');
  if (parts.length < 2) return null;
  
  const owner = parts[0].toLowerCase().replace(/_/g, '-').replace(/\./g, '-');
  const repoName = parts[1].toLowerCase().replace(/_/g, '-').replace(/\./g, '-');
  
  return `${owner}--${repoName}`;
}

/**
 * Check if a manifest's wiki_id matches the configured repository.
 * Matches if wiki_id starts with the normalized repo prefix.
 * 
 * @param {object} manifest - Manifest object with wiki_id or canonical_repo_identifier
 * @param {string} configuredRepo - Configured repository like "owner/repo"
 * @returns {boolean} - True if manifest matches the configured repo
 */
function manifestMatchesRepo(manifest, configuredRepo) {
  if (!manifest || !configuredRepo) return false;
  
  const expectedPrefix = normalizeRepoToWikiIdPrefix(configuredRepo);
  if (!expectedPrefix) return false;
  
  // Check wiki_id first (preferred method)
  if (manifest.wiki_id) {
    // wiki_id format: owner--repo--branch
    // Check if it starts with owner--repo--
    return manifest.wiki_id.toLowerCase().startsWith(expectedPrefix + '--');
  }
  
  // Fallback: check canonical_repo_identifier
  if (manifest.canonical_repo_identifier) {
    const manifestPrefix = normalizeRepoToWikiIdPrefix(manifest.canonical_repo_identifier);
    return manifestPrefix === expectedPrefix;
  }
  
  return false;
}

// Format settings field label: remove underscores, capitalize words
// Custom label overrides for specific field keys
const FIELD_LABEL_OVERRIDES = {
  'toolkit_configuration_code_toolkit': 'Code Toolkit ID',
  'code_toolkit': 'Code Toolkit ID',
};

function formatFieldLabel(key) {
  // Check for custom label override first
  if (FIELD_LABEL_OVERRIDES[key]) {
    return FIELD_LABEL_OVERRIDES[key];
  }
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
}

// Fields that should be read-only (non-editable)
const READ_ONLY_FIELDS = [
  'toolkit_configuration_llm_model',
  'toolkit_configuration_embedding_model',
  'llm_model',
  'embedding_model'
];

// Default values for settings fields
const FIELD_DEFAULTS = {
  toolkit_configuration_max_tokens: 64000,
  max_tokens: 64000
};

// Storage key prefix for persisting generation state (per-toolkit)
const GENERATION_STATE_KEY_PREFIX = 'deepwiki_generation_state';

// Build toolkit-specific storage key
function getGenerationStateKey(projectId, toolkitId) {
  if (projectId && toolkitId) {
    return `${GENERATION_STATE_KEY_PREFIX}.${projectId}.${toolkitId}`;
  }
  // Fallback to generic key (legacy)
  return GENERATION_STATE_KEY_PREFIX;
}

// Save generation state to localStorage (toolkit-specific)
function saveGenerationState(state) {
  try {
    const key = getGenerationStateKey(state.projectId, state.toolkitId);
    localStorage.setItem(key, JSON.stringify(state));
  } catch (e) {
    console.warn('[DeepWiki] Failed to save generation state:', e);
  }
}

// Update (merge) generation state in localStorage
function updateGenerationState(projectId, toolkitId, patch) {
  try {
    const key = getGenerationStateKey(projectId, toolkitId);
    const stored = localStorage.getItem(key);
    const prev = stored ? JSON.parse(stored) : {};
    const next = { ...(prev || {}), ...(patch || {}) };
    localStorage.setItem(key, JSON.stringify(next));
  } catch (e) {
    console.warn('[DeepWiki] Failed to update generation state:', e);
  }
}

// Load generation state from localStorage (toolkit-specific)
function loadGenerationState(projectId, toolkitId) {
  try {
    const key = getGenerationStateKey(projectId, toolkitId);
    const stored = localStorage.getItem(key);
    if (stored) {
      const state = JSON.parse(stored);
      // Check if state is still valid (not older than 4 hours)
      if (state.startTime && (Date.now() - state.startTime) < 4 * 60 * 60 * 1000) {
        return state;
      }
      // State is too old, clear it
      clearGenerationState(projectId, toolkitId);
    }
  } catch (e) {
    console.warn('[DeepWiki] Failed to load generation state:', e);
  }
  return null;
}

// Clear generation state from localStorage (toolkit-specific)
function clearGenerationState(projectId, toolkitId) {
  try {
    const key = getGenerationStateKey(projectId, toolkitId);
    localStorage.removeItem(key);
  } catch (e) {
    console.warn('[DeepWiki] Failed to clear generation state:', e);
  }
}

// Get bucket name from toolkit with fallback
// Hardcoded to 'wiki_artifacts' as the default bucket for all wiki artifacts
function getBucketName(toolkit) {
  if (!toolkit) return 'wiki_artifacts';
  // Check settings first (user-configured values), then toolkit_config, then fallback to hardcoded default
  return toolkit.settings?.toolkit_configuration_bucket || 
         toolkit.settings?.bucket || 
         toolkit.toolkit_config?.bucket || 
         'wiki_artifacts';
}

// Derive configured repository from toolkit or settings
// This function checks direct fields only - toolkit reference resolution is handled separately
function getConfiguredRepo(toolkit, settings, resolvedRepoName = null) {
  // If we have a resolved repository name from a toolkit reference, use it
  if (resolvedRepoName) {
    return resolvedRepoName;
  }

  const cfg = toolkit?.toolkit_config || {};
  const set = settings || toolkit?.settings || {};
  return (
    set.toolkit_configuration_github_repository ||
    set.github_repository ||
    set.repository ||
    set.repo ||
    cfg.github_repository ||
    cfg.repository ||
    null
  );
}

function DeepWikiApp() {
  const [mode, setMode] = useState('light');
  const [toolkit, setToolkit] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [projectId, setProjectId] = useState(null);
  const [toolkitId, setToolkitId] = useState(null);
  const [isDevelopment, setIsDevelopment] = useState(false);

  // DeepWiki specific state
  const [wikiStructure, setWikiStructure] = useState(null);
  const [artifactsList, setArtifactsList] = useState(null);
  const [currentPage, setCurrentPage] = useState(null);
  const [pageContent, setPageContent] = useState('');
  const [pageHeadings, setPageHeadings] = useState([]); // TOC from markdown
  const [pageLoading, setPageLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});
  const [editorOpen, setEditorOpen] = useState(false);
  const [editorContent, setEditorContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [savingEdit, setSavingEdit] = useState(false);
  const [saveFeedback, setSaveFeedback] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsText, setSettingsText] = useState('');
  const [settingsData, setSettingsData] = useState({});
  const [savingSettings, setSavingSettings] = useState(false);
  const [settingsError, setSettingsError] = useState(null);
  const [settingsSaved, setSettingsSaved] = useState(false);
  const [deletingWiki, setDeletingWiki] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [updateDialogOpen, setUpdateDialogOpen] = useState(false);
  // Planner mode (mirrors the wikis Generate dialog): "deepagents" = the
  // DeepAgents-driven structure planner, "cluster" = graph-clustering
  // planner. The cluster planner exposes the optional "smart skip tests"
  // toggle. The string values are the canonical names used by
  // ``DEEPWIKI_STRUCTURE_PLANNER`` (helm values / docker-compose) and the
  // backend resolver — keep them stable across the stack.
  //
  // Default = "cluster": deterministic on any scale (a 4-file repo yields
  // ~2 pages reliably; a large monorepo scales linearly with cluster
  // count). DeepAgents over-segments small repos because the agent loop
  // optimises for thoroughness rather than minimality.
  const [plannerMode, setPlannerMode] = useState('cluster');
  const [excludeTests, setExcludeTests] = useState(true);
  const [lastModifiedDate, setLastModifiedDate] = useState(null);
  const [resolvedRepoName, setResolvedRepoName] = useState(null);

  // Wiki versioning state
  const [bucketArtifacts, setBucketArtifacts] = useState([]);
  const [wikiManifests, setWikiManifests] = useState([]);
  const [selectedWikiManifestName, setSelectedWikiManifestName] = useState('');
  const [wikiVersionMode, setWikiVersionMode] = useState('legacy'); // 'manifest' | 'legacy'
  const [legacyVersionLabel, setLegacyVersionLabel] = useState('Latest (legacy)');
  const [repoIdentifierOverride, setRepoIdentifierOverride] = useState(null);
  const [analysisKeyOverride, setAnalysisKeyOverride] = useState(null);

  // Thinking steps state for progress tracking
  const [thinkingSteps, setThinkingSteps] = useState([]);
  const [generationElapsedTime, setGenerationElapsedTime] = useState(0);
  const generationStartTimeRef = useRef(null);
  const elapsedTimerRef = useRef(null);
  const currentStreamIdRef = useRef(null);
  const currentMessageIdRef = useRef(null);
  const currentTaskIdRef = useRef(null);
  const stepIdCounterRef = useRef(0);
  const [stoppingGeneration, setStoppingGeneration] = useState(false);
  // Track if current generation has errored (prevents success from overwriting)
  const generationErroredRef = useRef(false);
  
  // Reconnect timeout tracking
  const reconnectTimeoutRef = useRef(null);
  const lastSocketEventTimeRef = useRef(null);
  const isReconnectingRef = useRef(false);

  // Mermaid diagram fix state
  const [isFixingDiagram, setIsFixingDiagram] = useState(false);
  const [fixingBlockIndex, setFixingBlockIndex] = useState(null);
  const [lastFix, setLastFix] = useState(null); // { original, fixed, blockIndex }
  const [showDiffModal, setShowDiffModal] = useState(false);
  const [diffModalTab, setDiffModalTab] = useState(0); // 0 = code diff, 1 = preview
  const [fixFeedback, setFixFeedback] = useState(null); // { type: 'success' | 'error', message: string }
  const [lastErrorInfo, setLastErrorInfo] = useState(null); // Store for retry functionality
  // Track mermaid block count during render
  const mermaidBlockCounterRef = useRef(0);
  // Editor ref for scroll-to-line
  const editorRef = useRef(null);
  const [scrollToLine, setScrollToLine] = useState(null);

  // Chat drawer state
  const [chatOpen, setChatOpen] = useState(false);

  const bucketName = useMemo(() => {
    const name = getBucketName(toolkit);
    if (!name) return null;
    return sanitizeBucketName(name);
  }, [toolkit]);

  const authHeaders = useMemo(() => {
    return isDevelopment && import.meta.env.VITE_DEV_TOKEN
      ? { Authorization: `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
      : {};
  }, [isDevelopment]);

  // DeepWiki server base URL for slots endpoint
  const deepwikiBaseUrl = useMemo(() => {
    // In development, use VITE_DEEPWIKI_URL if available
    if (isDevelopment && import.meta.env.VITE_DEEPWIKI_URL) {
      return import.meta.env.VITE_DEEPWIKI_URL;
    }
    // In production via ui_host, use the API proxy path
    // UI is at /app/ui_host/deepwiki/ui/{project_id}/{toolkit_id}
    // DeepWiki API is proxied at /app/ui_host/deepwiki/api/{project_id}
    if (projectId) {
      return `${window.location.origin}/app/ui_host/deepwiki/api/${projectId}`;
    }
    // Fallback to origin (won't work but better than nothing)
    return window.location.origin;
  }, [isDevelopment, projectId]);

  // Cluster-wide slot availability
  const { 
    slots, 
    loading: slotsLoading, 
    error: slotsError, 
    slotsAvailable,
    refresh: refreshSlots 
  } = useSlots({
    baseUrl: deepwikiBaseUrl,
    pollingInterval: generating ? 10000 : 30000, // Poll faster during generation
    enabled: !loading && !!toolkit && !!projectId,
  });

  const wikiVersionStorageKey = useMemo(() => {
    if (!projectId || !bucketName) return null;
    return `deepwiki.selected_manifest.${projectId}.${bucketName}`;
  }, [projectId, bucketName]);

  const manifestSelectValue = useMemo(() => {
    if (!Array.isArray(wikiManifests) || wikiManifests.length === 0) return 'legacy_latest';
    if (selectedWikiManifestName && wikiManifests.some(m => m.name === selectedWikiManifestName)) {
      return selectedWikiManifestName;
    }
    return wikiManifests[0].name;
  }, [selectedWikiManifestName, wikiManifests]);

  // Get the currently selected manifest object (for wiki_id, etc.)
  const currentManifest = useMemo(() => {
    if (!Array.isArray(wikiManifests) || wikiManifests.length === 0) return null;
    return wikiManifests.find(m => m.name === manifestSelectValue) || wikiManifests[0];
  }, [wikiManifests, manifestSelectValue]);

  // Extract URL parameters
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('theme');
    const toolkitIdParam = urlParams.get('toolkit_id');
    const projectIdParam = urlParams.get('project_id');
    
    if (themeParam === 'dark' || themeParam === 'light') {
      setMode(themeParam);
    }

    const pathParts = window.location.pathname.split('/');
    const uiHostIndex = pathParts.indexOf('ui_host');
    
    let extractedProjectId = null;
    let extractedToolkitId = toolkitIdParam; // Fallback to query param
    let devMode = false;
    
    if (uiHostIndex !== -1 && pathParts.length > uiHostIndex + 3) {
      // Production mode via ui_host
      // URL: /app/ui_host/deepwiki/ui/{project_id}/{toolkit_id}
      extractedProjectId = pathParts[uiHostIndex + 3]; // project_id
      // Extract toolkit_id from path if available, otherwise use query param
      if (pathParts.length > uiHostIndex + 4 && pathParts[uiHostIndex + 4]) {
        extractedToolkitId = pathParts[uiHostIndex + 4]; // toolkit_id from path
      }
      devMode = false;
    } else {
      devMode = true;
      extractedProjectId = projectIdParam || getEnvVar('VITE_DEFAULT_PROJECT_ID');
      extractedToolkitId = extractedToolkitId || getEnvVar('VITE_DEFAULT_TOOLKIT_ID');
    }

    setProjectId(extractedProjectId);
    setToolkitId(extractedToolkitId);
    setIsDevelopment(devMode);
  }, []);

  // Helper function to resolve toolkit reference and get repository name
  const resolveToolkitReference = async (toolkitRefId, headers) => {
    if (!toolkitRefId || typeof toolkitRefId !== 'number') {
      return null;
    }

    try {
      const refUrl = `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitRefId}`;
      const refResponse = await fetch(refUrl, { headers });

      if (!refResponse.ok) {
        console.warn(`Failed to fetch referenced toolkit ${toolkitRefId}`);
        return null;
      }

      const refToolkit = await refResponse.json();
      const refSettings = refToolkit?.settings || {};

      // Extract repository name from GitHub toolkit settings
      // GitHub toolkit stores repository as 'repository' field
      const repoName = refSettings.repository ||
                       refSettings.github_repository ||
                       refSettings.repo;

      return repoName || null;
    } catch (err) {
      console.warn('Error resolving toolkit reference:', err);
      return null;
    }
  };

  // Fetch toolkit data
  useEffect(() => {
    if (!projectId || !toolkitId) return;

    const fetchToolkit = async () => {
      try {
        setLoading(true);
        const url = `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`;

        const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
          ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
          : {};

        const response = await fetch(url, { headers });

        if (!response.ok) {
          throw new Error(`Failed to fetch toolkit: ${response.statusText}`);
        }

        const data = await response.json();
        setToolkit(data);
        const settingsPayload = data?.settings || data?.toolkit_config || {};
        setSettingsText(JSON.stringify(settingsPayload, null, 2));
        setSettingsData(settingsPayload);
        setSettingsSaved(false);

        // Check if there's a toolkit reference (code_toolkit) that needs to be resolved
        // The field is stored as toolkit_configuration_code_toolkit after schema transformation
        const codeToolkitRef = settingsPayload.toolkit_configuration_code_toolkit || settingsPayload.code_toolkit;
        let resolvedRepo = null;
        if (codeToolkitRef && typeof codeToolkitRef === 'number') {
          resolvedRepo = await resolveToolkitReference(codeToolkitRef, headers);
          setResolvedRepoName(resolvedRepo);
        } else {
          setResolvedRepoName(null);
        }
        
        // Compute configured repo for filtering manifests
        // Priority: resolved repo from code_toolkit > direct settings > null
        const currentConfiguredRepo = resolvedRepo || getConfiguredRepo(data, settingsPayload, null);

        // Determine bucket name using the getBucketName helper
        const bucketName = getBucketName(data);

        if (bucketName) {
          loadArtifactsList(bucketName, false, currentConfiguredRepo);
        }

        setError(null);
      } catch (err) {
        console.error('Error fetching toolkit:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchToolkit();
  }, [projectId, toolkitId, isDevelopment]);

  // Parse artifacts into structured sections based on filename pattern:
  // Legacy: section_document_timestamp.md
  // New folder structure: {wiki_id}/wiki_pages/{section}/{page}.md or {wiki_id}/wiki_pages/{page}.md
  const parseArtifactsIntoStructure = useCallback((artifacts) => {
    const sectionsMap = new Map();
    const standaloneFiles = [];
    
    artifacts.forEach(artifact => {
      const filename = artifact.name;
      
      // Skip non-markdown files
      if (!filename.endsWith('.md')) {
        return;
      }
      
      // Skip wiki_structure files
      if (filename.includes('wiki_structure_') || filename.includes('/analysis/')) {
        return;
      }
      
      // Check for new folder structure: {wiki_id}/wiki_pages/...
      if (filename.includes('/wiki_pages/')) {
        // New folder structure: extract section and page from path
        const wikiPagesIdx = filename.indexOf('/wiki_pages/');
        const pathAfterWikiPages = filename.substring(wikiPagesIdx + '/wiki_pages/'.length);
        
        // Check if there's a section folder: section/page.md or just page.md
        const pathParts = pathAfterWikiPages.split('/');
        
        if (pathParts.length >= 2) {
          // Has section folder: section/page.md
          const sectionName = pathParts[0];
          const pageName = pathParts.slice(1).join('/').replace(/\.md$/, '');
          
          const sectionTitle = sectionName
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          const documentTitle = pageName
            .replace(/_/g, ' ')
            .replace(/-/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          if (!sectionsMap.has(sectionName)) {
            sectionsMap.set(sectionName, {
              id: sectionName,
              title: sectionTitle,
              subsections: []
            });
          }
          
          sectionsMap.get(sectionName).subsections.push({
            id: pageName,
            title: documentTitle,
            filename: filename
          });
        } else {
          // No section folder: just page.md at wiki_pages root
          const pageName = pathParts[0].replace(/\.md$/, '');
          const documentTitle = pageName
            .replace(/_/g, ' ')
            .replace(/-/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          standaloneFiles.push({
            id: pageName,
            title: documentTitle,
            filename: filename,
            isStandalone: true
          });
        }
        return;
      }
      
      // Legacy flat structure: section_document_timestamp.md
      // Remove .md extension
      const nameWithoutExt = filename.replace(/\.md$/, '');
      
      // Split by underscore
      const parts = nameWithoutExt.split('_');
      
      // Need at least 3 parts for timestamp (YYYYMMDD_HHMMSS)
      if (parts.length < 3) {
        return;
      }
      
      // Check if last two parts are timestamp (YYYYMMDD and HHMMSS pattern)
      const potentialDate = parts[parts.length - 2];
      const potentialTime = parts[parts.length - 1];
      const isTimestamp = /^\d{8}$/.test(potentialDate) && /^\d{6}$/.test(potentialTime);
      
      if (!isTimestamp) {
        return;
      }
      
      // If only 3 parts (name_YYYYMMDD_HHMMSS), it's a standalone file without section
      if (parts.length === 3) {
        const documentName = parts[0];
        const documentTitle = documentName
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
        
        standaloneFiles.push({
          id: documentName,
          title: documentTitle,
          filename: filename,
          isStandalone: true
        });
        return;
      }
      
      // Otherwise, extract section and document parts
      // Format: section_document_YYYYMMDD_HHMMSS
      const sectionName = parts[0];
      const documentParts = parts.slice(1, -2);
      const documentName = documentParts.join('_');
      
      // Format section title (capitalize, replace hyphens with spaces)
      const sectionTitle = sectionName
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      // Format document title (capitalize, replace hyphens with spaces)
      const documentTitle = documentName
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      // Add to sections map
      if (!sectionsMap.has(sectionName)) {
        sectionsMap.set(sectionName, {
          id: sectionName,
          title: sectionTitle,
          subsections: []
        });
      }
      
      sectionsMap.get(sectionName).subsections.push({
        id: documentName,
        title: documentTitle,
        filename: filename
      });
    });
    
    // Convert map to array and prepend standalone files
    return [...standaloneFiles, ...Array.from(sectionsMap.values())];
  }, []);

  const parseManifestTimestamp = useCallback((value) => {
    if (!value) return null;
    try {
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? null : d;
    } catch {
      return null;
    }
  }, []);

  const extractTimestampFromArtifactName = useCallback((filename) => {
    if (typeof filename !== 'string') return null;
    const m = filename.match(/_(\d{8})_(\d{6})\.md$/);
    if (!m) return null;
    const ymd = m[1];
    const hms = m[2];
    const iso = `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}T${hms.slice(0, 2)}:${hms.slice(2, 4)}:${hms.slice(4, 6)}Z`;
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? null : d;
  }, []);

  const pickLegacyLatestRunArtifacts = useCallback((artifacts) => {
    const md = (artifacts || []).filter(a => typeof a?.name === 'string' && a.name.endsWith('.md'));
    const withTs = md
      .map(a => ({ a, ts: extractTimestampFromArtifactName(a.name) }))
      .filter(item => item.ts);

    if (withTs.length === 0) {
      return { artifacts: md, label: 'Latest (legacy)' };
    }

    // Cluster by time gaps to approximate per-run grouping.
    const sorted = withTs.sort((x, y) => x.ts - y.ts);
    const GAP_MS = 10 * 60 * 1000; // 10 minutes
    const runs = [];
    let current = [sorted[0]];
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1];
      const next = sorted[i];
      if (next.ts - prev.ts > GAP_MS) {
        runs.push(current);
        current = [next];
      } else {
        current.push(next);
      }
    }
    runs.push(current);

    const latestRun = runs[runs.length - 1];
    const start = latestRun[0].ts;
    const end = latestRun[latestRun.length - 1].ts;
    const label = `Latest (legacy) · ${start.toLocaleString()} – ${end.toLocaleString()}`;
    return { artifacts: latestRun.map(x => x.a), label };
  }, [extractTimestampFromArtifactName]);

  const loadWikiManifests = useCallback(
    async (bucket, artifacts) => {
      // Support both new folder structure ({wiki_id}/wiki_manifest_*.json) 
      // and legacy flat structure (wiki_manifest_*.json)
      const manifestFiles = (artifacts || []).filter(a => {
        if (typeof a?.name !== 'string') return false;
        const name = a.name;
        // New folder structure: {wiki_id}/wiki_manifest_*.json
        if (name.includes('/wiki_manifest_') && name.endsWith('.json')) return true;
        // Legacy flat structure: wiki_manifest_*.json
        if (name.startsWith('wiki_manifest_') && name.endsWith('.json')) return true;
        return false;
      });
      if (manifestFiles.length === 0) return [];

      const sanitizedBucket = sanitizeBucketName(bucket);

      const manifests = await Promise.all(
        manifestFiles.map(async (file) => {
          try {
            const url = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}/${file.name}`;
            const response = await fetch(url, { headers: authHeaders });
            if (!response.ok) return null;
            const text = await response.text();
            const parsed = JSON.parse(text);
            const createdAt = parseManifestTimestamp(parsed?.created_at) || (file.modified ? new Date(file.modified) : null);
            
            // Extract wiki_id from folder path or from manifest content
            let wiki_id = parsed?.wiki_id || null;
            if (!wiki_id && file.name.includes('/')) {
              wiki_id = file.name.split('/')[0];
            }
            
            return {
              name: file.name,
              wiki_id: wiki_id,
              wiki_title: parsed?.wiki_title || null,
              description: parsed?.description || null,
              wiki_version_id: parsed?.wiki_version_id || null,
              created_at: createdAt,
              commit_hash: parsed?.commit_hash || null,
              canonical_repo_identifier: parsed?.canonical_repo_identifier || null,
              analysis_key: parsed?.analysis_key || null,
              pages: Array.isArray(parsed?.pages) ? parsed.pages.filter(x => typeof x === 'string') : [],
              schema_version: parsed?.schema_version || 1,
            };
          } catch {
            return null;
          }
        })
      );

      return manifests.filter(Boolean).sort((a, b) => {
        const at = a.created_at?.getTime?.() || 0;
        const bt = b.created_at?.getTime?.() || 0;
        return bt - at;
      });
    },
    [authHeaders, parseManifestTimestamp, projectId]
  );

  const applyArtifactsToWiki = useCallback(
    async (bucket, artifacts) => {
      const sanitizedBucket = sanitizeBucketName(bucket);

      console.log('[DeepWiki] applyArtifactsToWiki called with', { bucket, artifactCount: artifacts?.length });
      const structuredSections = parseArtifactsIntoStructure(artifacts);
      console.log('[DeepWiki] parseArtifactsIntoStructure returned', { sectionsCount: structuredSections?.length, sections: structuredSections });

      if (structuredSections.length > 0) {
        setWikiStructure({ sections: structuredSections });
        const expanded = {};
        const autoExpandLimit = Number(import.meta.env.VITE_DEEPWIKI_AUTO_EXPAND_LIMIT || 200);
        const normalizedLimit = Number.isFinite(autoExpandLimit) && autoExpandLimit > 0
          ? autoExpandLimit
          : 200;
        const totalPages = structuredSections.reduce(
          (count, section) => count + (section.subsections?.length || 0),
          0
        );
        const shouldAutoExpand = totalPages <= normalizedLimit;

        structuredSections.forEach((section, index) => {
          expanded[section.id] = shouldAutoExpand || index === 0;
        });
        setExpandedSections(expanded);
        setArtifactsList(null);

        if (structuredSections[0]?.subsections?.[0]) {
          const firstPage = structuredSections[0].subsections[0];
          await loadPageFromArtifact(sanitizedBucket, firstPage);
        }
        return;
      }

      const mdFiles = (artifacts || [])
        .filter(file => file?.name?.endsWith('.md'))
        .map(file => ({
          id: file.name.replace(/\.md$/, ''),
          title: file.name
            .replace(/_\d{8}_\d{6}\.md$/, '') // Remove timestamp
            .replace(/\.md$/, '')
            .replace(/_/g, ' ')
            .replace(/-/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' '),
          filename: file.name,
          size: file.size,
          modified: file.modified,
        }));

      if (mdFiles.length === 0) {
        setWikiStructure(null);
        setArtifactsList(null);
        setCurrentPage(null);
        setPageContent('');
        setPageHeadings([]);
        return;
      }

      setWikiStructure(null);
      setArtifactsList(mdFiles);
      await loadPageFromArtifact(sanitizedBucket, mdFiles[0]);
    },
    [loadPageFromArtifact, parseArtifactsIntoStructure]
  );

  // Load artifacts and parse into structured sections based on filenames
  // When forceSelectLatest=true, always pick the newest manifest (used after generation completes)
  // configuredRepoParam: optional repository to filter manifests by (e.g., "owner/repo")
  const loadArtifactsList = async (bucket, forceSelectLatest = false, configuredRepoParam = null) => {
    try {
      const sanitizedBucket = sanitizeBucketName(bucket);
      const url = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
      const storageKey = projectId ? `deepwiki.selected_manifest.${projectId}.${sanitizedBucket}` : null;
      
      const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
        ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
        : {};

      const response = await fetch(url, { headers });
      
      if (response.ok) {
        const data = await response.json();
        const artifacts = data.rows || [];

        setBucketArtifacts(artifacts);
        
        // Find the most recent modified date from artifacts
        if (artifacts.length > 0) {
          const mostRecent = artifacts.reduce((latest, artifact) => {
            if (!artifact.modified) return latest;
            const artifactDate = new Date(artifact.modified);
            return !latest || artifactDate > latest ? artifactDate : latest;
          }, null);
          setLastModifiedDate(mostRecent);
        }
        
        const manifests = await loadWikiManifests(bucket, artifacts);
        console.log('[DeepWiki] loadWikiManifests returned', { manifestsCount: manifests?.length, manifests });
        
        // Filter manifests by configured repository if provided
        // This ensures we only show wikis for the currently configured repo, not other repos in the shared bucket
        let filteredManifests = manifests;
        if (configuredRepoParam) {
          filteredManifests = manifests.filter(m => manifestMatchesRepo(m, configuredRepoParam));
          console.log('[DeepWiki] Filtered manifests by configuredRepo', {
            configuredRepo: configuredRepoParam,
            beforeFilter: manifests.length,
            afterFilter: filteredManifests.length,
            filteredOut: manifests.filter(m => !manifestMatchesRepo(m, configuredRepoParam)).map(m => m.wiki_id)
          });
        }
        
        setWikiManifests(filteredManifests);

        if (filteredManifests.length > 0) {
          setWikiVersionMode('manifest');

          let desired = '';
          // If forceSelectLatest is true (e.g., after generation), always pick the newest manifest
          if (!forceSelectLatest && storageKey) {
            try {
              const stored = localStorage.getItem(storageKey);
              if (stored && filteredManifests.some(m => m.name === stored)) {
                desired = stored;
              }
            } catch {
              // ignore
            }
          }
          if (!desired) {
            desired = filteredManifests[0].name;
          }

          // Save the selected manifest to localStorage
          if (storageKey) {
            try {
              localStorage.setItem(storageKey, desired);
            } catch {
              // ignore
            }
          }

          setSelectedWikiManifestName(desired);
          const selected = filteredManifests.find(m => m.name === desired) || filteredManifests[0];

          setRepoIdentifierOverride(selected?.canonical_repo_identifier || null);
          setAnalysisKeyOverride(selected?.analysis_key || null);

          const allowed = new Set(selected?.pages || []);
          const filtered = artifacts.filter(a => allowed.has(a.name));
          console.log('[DeepWiki] Filtering artifacts by manifest pages', { 
            allowedPages: selected?.pages?.length, 
            totalArtifacts: artifacts?.length, 
            filteredCount: filtered?.length,
            sampleAllowed: Array.from(allowed).slice(0, 3),
            sampleArtifacts: artifacts?.slice(0, 3).map(a => a.name)
          });
          await applyArtifactsToWiki(bucket, filtered);
          return;
        }

        // Legacy mode: no matching manifests found for this repository
        // Filter artifacts by expected wiki_id prefix if we have a configured repo
        setWikiVersionMode('legacy');
        setSelectedWikiManifestName('');
        setRepoIdentifierOverride(null);
        setAnalysisKeyOverride(null);

        // Compute expected wiki_id prefix from configured repo
        const expectedPrefix = configuredRepoParam ? normalizeRepoToWikiIdPrefix(configuredRepoParam) : null;
        
        // Filter artifacts to only include those belonging to this repo
        let repoArtifacts = artifacts;
        if (expectedPrefix) {
          // wiki_id format: owner--repo--branch, so artifacts would be at owner--repo--branch/...
          // Filter to only include artifacts starting with the expected prefix
          repoArtifacts = artifacts.filter(a => {
            if (!a?.name) return false;
            // Check if artifact path starts with expected wiki_id pattern
            // e.g., "fmtlib--fmt--main/wiki_pages/..." should match "fmtlib--fmt"
            const artifactPath = a.name.toLowerCase();
            return artifactPath.startsWith(expectedPrefix + '--');
          });
          console.log('[DeepWiki] Filtered legacy artifacts by repo prefix', {
            expectedPrefix,
            beforeFilter: artifacts.length,
            afterFilter: repoArtifacts.length
          });
        }
        
        // If no artifacts match this repo, show empty state
        if (repoArtifacts.length === 0) {
          console.log('[DeepWiki] No artifacts found for configured repo, showing empty state');
          setWikiStructure(null);
          setArtifactsList(null);
          setLegacyVersionLabel('');
          return;
        }

        const legacy = pickLegacyLatestRunArtifacts(repoArtifacts);
        setLegacyVersionLabel(legacy.label);
        await applyArtifactsToWiki(bucket, legacy.artifacts);
      }
    } catch (err) {
      console.error('Error loading artifacts list:', err);
    }
  };

  const handleWikiVersionSelect = useCallback(
    async (e) => {
      if (generating) return;
      const nextName = e.target.value;
      setSelectedWikiManifestName(nextName);

      try {
        if (wikiVersionStorageKey) {
          localStorage.setItem(wikiVersionStorageKey, nextName);
        }
      } catch {
        // ignore
      }

      const selected = wikiManifests.find(m => m.name === nextName);
      if (!selected || !bucketName) return;

      setRepoIdentifierOverride(selected?.canonical_repo_identifier || null);
      setAnalysisKeyOverride(selected?.analysis_key || null);

      const allowed = new Set(selected?.pages || []);
      const filtered = (bucketArtifacts || []).filter(a => allowed.has(a.name));
      await applyArtifactsToWiki(bucketName, filtered);
    },
    [applyArtifactsToWiki, bucketArtifacts, bucketName, generating, wikiManifests, wikiVersionStorageKey]
  );

  // Load individual wiki page (from structure)
  const loadPage = async (bucket, page) => {
    try {
      setPageLoading(true);
      setCurrentPage(page);
      
      const sanitizedBucket = sanitizeBucketName(bucket);
      const filename = page.filename || `${page.id}.md`;
      const url = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}/${filename}`;
      
      const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
        ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
        : {};

      const response = await fetch(url, { headers });
      
      if (response.ok) {
        const content = await response.text();
        const { tocItems, contentWithoutTOC } = extractTOCFromMarkdown(content);
        setOriginalContent(content);
        setEditorContent(content);
        setSaveFeedback(null);
        setPageContent(contentWithoutTOC);
        setPageHeadings(tocItems.length > 0 ? tocItems : extractHeadings(contentWithoutTOC));
      } else {
        setPageContent(`# ${page.title}\n\n*Content not available*`);
        setPageHeadings([]);
      }
    } catch (err) {
      console.error('Error loading page:', err);
      setPageContent(`# ${page.title}\n\n*Error loading content*`);
      setPageHeadings([]);
    } finally {
      setEditorOpen(false);
      setPageLoading(false);
    }
  };

  // Extract a markdown TOC block and return its items plus the markdown without that block.
  const extractTOCFromMarkdown = (markdown) => {
    const tocStartPattern = /^(?:#{1,6}\s*)?table of contents\s*$/i;
    const headingPattern = /^#{1,6}\s+/;
    const hrPattern = /^(?:-{3,}|\*{3,}|_{3,})\s*$/;
    const listPattern = /^\s*(?:[-*+]|\d+\.)\s+/;

    const lines = markdown.split('\n');
    const startIndex = lines.findIndex(line => tocStartPattern.test(line.trim()));

    if (startIndex === -1) {
      return { tocItems: [], contentWithoutTOC: markdown };
    }

    const tocLines = [];
    let endIndex = lines.length;

    for (let i = startIndex + 1; i < lines.length; i++) {
      const trimmed = lines[i].trim();

      // Skip a single blank line immediately after the TOC heading
      if (trimmed === '' && tocLines.length === 0) {
        continue;
      }

      if (hrPattern.test(trimmed)) {
        endIndex = i + 1; // drop the horizontal rule as part of the TOC block
        break;
      }

      const isListLine = listPattern.test(trimmed);
      const isBlank = trimmed === '';

      // Stop when a non-list, non-blank, non-hr line (likely the next section) is reached
      if (!isListLine && !isBlank) {
        endIndex = i;
        break;
      }

      if (headingPattern.test(trimmed)) {
        endIndex = i;
        break;
      }

      // Collect TOC list content (including blank lines between nested lists)
      if (isListLine || isBlank) {
        tocLines.push(lines[i]);
      }
    }

    const remainingLines = [...lines.slice(0, startIndex), ...lines.slice(endIndex)];
    const contentWithoutTOC = remainingLines
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^\s*\n/, '');

    const tocItems = tocLines
      .map(line => {
        const indent = (line.match(/^\s*/) || [''])[0].length;
        const linkMatch = line.match(/\[([^\]]+)\]\(([^)]+)\)/);
        if (!linkMatch) return null;
        const [, text, href] = linkMatch;
        const level = Math.max(1, Math.floor(indent / 2) + 1);
        const id = href.startsWith('#') ? href.slice(1) : href;
        return { level, text, id };
      })
      .filter(Boolean);

    return { tocItems, contentWithoutTOC };
  };

  // Extract headings from markdown content for TOC
  const extractHeadings = (markdown) => {
    const headingRegex = /^(#{1,6})\s+(.+)$/gm;
    const headings = [];
    let match;
    
    while ((match = headingRegex.exec(markdown)) !== null) {
      const level = match[1].length;
      const text = match[2].trim();
      const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
      headings.push({ level, text, id });
    }
    
    return headings;
  };

  // Load individual artifact file
  async function loadPageFromArtifact(bucket, artifact) {
    try {
      setPageLoading(true);
      setCurrentPage(artifact);
      
      const sanitizedBucket = sanitizeBucketName(bucket);
      const url = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}/${artifact.filename}`;
      
      const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
        ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
        : {};

      const response = await fetch(url, { headers });
      
      if (response.ok) {
        const content = await response.text();
        const { tocItems, contentWithoutTOC } = extractTOCFromMarkdown(content);
        setOriginalContent(content);
        setEditorContent(content);
        setSaveFeedback(null);
        setPageContent(contentWithoutTOC);
        setPageHeadings(tocItems.length > 0 ? tocItems : extractHeadings(contentWithoutTOC));
      } else {
        setPageContent(`# ${artifact.title}\n\n*Content not available*`);
        setPageHeadings([]);
      }
    } catch (err) {
      console.error('Error loading artifact:', err);
      setPageContent(`# ${artifact.title}\n\n*Error loading content*`);
      setPageHeadings([]);
    } finally {
      setEditorOpen(false);
      setPageLoading(false);
    }
  }

  // Check if wiki was updated less than 24 hours ago
  const isWikiFresh = () => {
    if (!lastModifiedDate) return false;
    const now = new Date();
    const hoursSinceUpdate = (now - lastModifiedDate) / (1000 * 60 * 60);
    return hoursSinceUpdate < 24;
  };

  // Show update confirmation dialog
  const handleUpdateWiki = () => {
    if (!toolkit || !projectId || !toolkitId) return;
    setUpdateDialogOpen(true);
  };

  const isWikiVersionSelectorLocked = generating;

  // Helper to cleanup generation state
  const cleanupGeneration = useCallback((unsubscribeFn, streamIdToLeave = null) => {
    // Leave the socket room for this stream (isolates events between tabs)
    const streamId = streamIdToLeave || currentStreamIdRef.current;
    if (streamId) {
      emitSocketEvent(sioEvents.test_toolkit_leave_room, {
        stream_id: streamId,
        event_name: 'test_toolkit_tool',
      });
    }
    
    setGenerating(false);
    setStoppingGeneration(false);
    currentStreamIdRef.current = null;
    currentMessageIdRef.current = null;
    currentTaskIdRef.current = null;
    generationStartTimeRef.current = null;
    isReconnectingRef.current = false;
    lastSocketEventTimeRef.current = null;
    // Note: Do NOT reset generationErroredRef here - let it persist until next generation starts
    // This ensures the error status stays visible after cleanup
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      // Clear both timeout and interval (they use the same ID space)
      clearTimeout(reconnectTimeoutRef.current);
      clearInterval(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    // Clear persisted generation state (use projectId/toolkitId from closure)
    clearGenerationState(projectId, toolkitId);
    // Unsubscribe from socket if function provided
    if (unsubscribeFn) {
      unsubscribeFn();
    }
  }, [projectId, toolkitId]);

  // Stop wiki generation
  const handleStopGeneration = useCallback(async () => {
    const taskId = currentTaskIdRef.current || loadGenerationState(projectId, toolkitId)?.taskId;
    if (!taskId || !projectId) {
      console.warn('[DeepWiki] No task ID available to stop');
      return;
    }
    
    setStoppingGeneration(true);
    
    try {
      const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
        ? { Authorization: `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
        : {};
      
      const response = await fetch(`/api/v2/elitea_core/application_task/prompt_lib/${projectId}/${taskId}`, {
        method: 'DELETE',
        headers,
      });
      
      if (response.ok || response.status === 204) {
        setGenerationStatus({ 
          status: 'error', 
          message: 'Generation stopped by user' 
        });
        setThinkingSteps(prev => [...prev, {
          id: `step-${++stepIdCounterRef.current}`,
          message: 'Generation stopped by user',
          timestamp: Date.now(),
          type: 'info',
        }]);
        cleanupGeneration(socketUnsubscribeRef.current);
        
        // Refresh artifacts in case a wiki was partially or fully generated
        const checkBucket = getBucketName(toolkit);
        if (checkBucket) {
          const sanitizedBucket = sanitizeBucketName(checkBucket);
          loadArtifactsList(sanitizedBucket, true, configuredRepo); // forceSelectLatest=true
        }
      } else {
        const errorText = await response.text();
        console.error('[DeepWiki] Failed to stop generation:', errorText);
        setStoppingGeneration(false);
      }
    } catch (err) {
      console.error('[DeepWiki] Error stopping generation:', err);
      setStoppingGeneration(false);
    }
  }, [projectId, toolkitId, isDevelopment, cleanupGeneration]);

  // Socket message handler for wiki generation events
  const handleWikiSocketMessage = useCallback((message) => {
    const { message_id, stream_id, type: socketMessageType, response_metadata, content } = message;

    const parseAgentResponseForError = (maybeContent, maybeMetadata) => {
      try {
        let contentObj = maybeContent && typeof maybeContent === 'object' ? maybeContent : null;
        let contentStr = typeof maybeContent === 'string' ? maybeContent : null;

        // Try to parse contentStr if it looks like JSON
        if (contentStr && (contentStr.trim().startsWith('{') || contentStr.trim().startsWith('['))) {
          try {
            const parsed = JSON.parse(contentStr);
            if (typeof parsed === 'object' && parsed !== null) {
              contentObj = parsed;
              // Extract message from parsed object
              contentStr = parsed.message || parsed.error || contentStr;
            }
          } catch (_parseErr) {
            // Not valid JSON, keep as string
          }
        }

        const status = contentObj?.status || maybeMetadata?.status;
        const errorCategory = contentObj?.error_category || maybeMetadata?.error_category;

        // Check for various error markers in the content string
        const contentLower = typeof contentStr === 'string' ? contentStr.toLowerCase() : '';
        const isServiceBusyMarker =
          (typeof contentStr === 'string' &&
            (contentStr.includes('[SERVICE_BUSY]') || 
             contentLower.includes('service busy') ||
             contentStr.includes('Max parallel wiki generations reached') ||
             contentStr.includes('slots taken')));
        
        // Check for inference_failed category or failed generation markers
        const isInferenceFailed = 
          errorCategory === 'inference_failed' ||
          (typeof contentStr === 'string' && contentStr.includes('inference_failed'));
        
        // Check for generation failure in the message text
        const isGenerationFailed =
          (typeof contentStr === 'string' &&
            (contentStr.includes('Generate_wiki failed') ||
             contentStr.includes('failed for model') ||
             contentLower.includes('runtimeerror') ||
             contentLower.includes('generation failed')));

        const isErrorStatus = status === 'Error' || errorCategory === 'service_busy' || isServiceBusyMarker || isInferenceFailed || isGenerationFailed;
        if (!isErrorStatus) {
          return { isError: false, message: null };
        }

        const activeWorkers = contentObj?.active_workers;
        const maxWorkers = contentObj?.max_workers;
        if (
          (errorCategory === 'service_busy' || isServiceBusyMarker) &&
          typeof activeWorkers === 'number' &&
          typeof maxWorkers === 'number'
        ) {
          return {
            isError: true,
            message: `Max parallel wiki generations reached: ${activeWorkers}/${maxWorkers} slots taken. Please wait for a running generation to finish and try again.`,
          };
        }

        // Attempt to extract the human-readable message from Syngen-style payload
        const rawResult = contentObj?.result;
        if (typeof rawResult === 'string' && rawResult.trim().startsWith('[')) {
          try {
            const objs = JSON.parse(rawResult);
            const firstMsg = Array.isArray(objs)
              ? objs.find(o => o?.object_type === 'message' && typeof o?.data === 'string')
              : null;
            if (firstMsg?.data) {
              return { isError: true, message: firstMsg.data };
            }
          } catch (_e) {
            // fall through
          }
        }

        const fallback =
          contentObj?.error ||
          contentObj?.message ||
          contentStr ||
          'Wiki generation failed';

        return { isError: true, message: fallback };
      } catch (_e) {
        return { isError: false, message: null };
      }
    };
    
    // Only process messages for our current generation
    if (currentStreamIdRef.current && stream_id !== currentStreamIdRef.current) {
      return;
    }
    
    // Track last socket event time for reconnect timeout detection
    lastSocketEventTimeRef.current = Date.now();
    
    // Clear reconnect timeout if we received a real event (not during initial reconnect)
    if (isReconnectingRef.current && reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
      isReconnectingRef.current = false;
    }
    
    
    switch (socketMessageType) {
      case SocketMessageType.StartTask: {
        // Task started - store task_id if provided
        const taskId = content?.task_id;
        if (taskId) {
          currentTaskIdRef.current = taskId;
          updateGenerationState(projectId, toolkitId, { taskId });
        }
        setGenerationStatus({ status: 'running', message: 'Wiki generation started...' });
        break;
      }
      
      case SocketMessageType.AgentThinkingStep:
      case SocketMessageType.AgentThinkingStepUpdate: {
        // Thinking step received - add to list
        const stepMessage = response_metadata?.message || content?.message || 'Processing...';
        const stepId = `step-${++stepIdCounterRef.current}`;
        
        
        setThinkingSteps((prev) => [...prev, {
          id: stepId,
          message: stepMessage,
          timestamp: Date.now(),
          type: socketMessageType,
          metadata: response_metadata,
        }]);
        
        // Update status with thinking step message
        setGenerationStatus({ status: 'running', message: stepMessage });
        break;
      }
      
      case SocketMessageType.AgentToolStart: {
        const toolName = response_metadata?.tool_name;
        setGenerationStatus({ status: 'running', message: `Running ${toolName || 'tool'}...` });
        break;
      }
      
      case SocketMessageType.AgentToolEnd: {
        // Tool completed - could be final or intermediate, OR an error
        
        // Check if this tool end contains an error (e.g., slots full)
        const toolEndParsed = parseAgentResponseForError(content, response_metadata);
        if (toolEndParsed?.isError) {
          console.error('[DeepWiki] Tool ended with error:', toolEndParsed.message);
          // Mark that this generation has errored - prevents subsequent success from overwriting
          generationErroredRef.current = true;
          setGenerationStatus({
            status: 'error',
            message: toolEndParsed.message || 'Wiki generation failed',
          });
          setThinkingSteps(prev => [
            ...prev,
            {
              id: `step-${++stepIdCounterRef.current}`,
              message: toolEndParsed.message || 'Wiki generation failed',
              timestamp: Date.now(),
              type: 'error',
              metadata: response_metadata,
            },
          ]);
          cleanupGeneration(socketUnsubscribeRef.current);
        }
        break;
      }
      
      case SocketMessageType.AgentResponse: {
        // Generation complete (or error wrapped as agent_response)

        // Skip if we already handled an error for this generation
        if (generationErroredRef.current) {
          break;
        }

        const parsed = parseAgentResponseForError(content, response_metadata);
        if (parsed?.isError) {
          generationErroredRef.current = true;
          setGenerationStatus({
            status: 'error',
            message: parsed.message || 'Wiki generation failed',
          });
          setThinkingSteps(prev => [
            ...prev,
            {
              id: `step-${++stepIdCounterRef.current}`,
              message: parsed.message || 'Wiki generation failed',
              timestamp: Date.now(),
              type: 'error',
              metadata: response_metadata,
            },
          ]);
          cleanupGeneration(socketUnsubscribeRef.current);
          break;
        }

        setGenerationStatus({ status: 'completed', message: 'Wiki generated successfully!' });
        
        // Refresh artifacts list and auto-select the newest manifest
        const nextBucketName = getBucketName(toolkit);
        if (nextBucketName) {
          loadArtifactsList(nextBucketName, true, configuredRepo); // forceSelectLatest=true after generation
        }
        cleanupGeneration(socketUnsubscribeRef.current);
        break;
      }
      
      case SocketMessageType.AgentToolError:
      case SocketMessageType.AgentException:
      case SocketMessageType.Error:
      case SocketMessageType.LlmError: {
        // Error occurred
        const errorMsg = typeof content === 'string' 
          ? content 
          : content?.message || content?.error || 'Wiki generation failed';
        console.error('[DeepWiki] Error:', errorMsg);
        
        setGenerationStatus({
          status: 'error',
          message: errorMsg,
        });
        cleanupGeneration(socketUnsubscribeRef.current);
        break;
      }
      
      case SocketMessageType.Chunk:
      case SocketMessageType.AIMessageChunk:
      case SocketMessageType.AgentLlmChunk: {
        // Streaming chunk - update progress
        if (content && typeof content === 'string') {
        }
        break;
      }
      
      default:
    }
  }, [toolkit, cleanupGeneration]);
  
  // Socket setup for test_toolkit_tool event
  const { subscribe: socketSubscribe, unsubscribe: socketUnsubscribe, emit: socketEmit } = 
    useManualSocket(sioEvents.test_toolkit_tool, handleWikiSocketMessage);
  
  // Store unsubscribe ref for cleanup
  const socketUnsubscribeRef = useRef(socketUnsubscribe);
  useEffect(() => {
    socketUnsubscribeRef.current = socketUnsubscribe;
  }, [socketUnsubscribe]);

  // Restore generation state after page reload or toolkit switch
  // Track which toolkit we've attempted restore for (not just boolean)
  const lastRestoredToolkitRef = useRef(null);
  
  useEffect(() => {
    // Only run when component mounts and toolkit is loaded
    if (!toolkit || !projectId || !toolkitId) return;
    
    // Build unique key for this toolkit
    const toolkitKey = `${projectId}-${toolkitId}`;
    
    // Skip if we've already attempted restore for this exact toolkit
    // or if generation is in progress for this toolkit
    if (lastRestoredToolkitRef.current === toolkitKey) return;
    if (generating && currentStreamIdRef.current) return;
    
    // Mark that we're attempting restore for this toolkit
    lastRestoredToolkitRef.current = toolkitKey;
    
    const savedState = loadGenerationState(projectId, toolkitId);
    if (!savedState) return;
    
    // With toolkit-specific keys, no need to check if state matches - it always does
    // The loadGenerationState already uses the correct toolkit-specific key
    
    
    // Mark that we are reconnecting (used for timeout detection)
    isReconnectingRef.current = true;
    
    // Restore generation state
    currentStreamIdRef.current = savedState.streamId;
    currentMessageIdRef.current = savedState.messageId;
    generationStartTimeRef.current = savedState.startTime;
    currentTaskIdRef.current = savedState.taskId || null;
    
    // Calculate elapsed time since generation started
    const elapsedSeconds = Math.floor((Date.now() - savedState.startTime) / 1000);
    setGenerationElapsedTime(elapsedSeconds);
    
    // Set generating state with reconnection indicator
    setGenerating(true);
    setGenerationStatus({ 
      status: 'running', 
      message: `Wiki generation in progress...`
    });
    
    // Add a thinking step to indicate reconnection
    setThinkingSteps([{
      id: 'reconnect-notice',
      message: `Reconnecting to generation session...`,
      timestamp: Date.now(),
      type: 'info',
      isReconnectNotice: true,
    }]);
    
    // Restart elapsed time counter
    elapsedTimerRef.current = setInterval(() => {
      if (generationStartTimeRef.current) {
        setGenerationElapsedTime(Math.floor((Date.now() - generationStartTimeRef.current) / 1000));
      }
    }, 1000);
    
    // Subscribe to socket events first
    socketSubscribe();
    
    // Emit event to rejoin the socket room for this stream
    // This allows us to receive any remaining events from the ongoing generation
    const rejoined = emitSocketEvent(sioEvents.test_toolkit_enter_room, {
      stream_id: savedState.streamId,
      event_name: 'test_toolkit_tool',
    });
    
    if (rejoined) {
      setThinkingSteps(prev => prev.map(step => 
        step.id === 'reconnect-notice' 
          ? { ...step, message: 'Rejoined generation session. Waiting for updates...' }
          : step
      ));
    } else {
      console.warn('[DeepWiki] Failed to emit test_toolkit_enter_room - socket may not be connected');
    }
    
    // Poll task status to detect completion when socket events are missed
    // This is the most deterministic approach - directly check the task state
    const POLL_INTERVAL_MS = 5000; // Poll every 5 seconds
    const taskId = savedState.taskId;
    const checkBucket = getBucketName(toolkit);
    
    if (taskId) {
      
      const pollTaskStatus = async () => {
        // Stop polling if no longer reconnecting (received socket events or cleaned up)
        if (!isReconnectingRef.current) {
          if (reconnectTimeoutRef.current) {
            clearInterval(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
          return;
        }
        
        try {
          const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
            ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
            : {};
          
          const response = await fetch(
            `/api/v2/elitea_core/application_task/prompt_lib/${projectId}/${taskId}`,
            { headers }
          );
          
          if (response.ok) {
            const data = await response.json();
            const status = data.status;
            
            
            // Handle terminal states - normalize to lowercase for comparison
            const normalizedStatus = typeof status === 'string' ? status.toLowerCase() : '';
            
            if (status === 'SUCCESS' || normalizedStatus === 'success') {
              setGenerationStatus({ status: 'completed', message: 'Wiki generation completed!' });
              setThinkingSteps(prev => [...prev, {
                id: `step-${++stepIdCounterRef.current}`,
                message: 'Generation completed. Refreshing wiki...',
                timestamp: Date.now(),
                type: 'success',
              }]);
              
              // Refresh artifacts and cleanup
              if (checkBucket) {
                const sanitizedBucket = sanitizeBucketName(checkBucket);
                loadArtifactsList(sanitizedBucket, true, configuredRepo); // forceSelectLatest=true
              }
              cleanupGeneration(socketUnsubscribeRef.current);
              return;
            }
            
            // Handle "stopped" status - task finished but was marked as stopped
            // This happens when generation completes normally but status shows as stopped
            if (normalizedStatus === 'stopped' || normalizedStatus === 'complete' || normalizedStatus === 'completed') {
              
              // Check if wiki was actually generated by looking at artifacts
              if (checkBucket) {
                const sanitizedBucket = sanitizeBucketName(checkBucket);
                try {
                  const artifactsUrl = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
                  const artifactsResponse = await fetch(artifactsUrl, { headers });
                  if (artifactsResponse.ok) {
                    const artifactsData = await artifactsResponse.json();
                    const artifacts = artifactsData.rows || [];
                    
                    // Check if any artifacts were modified after generation started
                    const hasNewArtifacts = artifacts.some(a => {
                      if (!a.modified) return false;
                      const modifiedTime = new Date(a.modified).getTime();
                      return modifiedTime > savedState.startTime;
                    });
                    
                    if (hasNewArtifacts) {
                      setGenerationStatus({ status: 'completed', message: 'Wiki generation completed!' });
                      setThinkingSteps(prev => [...prev, {
                        id: `step-${++stepIdCounterRef.current}`,
                        message: 'Generation completed. Refreshing wiki...',
                        timestamp: Date.now(),
                        type: 'success',
                      }]);
                      loadArtifactsList(sanitizedBucket, true, configuredRepo); // forceSelectLatest=true
                      cleanupGeneration(socketUnsubscribeRef.current);
                      return;
                    }
                  }
                } catch (artifactErr) {
                  console.warn('[DeepWiki] Error checking artifacts:', artifactErr);
                }
              }
              
              // No new artifacts - treat as cancelled
              setGenerationStatus({ 
                status: 'error', 
                message: 'Generation was stopped' 
              });
              cleanupGeneration(socketUnsubscribeRef.current);
              return;
            }
            
            if (status === 'FAILURE' || status === 'REVOKED' || normalizedStatus === 'failure' || normalizedStatus === 'revoked' || normalizedStatus === 'error') {
              setGenerationStatus({ 
                status: 'error', 
                message: status === 'REVOKED' || normalizedStatus === 'revoked' ? 'Generation was cancelled' : 'Generation failed'
              });
              setThinkingSteps(prev => [...prev, {
                id: `step-${++stepIdCounterRef.current}`,
                message: status === 'REVOKED' || normalizedStatus === 'revoked' ? 'Generation was cancelled.' : 'Generation failed.',
                timestamp: Date.now(),
                type: 'error',
              }]);
              cleanupGeneration(socketUnsubscribeRef.current);
              return;
            }
            
            // PENDING or STARTED - task still running, continue polling
            // Update the thinking step to show we're actively monitoring
            setThinkingSteps(prev => prev.map(step => 
              step.id === 'reconnect-notice' 
                ? { ...step, message: `Monitoring generation... (task: ${status})` }
                : step
            ));
            
          } else if (response.status === 404) {
            // Task not found - might have expired or been cleaned up
            
            // Check artifacts as fallback
            if (checkBucket) {
              const sanitizedBucket = sanitizeBucketName(checkBucket);
              try {
                const artifactsUrl = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
                const artifactsResponse = await fetch(artifactsUrl, { headers });
                if (artifactsResponse.ok) {
                  const artifactsData = await artifactsResponse.json();
                  const artifacts = artifactsData.rows || [];
                  
                  // Check if any artifacts were modified after generation started
                  const hasNewArtifacts = artifacts.some(a => {
                    if (!a.modified) return false;
                    const modifiedTime = new Date(a.modified).getTime();
                    return modifiedTime > savedState.startTime;
                  });
                  
                  if (hasNewArtifacts) {
                    setGenerationStatus({ status: 'completed', message: 'Wiki generation completed!' });
                    loadArtifactsList(sanitizedBucket, false, configuredRepo);
                    cleanupGeneration(socketUnsubscribeRef.current);
                    return;
                  }
                }
              } catch (artifactErr) {
                console.warn('[DeepWiki] Error checking artifacts:', artifactErr);
              }
            }
            
            // No artifacts found - generation probably failed
            setGenerationStatus({ 
              status: 'error', 
              message: 'Generation session expired. Please try again.' 
            });
            cleanupGeneration(socketUnsubscribeRef.current);
          }
        } catch (err) {
          console.warn('[DeepWiki] Error polling task status:', err);
          // Continue polling on error - might be temporary network issue
        }
      };
      
      // Start polling immediately and then at intervals
      pollTaskStatus();
      reconnectTimeoutRef.current = setInterval(pollTaskStatus, POLL_INTERVAL_MS);
      
    } else {
      // No task ID available - fall back to artifact checking after timeout
      
      const RECONNECT_TIMEOUT_MS = 30000;
      reconnectTimeoutRef.current = setTimeout(async () => {
        if (!isReconnectingRef.current) return;
        
        
        if (checkBucket) {
          const sanitizedBucket = sanitizeBucketName(checkBucket);
          try {
            const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
              ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
              : {};
            const url = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
            const response = await fetch(url, { headers });
            
            if (response.ok) {
              const data = await response.json();
              const artifacts = data.rows || [];
              const hasNewArtifacts = artifacts.some(a => {
                if (!a.modified) return false;
                return new Date(a.modified).getTime() > savedState.startTime;
              });
              
              if (hasNewArtifacts) {
                setGenerationStatus({ status: 'completed', message: 'Wiki generation completed!' });
                loadArtifactsList(sanitizedBucket, false, configuredRepo);
                cleanupGeneration(socketUnsubscribeRef.current);
                return;
              }
            }
          } catch (err) {
            console.warn('[DeepWiki] Error checking artifacts:', err);
          }
        }
        
        setGenerationStatus({ 
          status: 'error', 
          message: 'Lost connection to generation session. Please check the wiki or re-generate.' 
        });
        cleanupGeneration(socketUnsubscribeRef.current);
        
      }, RECONNECT_TIMEOUT_MS);
    }
    
  }, [toolkit, projectId, toolkitId, socketSubscribe, generating, isDevelopment, cleanupGeneration]);

  // Generate wiki using Socket.IO
  const handleGenerateWiki = useCallback(() => {
    if (!toolkit || !projectId || !toolkitId) return;
    
    // Pre-check slot availability before starting
    if (slots && !slots.canStart) {
      setGenerationStatus({ 
        status: 'error', 
        message: `All ${slots.total} generation slots are in use. Please wait for a running generation to finish.` 
      });
      return;
    }
    
    setUpdateDialogOpen(false);
    setGenerating(true);
    setGenerationStatus({ status: 'running', message: 'Starting wiki generation...' });
    setThinkingSteps([]);
    setGenerationElapsedTime(0);
    stepIdCounterRef.current = 0;
    currentTaskIdRef.current = null;
    // Reset error flag for new generation
    generationErroredRef.current = false;
    
    // Refresh slots after starting (will show updated count)
    refreshSlots();
    
    // Start elapsed time counter
    const startTime = Date.now();
    generationStartTimeRef.current = startTime;
    elapsedTimerRef.current = setInterval(() => {
      if (generationStartTimeRef.current) {
        setGenerationElapsedTime(Math.floor((Date.now() - generationStartTimeRef.current) / 1000));
      }
    }, 1000);

    try {
      // IMPORTANT: use the toolkit's original stored settings
      const originalSettings = toolkit.settings || {};

      // LLM config must come from toolkit settings
      const llmModel = originalSettings.toolkit_configuration_llm_model;
      const maxTokens = originalSettings.toolkit_configuration_max_tokens;

      if (!llmModel) {
        throw new Error('Toolkit settings missing `llm_model`. Set it in toolkit settings first.');
      }
      if (!maxTokens) {
        throw new Error('Toolkit settings missing `max_tokens`. Set it in toolkit settings first.');
      }

      const toolkitName = toolkit.toolkit_name || toolkit.name || String(toolkitId);
      
      // Generate unique IDs for tracking
      const streamId = uuidv4();
      const messageId = uuidv4();
      currentStreamIdRef.current = streamId;
      currentMessageIdRef.current = messageId;
      
      // Persist generation state for recovery after page reload
      saveGenerationState({
        streamId,
        messageId,
        startTime,
        projectId,
        toolkitId,
        toolkitName,
        taskId: null,
      });
      
      
      // Subscribe to socket events before emitting
      socketSubscribe();
      
      // Enter the socket room for this stream to isolate events from other tabs
      // This is CRITICAL for multi-tab support - without it, all tabs receive all events
      const roomEntered = emitSocketEvent(sioEvents.test_toolkit_enter_room, {
        stream_id: streamId,
        event_name: 'test_toolkit_tool',
      });
      if (roomEntered) {
      } else {
        console.warn('[DeepWiki] Failed to enter socket room - events may leak between tabs');
      }
      
      // Build socket payload (matching EliteAUI pattern)
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
        tool_name: 'generate_wiki',
        tool_params: {
          query: 'GO',
          // Forward the user-selected planner choice + (cluster-only) test
          // exclusion. The backend bridges these into the subprocess worker
          // env / RunnableConfig.
          planner_type: plannerMode,
          exclude_tests: plannerMode === 'cluster' ? excludeTests : null,
        },
        llm_model: llmModel,
        llm_settings: {
          max_tokens: maxTokens,
          model_name: llmModel,
        },
      };
      
      // Emit the socket event
      const emitResult = socketEmit(payload);
      
      if (!emitResult) {
        throw new Error('Failed to emit socket event - socket may not be connected');
      }
      
      // Note: No timeout - generation can take hours for large repos
      // Session will end naturally on AgentResponse, AgentToolError, or AgentException
      
    } catch (err) {
      console.error('[DeepWiki] Error starting wiki generation:', err);
      setGenerationStatus({
        status: 'error',
        message: err?.message || 'Failed to start wiki generation',
      });
      cleanupGeneration(socketUnsubscribeRef.current);
    }
  }, [toolkit, projectId, toolkitId, socketSubscribe, socketEmit, cleanupGeneration, slots, refreshSlots, plannerMode, excludeTests]);

  // Create theme with EliteA design tokens
  const theme = useMemo(
    () => {
      const colors = mode === 'dark' ? designTokens.colors.dark : designTokens.colors.light;
      return createTheme({
        typography: {
          fontFamily: designTokens.typography.fontFamily,
          fontFeatureSettings: '"clig" 0, "liga" 0',
        },
        palette: {
          mode,
          primary: {
            main: colors.primary,
          },
          background: {
            default: colors.background.default,
            paper: colors.background.secondary,
          },
          text: {
            primary: colors.text.primary,
            secondary: colors.text.secondary,
          },
          divider: colors.border,
          action: {
            hover: colors.background.hover,
            selected: colors.background.selected,
          },
        },
        components: {
          MuiButton: {
            styleOverrides: {
              root: {
                textTransform: 'none',
                fontFamily: designTokens.typography.fontFamily,
                fontWeight: 500,
                borderRadius: '28px',
                fontSize: '12px',
                lineHeight: '16px',
                padding: '6px 16px',
              },
            },
          },
          MuiCssBaseline: {
            styleOverrides: {
              '*': {
                scrollbarWidth: 'thin',
                scrollbarColor: `${colors.border} transparent`,
              },
              body: {
                fontFamily: designTokens.typography.fontFamily,
              },
            },
          },
        },
      });
    },
    [mode]
  );



  // Toggle section expansion
  const toggleSection = (sectionId) => {
    const isCurrentlyExpanded = expandedSections[sectionId];
    
    // Toggle expansion
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
    
    // If we're expanding (not collapsing), load the first document
    if (!isCurrentlyExpanded) {
      const section = wikiStructure?.sections?.find(s => s.id === sectionId);
      if (section?.subsections?.[0]) {
        if (bucketName) {
          loadPageFromArtifact(bucketName, section.subsections[0]);
        }
      }
    }
  };

  // Render hierarchical navigation
  const renderSidebarNavigation = () => {
    if ((!wikiStructure?.sections || wikiStructure.sections.length === 0) && Array.isArray(artifactsList) && artifactsList.length > 0) {
      return artifactsList.map((page) => {
        return (
          <Box key={page.id} sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => (bucketName ? loadPageFromArtifact(bucketName, page) : null)}
              selected={currentPage?.id === page.id}
              sx={{
                py: 1,
                px: 2.5,
                borderRadius: '8px',
                mx: 1,
                '&:hover': {
                  bgcolor: 'action.hover',
                },
                '&.Mui-selected': {
                  bgcolor: 'action.selected',
                  borderLeft: '3px solid',
                  borderColor: 'primary.main',
                  '&:hover': { bgcolor: 'action.selected' },
                },
              }}
            >
              <DescriptionIcon sx={{ fontSize: 18, mr: 1.5, color: 'text.secondary' }} />
              <Typography
                variant="body2"
                sx={{
                  ...designTokens.typography.labelMedium,
                  color: currentPage?.id === page.id ? 'text.primary' : 'text.secondary',
                }}
              >
                {page.title}
              </Typography>
            </ListItemButton>
          </Box>
        );
      });
    }

    if (!wikiStructure?.sections || wikiStructure.sections.length === 0) {
      return (
        <Box sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary">
            No wiki generated yet. Click "Generate Wiki" to create documentation.
          </Typography>
        </Box>
      );
    }

    return wikiStructure.sections.map((section) => {
      // Handle standalone files (no subsections)
      if (section.isStandalone) {
        return (
          <Box key={section.id} sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => (bucketName ? loadPageFromArtifact(bucketName, section) : null)}
              selected={currentPage?.id === section.id}
              sx={{
                py: 1,
                px: 2.5,
                borderRadius: '8px',
                mx: 1,
                '&:hover': { 
                  bgcolor: 'action.hover',
                },
                '&.Mui-selected': {
                  bgcolor: 'action.selected',
                  borderLeft: '3px solid',
                  borderColor: 'primary.main',
                  '&:hover': { bgcolor: 'action.selected' }
                }
              }}
            >
              <DescriptionIcon sx={{ fontSize: 18, mr: 1.5, color: 'text.secondary' }} />
              <Typography variant="body2" sx={{ ...designTokens.typography.labelMedium, color: currentPage?.id === section.id ? 'text.primary' : 'text.secondary' }}>
                {section.title}
              </Typography>
            </ListItemButton>
          </Box>
        );
      }

      // Handle sections with subsections
      return (
        <Box key={section.id} sx={{ mb: 0.5 }}>
          <ListItemButton
            onClick={() => toggleSection(section.id)}
            sx={{
              py: 1,
              px: 2.5,
              borderRadius: '8px',
              mx: 1,
              '&:hover': { bgcolor: 'action.hover' }
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
              {expandedSections[section.id] ? (
                <ExpandMoreIcon sx={{ fontSize: 18, mr: 1.5, color: 'text.secondary' }} />
              ) : (
                <ChevronRightIcon sx={{ fontSize: 18, mr: 1.5, color: 'text.secondary' }} />
              )}
              <Typography variant="body2" sx={{ ...designTokens.typography.labelMedium, color: 'text.primary' }}>
                {section.title}
              </Typography>
            </Box>
          </ListItemButton>
          
          <Collapse in={expandedSections[section.id]} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {section.subsections?.map((subsection, idx) => (
              <ListItemButton
                key={subsection.id || idx}
                selected={currentPage?.id === subsection.id}
                onClick={() => (bucketName ? loadPageFromArtifact(bucketName, subsection) : null)}
                sx={{
                  pl: 5.5,
                  pr: 2.5,
                  py: 0.75,
                  borderRadius: '8px',
                  mx: 1,
                  '&:hover': {
                    bgcolor: 'action.hover'
                  },
                  '&.Mui-selected': {
                    bgcolor: 'action.selected',
                    borderLeft: '3px solid',
                    borderColor: 'primary.main',
                    '&:hover': {
                      bgcolor: 'action.selected'
                    }
                  }
                }}
              >
                <Typography variant="body2" sx={{ ...designTokens.typography.bodyMedium, color: currentPage?.id === subsection.id ? 'text.primary' : 'text.secondary' }}>
                  {subsection.title}
                </Typography>
              </ListItemButton>
            ))}
          </List>
        </Collapse>
        </Box>
      );
    });
  };

  // Render table of contents from headings
  const renderTableOfContents = () => {
    if (!pageHeadings || pageHeadings.length === 0) return null;

    return (
      <Box
        sx={{
          pl: 3,
          borderLeft: '2px solid',
          borderColor: 'divider'
        }}
      >
        <Typography 
          variant="overline" 
          sx={{ 
            ...designTokens.typography.labelSmall,
            letterSpacing: '0.72px',
            textTransform: 'uppercase',
            color: 'text.secondary', 
            mb: 2, 
            display: 'block',
            fontWeight: 600
          }}
        >
          On this page
        </Typography>
        <List dense sx={{ py: 0 }}>
          {pageHeadings.map((heading, idx) => (
            <ListItem
              key={idx}
              disablePadding
              sx={{ pl: (heading.level - 1) * 2 }}
            >
              <ListItemButton
                component="a"
                href={`#${heading.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById(heading.id);
                  if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }
                }}
                sx={{
                  py: 0.5,
                  px: 1.5,
                  minHeight: 'auto',
                  borderRadius: '6px',
                  transition: 'all 0.2s ease',
                  '&:hover': { 
                    bgcolor: 'action.hover',
                    transform: 'translateX(2px)'
                  }
                }}
              >
                <Typography
                  variant="body2"
                  sx={{
                    ...designTokens.typography.bodySmall,
                    fontSize: heading.level === 1 ? '13px' : '12px',
                    fontWeight: heading.level === 1 ? 500 : 400,
                    lineHeight: heading.level === 1 ? '18px' : '16px',
                    color: heading.level === 1 ? 'text.primary' : 'text.secondary'
                  }}
                >
                  {heading.text}
                </Typography>
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    );
  };

  const handleOpenEditor = useCallback(() => {
    if (!currentPage) return;
    setEditorContent(originalContent || pageContent);
    setSaveFeedback(null);
    setEditorOpen(true);
  }, [currentPage, originalContent, pageContent]);

  const handleCloseEditor = () => {
    setEditorOpen(false);
    setSaveFeedback(null);
  };

  // ============================================
  // Mermaid Diagram Fix Helpers
  // ============================================

  /**
   * Extract all mermaid blocks from markdown content
   * Returns array of { code, startLine, endLine, index }
   */
  const extractMermaidBlocks = useCallback((markdownContent) => {
    const blocks = [];
    const lines = markdownContent.split('\n');
    let inMermaidBlock = false;
    let blockStartLine = 0;
    let blockContent = [];
    let blockIndex = 0;

    lines.forEach((line, lineIndex) => {
      if (line.trim().startsWith('```mermaid')) {
        inMermaidBlock = true;
        blockStartLine = lineIndex + 1; // 1-indexed
        blockContent = [];
      } else if (inMermaidBlock && line.trim() === '```') {
        blocks.push({
          code: blockContent.join('\n'),
          startLine: blockStartLine,
          endLine: lineIndex + 1, // 1-indexed
          index: blockIndex,
        });
        inMermaidBlock = false;
        blockIndex++;
      } else if (inMermaidBlock) {
        blockContent.push(line);
      }
    });

    return blocks;
  }, []);

  /**
   * Replace a mermaid block in markdown by index
   */
  const replaceMermaidBlock = useCallback((markdownContent, blockIndex, newCode) => {
    const blocks = extractMermaidBlocks(markdownContent);
    const block = blocks[blockIndex];
    
    if (!block) {
      console.error(`Mermaid block ${blockIndex} not found`);
      return markdownContent;
    }

    const lines = markdownContent.split('\n');
    // Replace lines from startLine (after ```mermaid) to endLine-1 (before ```)
    const beforeBlock = lines.slice(0, block.startLine); // includes ```mermaid line
    const afterBlock = lines.slice(block.endLine - 1); // includes closing ``` and after
    
    return [...beforeBlock, newCode, ...afterBlock].join('\n');
  }, [extractMermaidBlocks]);

  /**
   * Extract mermaid code from LLM response (handles markdown code blocks)
   */
  const extractMermaidFromResponse = useCallback((response) => {
    // Try to extract from ```mermaid ... ``` block first
    const mermaidBlockMatch = response.match(/```mermaid\n([\s\S]*?)```/);
    if (mermaidBlockMatch) {
      return mermaidBlockMatch[1].trim();
    }
    
    // Try generic code block
    const codeBlockMatch = response.match(/```\n?([\s\S]*?)```/);
    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }
    
    // Return as-is if no code block found
    return response.trim();
  }, []);

  /**
   * Call LLM to fix a mermaid diagram
   */
  const callLLMToFixDiagram = useCallback(async (errorInfo) => {
    const originalSettings = toolkit?.settings || {};
    const llmModel = originalSettings.toolkit_configuration_llm_model;
    
    if (!llmModel) {
      throw new Error('No LLM model configured in toolkit settings');
    }

    // Comprehensive prompt that teaches syntax rules for different diagram types
    const fullPrompt = `You are a diagram syntax fixer. Analyze and fix the diagram below.

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

### G) CLASS ATTRIBUTES

✅ CORRECT FORMAT:
  +type attributeName
  -type attributeName
  #type attributeName
  ~type attributeName

### H) PARTICIPANT SYNTAX FOR sequenceDiagram
Participants must use simple text for display names. Do NOT use brackets or parentheses in participant definitions.

❌ WRONG:
  participant API as API["api/v2/endpoint.py::ClassName.method"]
  participant DB as DB["Database session"]
  participant Utils as Utils["function_name"]
  participant GC as get_configurations()

✅ CORRECT:
  participant API as API: ClassName.method
  participant DB as Database Session
  participant Utils as function_name
  participant GC as get_configurations

### I) RESERVED KEYWORDS AS PARTICIPANT IDs FOR sequenceDiagram
Do NOT use reserved keywords as participant IDs. Mermaid will interpret them as commands, not participant names.

RESERVED KEYWORDS (case-insensitive):
  alt, else, opt, loop, par, and, critical, break, end, rect, note, over, participant, actor

❌ WRONG:
  participant OPT as Options Generator
  participant ALT as Alternative Service
  participant END as End Handler
  participant LOOP as Loop Processor
  participant NOTE as Note Service
  ...
  GC->>OPT: Generate options
  OPT-->>GC: Results

✅ CORRECT (rename to avoid keywords):
  participant OPTGEN as Options Generator
  participant ALTSVC as Alternative Service
  participant ENDH as End Handler
  participant LOOPPROC as Loop Processor
  participant NOTESVC as Note Service
  ...
  GC->>OPTGEN: Generate options
  OPTGEN-->>GC: Results

HOW TO FIX:
1. Check all participant IDs against the reserved keywords list
2. If a participant ID matches a keyword, rename it by:
   - Adding a suffix: OPT → OPTGEN, OPTS, OPTHANDLER
   - Adding a prefix: OPT → MYOPT, SVCOPT
   - Using a different name entirely: OPT → OPTIONS, GENERATOR
3. Update ALL references to the renamed participant in messages

### J) PARTICIPANT ID CONSISTENCY FOR sequenceDiagram
The participant ID used in messages MUST exactly match the ID defined in the participant declaration.

❌ WRONG (ID mismatch):
  participant OPT_ as Options Generator
  ...
  GC->>OPT: Generate options
  OPT-->>GC: Available options

  participant Svc1 as Service One
  ...
  Svc->>DB: Query data

✅ CORRECT (IDs match exactly):
  participant OPTGEN as Options Generator
  ...
  GC->>OPTGEN: Generate options
  OPTGEN-->>GC: Available options

  participant Svc as Service One
  ...
  Svc->>DB: Query data

HOW TO FIX:
1. List all participant IDs from declarations
2. Search all messages for participant IDs
3. If a message uses an ID not in your list:
   - Fix the message to use the correct ID, OR
   - Fix the participant declaration to match
4. Ensure renamed participants are updated everywhere

### K) SPECIAL CHARACTERS IN sequenceDiagram MESSAGES
Avoid these characters in message text: < > { } [ ] "
They break parsing. Simplify or remove them.

❌ WRONG:
  Client->>API: PUT /... (body: {"toolkit_id": <id>})
  API-->>Client: 404 "Version doesn't exist"
  API-->>Client: 400 {"error": str(e)}
  API->>DB: version.meta["attachment_toolkit_id"]=toolkit_id
  C->>API: GET /api/v2/configurations/{project_id}

✅ CORRECT:
  Client->>API: PUT request with toolkit_id
  API-->>Client: 404 Version does not exist
  API-->>Client: 400 error response
  API->>DB: Update version.meta attachment_toolkit_id
  C->>API: GET /api/v2/configurations/project_id

### L) HTML TAGS IN sequenceDiagram MESSAGES
Do NOT use HTML tags like <br/> in messages. Use simple text instead.

❌ WRONG:
  API->>DB: version.meta["id"]=toolkit_id<br/>COMMIT

✅ CORRECT:
  API->>DB: Update version.meta id, COMMIT

### M) ALT/OPT/LOOP BLOCKS IN sequenceDiagram
Every alt, opt, loop, par, critical block MUST have a matching "end".
Nested blocks each need their OWN "end" statement.

❌ WRONG (missing end for outer alt):
  alt condition1
      A->>B: message1
  else condition2
      B->>C: message2
      alt nested_condition1
          C->>D: message3
      else nested_condition2
          D->>E: message4
      end

✅ CORRECT (each alt has matching end):
  alt condition1
      A->>B: message1
  else condition2
      B->>C: message2
      alt nested_condition1
          C->>D: message3
      else nested_condition2
          D->>E: message4
      end
  end

Count your blocks:
- 1 alt/opt/loop/par = 1 end
- 2 nested alt blocks = 2 end statements
- 3 nested blocks = 3 end statements

### N) SEQUENCE DIAGRAM MESSAGE ARROWS

✅ VALID ARROW TYPES:
  A->>B: Solid line with arrowhead (synchronous)
  A-->>B: Dotted line with arrowhead (asynchronous/response)
  A-xB: Solid line with X (lost message)
  A--xB: Dotted line with X
  A-)B: Solid line with open arrow (async)
  A--)B: Dotted line with open arrow (async)

## STEP 3: VALIDATION CHECKLIST
Before returning, verify:

For ALL diagrams:
[ ] All brackets [] {} () are properly paired
[ ] All quotes "" are properly closed
[ ] No HTML tags in text

For sequenceDiagram:
[ ] No participant ID is a reserved keyword (alt, else, opt, loop, par, and, critical, break, end, rect, note, over, participant, actor)
[ ] All participants use simple text (no brackets [] or parentheses () in definitions)
[ ] All participant IDs in messages match their declarations exactly
[ ] All messages avoid special characters < > { } [ ] "
[ ] Every alt/opt/loop/par has a matching "end"
[ ] Nested blocks have correct number of "end" statements

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

    const socketId = getSocketId();
    if (!socketId) {
      throw new Error('Socket not connected. Please refresh the page and try again.');
    }

    const url = `/api/v2/elitea_core/predict_llm/prompt_lib/${projectId}`;
    const streamId = uuidv4();
    const messageId = uuidv4();

    const requestBody = {
      user_input: fullPrompt,
      chat_history: [],
      llm_settings: {
        model_name: llmModel,
        max_tokens: 4096,
        temperature: 1.0,
      },
      sid: socketId,
      await_task_timeout: 0, // Non-blocking streaming mode (avoids user token requirement)
      stream_id: streamId,
      message_id: messageId,
    };

    // Use streaming mode: send request, collect response via socket.io events
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

      // Fire the request (non-blocking: returns task_id immediately)
      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders,
        },
        body: JSON.stringify(requestBody),
      }).then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          clearTimeout(timeoutId);
          cleanup();
          reject(new Error(`LLM request failed: ${text}`));
        }
        // else: 200 with {task_id} — response arrives via socket
      }).catch((err) => {
        clearTimeout(timeoutId);
        cleanup();
        reject(err);
      });
    });

    if (!llmResponse) {
      throw new Error('Empty response from LLM');
    }

    return extractMermaidFromResponse(llmResponse);
  }, [toolkit, projectId, authHeaders, extractMermaidFromResponse]);

  /**
   * Save markdown content to artifact (internal helper)
   */
  const saveMarkdownContent = useCallback(async (content) => {
    if (!currentPage || !bucketName) {
      throw new Error('No page selected or bucket not available');
    }

    const filename = currentPage.filename || `${currentPage.id}.md`;
    const url = `/api/v2/artifacts/artifacts/default/${projectId}/${bucketName}`;

    const formData = new FormData();
    formData.append('file', new Blob([content], { type: 'text/markdown' }), filename);

    const response = await fetch(url, {
      method: 'POST',
      headers: authHeaders,
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'Failed to save changes');
    }

    return true;
  }, [currentPage, bucketName, projectId, authHeaders]);

  /**
   * Handle Quick Fix button click - gets LLM fix and shows diff preview (does NOT auto-save)
   */
  const handleQuickFix = useCallback(async (errorInfo) => {
    setIsFixingDiagram(true);
    setFixingBlockIndex(errorInfo.blockIndex);
    setFixFeedback(null);
    setLastErrorInfo(errorInfo); // Store for retry

    try {
      // Call LLM to fix the diagram
      const fixedCode = await callLLMToFixDiagram(errorInfo);
      
      // Get current content
      const currentContent = originalContent || pageContent;
      
      // Store proposed fix for preview (NOT saved yet)
      setLastFix({
        original: errorInfo.mermaidCode,
        fixed: fixedCode,
        blockIndex: errorInfo.blockIndex,
        fullOriginalContent: currentContent,
        errorInfo, // Keep for retry
        pending: true, // Flag indicating fix is not yet applied
      });

      // Show diff modal for user to review before accepting
      setDiffModalTab(0); // Start on code diff tab
      setShowDiffModal(true);
      
      setFixFeedback({
        type: 'info',
        message: 'Review the proposed fix below',
      });

    } catch (err) {
      // Categorize errors for better user feedback
      let errorMessage = 'Failed to fix diagram';
      const errMsg = err.message?.toLowerCase() || '';
      
      if (errMsg.includes('timeout') || errMsg.includes('timed out')) {
        errorMessage = 'Request timed out. The model may be busy - try again.';
      } else if (errMsg.includes('empty response')) {
        errorMessage = 'Model returned empty response. Try again or use a different model.';
      } else if (errMsg.includes('invalid json')) {
        errorMessage = 'Invalid response from server. Please try again.';
      } else if (errMsg.includes('no llm model')) {
        errorMessage = 'No LLM model configured. Check toolkit settings.';
      } else if (errMsg.includes('401') || errMsg.includes('unauthorized')) {
        errorMessage = 'Authentication error. Please refresh the page.';
      } else if (errMsg.includes('429') || errMsg.includes('rate limit')) {
        errorMessage = 'Rate limit exceeded. Please wait and try again.';
      } else if (errMsg.includes('500') || errMsg.includes('server error')) {
        errorMessage = 'Server error. Please try again later.';
      } else if (err.message) {
        errorMessage = `Fix failed: ${err.message}`;
      }
      
      setFixFeedback({
        type: 'error',
        message: errorMessage,
        canRetry: true,
      });
    } finally {
      setIsFixingDiagram(false);
      setFixingBlockIndex(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    callLLMToFixDiagram, 
    originalContent, 
    pageContent, 
    replaceMermaidBlock, 
    saveMarkdownContent,
  ]);

  /**
   * Handle Navigate to Error button click - open editor at error location
   */
  const handleNavigateToError = useCallback((errorInfo) => {
    const content = originalContent || pageContent;
    const blocks = extractMermaidBlocks(content);
    const block = blocks[errorInfo.blockIndex];
    
    if (block && errorInfo.lineNumber) {
      // Calculate absolute line: block start + error line within block
      const absoluteLine = block.startLine + errorInfo.lineNumber;
      
      // Open editor and set scroll target
      setEditorContent(content);
      setSaveFeedback(null);
      setScrollToLine(absoluteLine);
      setEditorOpen(true);
    } else {
      // Just open the editor at the start
      setEditorContent(content);
      setSaveFeedback(null);
      setEditorOpen(true);
    }
  }, [originalContent, pageContent, extractMermaidBlocks]);

  /**
   * Handle Accept & Save - applies the pending fix and saves to artifact
   */
  const handleAcceptFix = useCallback(async () => {
    if (!lastFix?.pending || !lastFix?.fixed) {
      setFixFeedback({ type: 'error', message: 'No pending fix to accept' });
      return;
    }

    try {
      // Get the original content and apply the fix
      const currentContent = lastFix.fullOriginalContent;
      const updatedContent = replaceMermaidBlock(currentContent, lastFix.blockIndex, lastFix.fixed);
      
      // Save to artifact
      await saveMarkdownContent(updatedContent);
      
      // Update local state to re-render
      const { tocItems, contentWithoutTOC } = extractTOCFromMarkdown(updatedContent);
      setOriginalContent(updatedContent);
      setEditorContent(updatedContent);
      setPageContent(contentWithoutTOC);
      setPageHeadings(tocItems.length > 0 ? tocItems : extractHeadings(contentWithoutTOC));
      
      // Mark fix as applied (no longer pending)
      setLastFix(prev => prev ? { ...prev, pending: false } : null);
      setShowDiffModal(false);
      setFixFeedback({ type: 'success', message: 'Diagram fixed and saved!' });
      
      setTimeout(() => setFixFeedback(null), 3000);
    } catch (err) {
      setFixFeedback({ type: 'error', message: `Failed to save fix: ${err.message}` });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastFix, replaceMermaidBlock, saveMarkdownContent]);

  /**
   * Handle Cancel - discards the pending fix without saving
   */
  const handleCancelFix = useCallback(() => {
    setLastFix(null);
    setLastErrorInfo(null);
    setShowDiffModal(false);
    setFixFeedback(null);
  }, []);

  /**
   * Handle undo of an already-applied fix (reverts to original)
   */
  const handleUndoFix = useCallback(async () => {
    if (!lastFix?.fullOriginalContent || lastFix?.pending) {
      setFixFeedback({ type: 'error', message: 'No applied fix to undo' });
      return;
    }

    try {
      const content = lastFix.fullOriginalContent;
      await saveMarkdownContent(content);
      
      const { tocItems, contentWithoutTOC } = extractTOCFromMarkdown(content);
      setOriginalContent(content);
      setEditorContent(content);
      setPageContent(contentWithoutTOC);
      setPageHeadings(tocItems.length > 0 ? tocItems : extractHeadings(contentWithoutTOC));
      
      setLastFix(null);
      setShowDiffModal(false);
      setFixFeedback({ type: 'success', message: 'Fix reverted' });
      
      setTimeout(() => setFixFeedback(null), 3000);
    } catch (err) {
      setFixFeedback({ type: 'error', message: `Failed to undo: ${err.message}` });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastFix, saveMarkdownContent]);

  /**
   * Handle retry of diagram fix - gets a new fix from LLM
   */
  const handleRetryFix = useCallback(async () => {
    const errorInfo = lastErrorInfo || lastFix?.errorInfo;
    if (!errorInfo) {
      setFixFeedback({ type: 'error', message: 'No diagram info available for retry' });
      return;
    }
    
    // Clear the current pending fix
    setLastFix(null);
    setShowDiffModal(false);
    
    // Retry with the same error info
    setTimeout(() => {
      handleQuickFix(errorInfo);
    }, 100);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastErrorInfo, lastFix, handleQuickFix]);

  const configuredRepo = useMemo(() => getConfiguredRepo(toolkit, settingsData, resolvedRepoName), [toolkit, settingsData, resolvedRepoName]);

  const handleOpenSettings = () => setSettingsOpen(true);
  const handleCloseSettings = () => {
    setSettingsOpen(false);
    setSettingsSaved(false);
    setSettingsError(null);
  };

  const handleSettingsTextChange = event => {
    const nextText = event.target.value;
    setSettingsText(nextText);
    setSettingsSaved(false);
    try {
      const parsed = nextText ? JSON.parse(nextText) : {};
      setSettingsData(parsed);
      setSettingsError(null);
    } catch (err) {
      setSettingsError(`Invalid JSON: ${err.message}`);
    }
  };

  const handleSettingFieldChange = (key, value) => {
    setSettingsData(prev => {
      const updated = { ...(prev || {}), [key]: value };
      setSettingsText(JSON.stringify(updated, null, 2));
      return updated;
    });
    setSettingsSaved(false);
    setSettingsError(null);
  };

  const handleSaveSettings = async () => {
    if (!projectId || !toolkitId) {
      setSettingsError('Missing project or toolkit identifier.');
      return;
    }

    let parsedSettings = {};
    try {
      parsedSettings = settingsText ? JSON.parse(settingsText) : settingsData || {};
    } catch (err) {
      setSettingsError(`Invalid JSON: ${err.message}`);
      return;
    }

    setSavingSettings(true);
    setSettingsError(null);
    setSettingsSaved(false);

    try {
      const url = `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`;
      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders,
        },
        body: JSON.stringify({
          ...(toolkit || {}),
          settings: parsedSettings,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Failed to update settings (${response.status})`);
      }

      const updated = await response.json();
      const normalized = updated?.data || updated;
      setToolkit(normalized);

      const nextSettings = normalized?.settings || parsedSettings;
      setSettingsData(nextSettings || {});
      setSettingsText(JSON.stringify(nextSettings || {}, null, 2));
      setSettingsSaved(true);
    } catch (err) {
      console.error('Error saving settings:', err);
      setSettingsError(err.message || 'Unable to update settings.');
    } finally {
      setSavingSettings(false);
    }
  };

  const handleDeleteWiki = async () => {
    if (!bucketName || !projectId) {
      setSettingsError('Bucket or project is not set.');
      return;
    }

    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    setDeleteDialogOpen(false);
    setDeletingWiki(true);
    setSettingsError(null);
    setSettingsSaved(false);

    const sanitizedBucket = sanitizeBucketName(bucketName);

    try {
      // Step 1: Delete the toolkit (main object)
      if (toolkitId && projectId) {
        try {
          const toolkitUrl = `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`;
          const toolkitResponse = await fetch(toolkitUrl, {
            method: 'DELETE',
            headers: {
              ...authHeaders,
            },
          });

          if (!toolkitResponse.ok) {
            const errorText = await toolkitResponse.text();
            console.warn('Toolkit deletion failed:', toolkitResponse.status, errorText);
            // Continue anyway - we want to clean up artifacts even if toolkit deletion fails
          }
        } catch (toolkitErr) {
          console.warn('Could not delete toolkit, but continuing with artifact cleanup:', toolkitErr);
        }
      }

      // Step 2: Delete wiki artifacts using platform APIs
      // With the new folder-per-wiki structure, we delete only this wiki's artifacts,
      // not the entire bucket (which may contain multiple wikis)
      // Try to get wiki_id from current manifest, or derive from manifest name
      let wikiId = currentManifest?.wiki_id;
      if (!wikiId && currentManifest?.name) {
        // Extract wiki_id from manifest path: {wiki_id}/wiki_manifest_{version}.json
        const pathParts = currentManifest.name.split('/');
        if (pathParts.length > 1) {
          wikiId = pathParts[0];
        }
      }
      
      console.log('Delete wiki - wikiId:', wikiId, 'currentManifest:', currentManifest);
      
      if (wikiId && projectId && sanitizedBucket) {
        try {
          // 2a. List all artifacts in the bucket
          const listUrl = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
          const listResponse = await fetch(listUrl, { headers: authHeaders });
          
          if (listResponse.ok) {
            const listData = await listResponse.json();
            // API returns {rows: files}, extract the rows array
            const allArtifacts = listData?.rows || [];
            
            // 2b. Filter artifacts that belong to this wiki (start with wiki_id/)
            const wikiPrefix = `${wikiId}/`;
            const wikiArtifacts = allArtifacts.filter(a => {
              const name = typeof a === 'string' ? a : a?.name;
              return name && name.startsWith(wikiPrefix);
            });
            
            console.log(`Found ${wikiArtifacts.length} artifacts to delete for wiki ${wikiId}`);
            
            // 2c. Delete each artifact belonging to this wiki
            let deletedCount = 0;
            for (const artifact of wikiArtifacts) {
              const artifactName = typeof artifact === 'string' ? artifact : artifact?.name;
              if (!artifactName) continue;
              
              try {
                // DELETE endpoint expects filename as query parameter, not path segment
                const deleteUrl = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}?filename=${encodeURIComponent(artifactName)}`;
                const deleteResp = await fetch(deleteUrl, {
                  method: 'DELETE',
                  headers: authHeaders,
                });
                if (deleteResp.ok) {
                  deletedCount++;
                } else {
                  console.warn(`Failed to delete artifact ${artifactName}: ${deleteResp.status}`);
                }
              } catch (delErr) {
                console.warn(`Error deleting artifact ${artifactName}:`, delErr);
              }
            }
            
            console.log(`Deleted ${deletedCount}/${wikiArtifacts.length} artifacts for wiki ${wikiId}`);
            
            // 2d. Update the registry to remove this wiki
            try {
              const registryPath = '_registry/wikis.json';
              const registryUrl = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}/${encodeURIComponent(registryPath)}`;
              const registryResp = await fetch(registryUrl, { headers: authHeaders });
              
              if (registryResp.ok) {
                const registryText = await registryResp.text();
                const registry = JSON.parse(registryText);
                
                // Filter out the deleted wiki
                if (Array.isArray(registry.wikis)) {
                  registry.wikis = registry.wikis.filter(w => w.id !== wikiId);
                  
                  // Upload updated registry
                  const formData = new FormData();
                  const registryBlob = new Blob([JSON.stringify(registry, null, 2)], { type: 'application/json' });
                  formData.append('file', registryBlob, registryPath);
                  
                  const uploadUrl = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
                  await fetch(uploadUrl, {
                    method: 'POST',
                    headers: {
                      ...authHeaders,
                      // Don't set Content-Type - let browser set it with boundary for FormData
                    },
                    body: formData,
                  });
                  
                  console.log(`Removed wiki ${wikiId} from registry`);
                }
              }
            } catch (regErr) {
              console.warn('Could not update registry, but wiki artifacts were deleted:', regErr);
            }
            
          } else {
            console.warn(`Failed to list artifacts: ${listResponse.status}`);
          }
        } catch (wikiErr) {
          console.warn('Error during wiki artifact deletion:', wikiErr);
        }
      } else if (sanitizedBucket && projectId) {
        // Fallback for legacy wikis without wiki_id: 
        // Delete all artifacts in the bucket (legacy wikis use one bucket per wiki)
        console.warn('No wiki_id found, falling back to deleting all artifacts in bucket for legacy wiki');
        try {
          // List all artifacts
          const listUrl = `/api/v2/artifacts/artifacts/default/${projectId}/${sanitizedBucket}`;
          const listResponse = await fetch(listUrl, { headers: authHeaders });
          
          if (listResponse.ok) {
            const listData = await listResponse.json();
            // API returns {rows: files}, extract the rows array
            const allArtifacts = listData?.rows || [];
            console.log(`Found ${allArtifacts.length} artifacts to delete in legacy bucket ${sanitizedBucket}`);
            
            // Delete each artifact
            let deletedCount = 0;
            for (const artifact of allArtifacts) {
              const artifactName = typeof artifact === 'string' ? artifact : artifact?.name;
              if (!artifactName) continue;
              
              try {
                // DELETE endpoint expects filename as query parameter, not path segment
                const deleteUrl = `/api/v2/artifacts/artifact/default/${projectId}/${sanitizedBucket}?filename=${encodeURIComponent(artifactName)}`;
                const deleteResp = await fetch(deleteUrl, {
                  method: 'DELETE',
                  headers: authHeaders,
                });
                if (deleteResp.ok) {
                  deletedCount++;
                }
              } catch (delErr) {
                console.warn(`Error deleting artifact ${artifactName}:`, delErr);
              }
            }
            console.log(`Deleted ${deletedCount} legacy artifacts`);
          }
        } catch (bucketErr) {
          console.warn('Could not delete legacy artifacts:', bucketErr);
        }
      }

      // Clear local state
      setWikiStructure(null);
      setArtifactsList(null);
      setCurrentPage(null);
      setPageContent('');
      setPageHeadings([]);
      setEditorOpen(false);
      setSettingsSaved(true);
      setWikiManifests([]);
      setSelectedWikiManifestName('');
      
      // Wiki deleted - stay on page to allow generating a new wiki
      // No redirect - user can now generate a fresh wiki for this toolkit
    } catch (err) {
      console.error('Error deleting wiki:', err);
      setSettingsError(err.message || 'Unable to delete wiki.');
    } finally {
      setDeletingWiki(false);
    }
  };

  const handleSaveEdits = async () => {
    if (!currentPage) {
      setSaveFeedback({ type: 'error', message: 'No page selected to edit.' });
      return;
    }

    if (!bucketName) {
      setSaveFeedback({ type: 'error', message: 'Bucket is not available for this toolkit.' });
      return;
    }

    setSavingEdit(true);
    setSaveFeedback(null);

    const filename = currentPage.filename || `${currentPage.id}.md`;
    const url = `/api/v2/artifacts/artifacts/default/${projectId}/${bucketName}`;

    const headers = isDevelopment && import.meta.env.VITE_DEV_TOKEN
      ? { 'Authorization': `Bearer ${import.meta.env.VITE_DEV_TOKEN}` }
      : {};

    const formData = new FormData();
    formData.append('file', new Blob([editorContent], { type: 'text/markdown' }), filename);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to save changes');
      }

      const { tocItems, contentWithoutTOC } = extractTOCFromMarkdown(editorContent);
      setOriginalContent(editorContent);
      setPageContent(contentWithoutTOC);
      setPageHeadings(tocItems.length > 0 ? tocItems : extractHeadings(contentWithoutTOC));
      setSaveFeedback({ type: 'success', message: 'Saved changes.' });
      setEditorOpen(false);
    } catch (err) {
      console.error('Error saving edits:', err);
      setSaveFeedback({ type: 'error', message: err.message || 'Unable to save changes.' });
    } finally {
      setSavingEdit(false);
    }
  };

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box display="flex" alignItems="center" justifyContent="center" minHeight="100vh">
          <CircularProgress />
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Left Sidebar - Navigation */}
        <Box
          sx={{
            width: SIDEBAR_WIDTH,
            flexShrink: 0,
            borderRight: '1px solid',
            borderColor: 'divider',
            height: '100vh',
            position: 'sticky',
            top: 0,
            overflow: 'auto',
            bgcolor: 'background.paper'
          }}
        >
          {/* Header */}
          <Box sx={{ p: 2.5, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Typography 
              variant="h6" 
              sx={{ 
                ...designTokens.typography.headingMedium,
                color: 'text.primary',
                mb: 0.5 
              }}
            >
              {toolkit?.name || 'DeepWiki'}
            </Typography>
            <Typography 
              variant="caption" 
              sx={{
                ...designTokens.typography.bodySmall,
                color: 'text.secondary'
              }}
            >
              Last indexed: {new Date().toLocaleDateString()}
            </Typography>
            {/* Cluster-wide slot availability indicator */}
            <SlotsIndicator 
              slots={slots} 
              loading={slotsLoading} 
              error={slotsError}
              compact
            />
          </Box>

          {/* Actions */}
          <Box sx={{ p: 2.5, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Stack spacing={1.5}>
              <Button
                fullWidth
                variant="contained"
                size="small"
                startIcon={generating ? <CircularProgress size={16} /> : <GenerateIcon />}
                onClick={handleUpdateWiki}
                disabled={generating || !toolkit || !slotsAvailable}
                sx={{
                  ...designTokens.typography.labelMedium,
                  borderRadius: '28px',
                  py: 1,
                }}
              >
                {generating ? 'Generating...' : !slotsAvailable ? 'All Slots Busy' : 'Generate Wiki'}
              </Button>
              {generating && (
                <Button
                  fullWidth
                  variant="outlined"
                  size="small"
                  color="error"
                  startIcon={stoppingGeneration ? <CircularProgress size={16} color="error" /> : <StopIcon />}
                  onClick={handleStopGeneration}
                  disabled={stoppingGeneration}
                  sx={{
                    ...designTokens.typography.labelMedium,
                    borderRadius: '28px',
                    py: 1,
                  }}
                >
                  {stoppingGeneration ? 'Stopping...' : 'Stop Generation'}
                </Button>
              )}
              <Button
                fullWidth
                variant="outlined"
                size="small"
                startIcon={<SettingsIcon />}
                onClick={handleOpenSettings}
                sx={{
                  ...designTokens.typography.labelMedium,
                  borderRadius: '28px',
                  py: 1,
                  borderColor: 'divider',
                }}
              >
                Settings
              </Button>

              {!!bucketName && (
                <Box>
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Wiki version"
                    value={manifestSelectValue}
                    onChange={!isWikiVersionSelectorLocked && wikiManifests.length > 0 ? handleWikiVersionSelect : undefined}
                    disabled={isWikiVersionSelectorLocked || wikiManifests.length === 0}
                  >
                    {wikiManifests.length > 0 ? (
                      wikiManifests.map(m => {
                        const when = m.created_at ? m.created_at.toLocaleString() : 'Unknown time';
                        const commit = m.commit_hash ? ` · ${m.commit_hash.slice(0, 8)}` : '';
                        return (
                          <MenuItem key={m.name} value={m.name}>
                            {when}{commit}
                          </MenuItem>
                        );
                      })
                    ) : (
                      <MenuItem value="legacy_latest">{legacyVersionLabel}</MenuItem>
                    )}
                  </TextField>

                  {isWikiVersionSelectorLocked && (
                    <Typography
                      variant="caption"
                      sx={{
                        ...designTokens.typography.bodySmall,
                        color: 'text.secondary',
                        display: 'block',
                        mt: 0.5,
                      }}
                    >
                      Generating… version selection is locked.
                    </Typography>
                  )}

                  {wikiVersionMode === 'legacy' && (
                    <Typography
                      variant="caption"
                      sx={{
                        ...designTokens.typography.bodySmall,
                        color: 'text.secondary',
                        display: 'block',
                        mt: 0.5,
                      }}
                    >
                      No manifests found; using legacy grouping.
                    </Typography>
                  )}
                </Box>
              )}
            </Stack>
            {generating && <LinearProgress sx={{ mt: 1.5 }} />}
          </Box>

          {/* Navigation Tree */}
          <List sx={{ py: 1 }}>
            {renderSidebarNavigation()}
          </List>
        </Box>

        {/* Main Content Area */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            minWidth: 0
          }}
        >
          {/* Status Messages */}
          {(error || generationStatus) && (
            <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
              {error && (
                <Alert severity="error" sx={{ mb: 1 }}>
                  {error}
                </Alert>
              )}
              {generationStatus?.status === 'error' && (
                <Alert severity="error" onClose={() => setGenerationStatus(null)}>
                  {generationStatus.message}
                </Alert>
              )}
              {generationStatus?.status === 'completed' && (
                <Alert severity="success" onClose={() => setGenerationStatus(null)}>
                  {generationStatus.message}
                </Alert>
              )}
            </Box>
          )}

          {/* Thinking Steps Panel - shown during generation */}
          {generating && (
            <ThinkingStepsPanel
              steps={thinkingSteps}
              status={generationStatus?.status}
              statusMessage={generationStatus?.message}
              elapsedTime={generationElapsedTime}
              mode={mode}
            />
          )}

          {/* Content Container */}
          <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'visible' }}>
            {/* Article Content */}
            <Box
              sx={{
                flexGrow: 1,
                overflow: 'visible',
                px: { xs: 2, sm: 4, md: 6 },
                py: 4,
                maxWidth: pageHeadings.length > 0 ? '900px' : '1200px',
                mx: pageHeadings.length > 0 ? 0 : 'auto'
              }}
            >
              {pageLoading ? (
                <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
                  <CircularProgress />
                </Box>
              ) : currentPage ? (
                <WikiPage
                  currentPage={currentPage}
                  pageContent={pageContent}
                  mode={mode}
                  isFixingDiagram={isFixingDiagram}
                  fixingBlockIndex={fixingBlockIndex}
                  onQuickFix={handleQuickFix}
                  onNavigateToError={handleNavigateToError}
                />
              ) : (
                <Box textAlign="center" py={8}>
                  <Typography variant="h4" color="text.secondary" gutterBottom sx={{ fontWeight: 500 }}>
                    Welcome to DeepWiki
                  </Typography>
                  <Typography variant="body1" color="text.secondary" paragraph>
                    {configuredRepo ? `Repository: ${configuredRepo}` : 'No repository configured'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {wikiStructure
                      ? 'Select a page from the sidebar to view documentation'
                      : 'Click "Generate Wiki" to create documentation from your repository'}
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Right Sidebar - Table of Contents */}
            {pageHeadings.length > 0 && (
              <Box
                sx={{
                  width: 240,
                  flexShrink: 0,
                  display: { xs: 'none', lg: 'block' },
                  position: 'sticky',
                  top: 48,
                  alignSelf: 'flex-start',
                  maxHeight: 'calc(100vh - 100px)',
                  overflow: 'auto',
                  pt: 0,
                  pb: 3,
                  pr: 2
                }}
              >
                {!editorOpen && currentPage && (
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'center',
                      mb: 2,
                      position: 'sticky',
                      top: 0,
                      zIndex: 1,
                      backgroundColor: 'background.default',
                      pt: 0
                    }}
                  >
                    <Tooltip title="Edit page">
                      <Fab size="small" color="primary" onClick={handleOpenEditor}>
                        <EditIcon />
                      </Fab>
                    </Tooltip>
                  </Box>
                )}
                {renderTableOfContents()}
              </Box>
            )}
          </Box>

          {/* Dev Info Footer */}
          {isDevelopment && (
            <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
              <Typography variant="caption" color="text.secondary">
                <strong>Dev Mode:</strong> Project: {projectId} | Toolkit: {toolkitId}
                {getBucketName(toolkit) && ` | Bucket: ${sanitizeBucketName(getBucketName(toolkit))}`}
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      <Drawer
        anchor="right"
        open={editorOpen}
        onClose={handleCloseEditor}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: '80vw', md: '80vw' },
            bgcolor: 'background.default'
          }
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 3,
              py: 2,
              borderBottom: '1px solid',
              borderColor: 'divider'
            }}
          >
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Edit Markdown
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {currentPage?.title || 'Current page'}
              </Typography>
            </Box>
            <Stack direction="row" spacing={1}>
              <Tooltip title="Close">
                <IconButton onClick={handleCloseEditor}>
                  <CloseIcon />
                </IconButton>
              </Tooltip>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveEdits}
                disabled={savingEdit}
              >
                {savingEdit ? 'Saving...' : 'Save'}
              </Button>
            </Stack>
          </Box>

          {saveFeedback && (
            <Alert
              severity={saveFeedback.type}
              sx={{ mx: 3, mt: 2 }}
              onClose={() => setSaveFeedback(null)}
            >
              {saveFeedback.message}
            </Alert>
          )}

          <Box sx={{ flex: 1, overflow: 'auto', px: 2, py: 2 }}>
            <CodeMirror
              value={editorContent}
              height="calc(100vh - 220px)"
              extensions={[markdownLang(), EditorView.lineWrapping]}
              theme={mode === 'dark' ? oneDark : undefined}
              onChange={(value) => setEditorContent(value)}
              onCreateEditor={(view) => {
                editorRef.current = view;
                // Scroll to target line if set
                if (scrollToLine && view) {
                  setTimeout(() => {
                    const line = view.state.doc.line(Math.min(scrollToLine, view.state.doc.lines));
                    view.dispatch({
                      selection: { anchor: line.from },
                      scrollIntoView: true,
                      effects: EditorView.scrollIntoView(line.from, { y: 'center' }),
                    });
                    setScrollToLine(null);
                  }, 100);
                }
              }}
              basicSetup={{
                lineNumbers: true,
                foldGutter: true,
                highlightActiveLine: true,
                indentOnInput: true,
                autocompletion: true,
              }}
              style={{
                fontSize: '14px',
                fontFamily: '"Fira Code", "Courier New", monospace',
                borderRadius: 8,
                border: '1px solid',
                borderColor: 'rgba(255,255,255,0.08)',
              }}
            />
          </Box>

          <Divider />
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              px: 3,
              py: 2,
              gap: 2
            }}
          >
            <Typography variant="caption" color="text.secondary">
              Saving overwrites the artifact in bucket {bucketName || '-'}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button onClick={handleCloseEditor} startIcon={<CloseIcon />} disabled={savingEdit}>
                Cancel
              </Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveEdits}
                disabled={savingEdit}
              >
                {savingEdit ? 'Saving...' : 'Save changes'}
              </Button>
            </Stack>
          </Box>
        </Box>
      </Drawer>

      <Drawer
        anchor="right"
        open={settingsOpen}
        onClose={handleCloseSettings}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 420, md: 480 },
            bgcolor: 'background.default'
          }
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 3,
              py: 2,
              borderBottom: '1px solid',
              borderColor: 'divider'
            }}
          >
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Toolkit Settings
              </Typography>
            </Box>
            <Tooltip title="Close">
              <IconButton onClick={handleCloseSettings}>
                <CloseIcon />
              </IconButton>
            </Tooltip>
          </Box>

          <Box
            sx={{
              px: 3,
              py: 2,
              flex: 1,
              overflow: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
            }}
          >
            {/* Info Section with separators */}
            <Box sx={{ display: 'grid', rowGap: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Project ID
              </Typography>
              <Typography variant="body2" color="text.primary">
                {projectId || '—'}
              </Typography>
              <Divider sx={{ my: 1 }} />
              
              <Typography variant="caption" color="text.secondary">
                Toolkit ID
              </Typography>
              <Typography variant="body2" color="text.primary">
                {toolkitId || '—'}
              </Typography>
              <Divider sx={{ my: 1 }} />
              
              <Typography variant="caption" color="text.secondary">
                Toolkit Name
              </Typography>
              <Typography variant="body2" color="text.primary" sx={{ fontWeight: 500 }}>
                {toolkit?.name || toolkit?.toolkit_name || '—'}
              </Typography>
              <Divider sx={{ my: 1 }} />
              
              <Typography variant="caption" color="text.secondary">
                Repository
              </Typography>
              <Typography variant="body2" color="text.primary">
                {configuredRepo || '—'}
              </Typography>
              <Divider sx={{ my: 1 }} />
              
              <Typography variant="caption" color="text.secondary">
                Bucket
              </Typography>
              <Typography variant="body2" color="text.primary">
                {bucketName || '—'}
              </Typography>
            </Box>

            {!!Object.keys(settingsData || {}).length && (
              <Box sx={{ display: 'grid', rowGap: 1 }}>
                <Typography variant="subtitle2" sx={{ mt: 1 }}>
                  Configuration
                </Typography>
                  {Object.entries(settingsData || {})
                    .filter(([key]) => !['class', 'module', 'toolkit', 'provider'].includes(key))
                    .map(([key, value]) => {
                      const valueType = typeof value;
                      const isReadOnly = READ_ONLY_FIELDS.includes(key);
                      const formattedLabel = formatFieldLabel(key);

                      if (valueType === 'boolean') {
                        return (
                          <FormControlLabel
                            key={key}
                            control={
                              <Switch
                                checked={!!value}
                                onChange={e => handleSettingFieldChange(key, e.target.checked)}
                                disabled={isReadOnly}
                              />
                            }
                            label={formattedLabel}
                          />
                        );
                      }

                      if (valueType === 'number') {
                        return (
                          <TextField
                            key={key}
                            label={formattedLabel}
                            type="number"
                            value={value}
                            onChange={e => handleSettingFieldChange(key, e.target.value === '' ? '' : Number(e.target.value))}
                            fullWidth
                            size="small"
                            disabled={isReadOnly}
                            InputProps={{
                              readOnly: isReadOnly,
                            }}
                          />
                        );
                      }

                      if (valueType === 'string') {
                        // Special validation for bucket field
                        const isBucketField = key.toLowerCase().includes('bucket');
                        const bucketError = isBucketField ? validateBucketName(value) : null;
                        
                        return (
                          <TextField
                            key={key}
                            label={formattedLabel}
                            value={value}
                            onChange={e => handleSettingFieldChange(key, e.target.value)}
                            fullWidth
                            size="small"
                            disabled={isReadOnly}
                            error={!!bucketError}
                            helperText={bucketError || (isBucketField ? 'Lowercase letters, numbers, and hyphens only' : '')}
                            InputProps={{
                              readOnly: isReadOnly,
                            }}
                          />
                        );
                      }

                      return null;
                    })}
                </Box>
            )}

            {settingsError && (
              <Alert
                severity="error"
                onClose={() => setSettingsError(null)}
              >
                {settingsError}
              </Alert>
            )}

            {settingsSaved && (
              <Alert severity="success">Settings updated.</Alert>
            )}

            {toolkit?.toolkit_config && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Toolkit Config (read-only)
                </Typography>
                <Paper variant="outlined" sx={{ p: 1.5, maxHeight: 200, overflow: 'auto' }}>
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12 }}>
                    {JSON.stringify(toolkit.toolkit_config, null, 2)}
                  </pre>
                </Paper>
              </Box>
            )}
          </Box>

          <Divider sx={{ mt: 'auto' }} />
          <Box sx={{ px: 3, py: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
            <Button
              color="error"
              startIcon={<DeleteIcon />}
              onClick={handleDeleteWiki}
              disabled={deletingWiki || !bucketName}
            >
              {deletingWiki ? 'Deleting...' : 'Delete Wiki'}
            </Button>
            <Stack direction="row" spacing={1}>
              <Button onClick={handleCloseSettings} startIcon={<CloseIcon />}>Close</Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveSettings}
                disabled={savingSettings}
              >
                {savingSettings ? 'Saving...' : 'Save Settings'}
              </Button>
            </Stack>
          </Box>
        </Box>
      </Drawer>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <DialogTitle id="delete-dialog-title">
          Delete Wiki?
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="delete-dialog-description">
            {currentManifest?.wiki_id 
              ? `This will delete all artifacts for wiki "${currentManifest.wiki_id}" and unregister it. This action cannot be undone.`
              : 'This will delete all wiki artifacts for this toolkit. This action cannot be undone.'
            }
            {' '}Are you sure you want to continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirmDelete} color="error" variant="contained" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Update Confirmation Dialog */}
      <Dialog
        open={updateDialogOpen}
        onClose={() => setUpdateDialogOpen(false)}
        aria-labelledby="update-dialog-title"
        aria-describedby="update-dialog-description"
      >
        <DialogTitle id="update-dialog-title">
          Generate Wiki Documentation?
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="update-dialog-description">
            {isWikiFresh() ? (
              <>
                Your documentation was last updated{' '}
                <strong>
                  {lastModifiedDate && 
                    `${Math.round((new Date() - lastModifiedDate) / (1000 * 60 * 60))} hours ago`
                  }
                </strong>
                . It's still fresh!
                <br /><br />
                Are you sure you want to regenerate it now?
              </>
            ) : (
              'This will regenerate all wiki documentation from your repository. This may take several minutes.'
            )}
          </DialogContentText>
          <Box sx={{ mt: 2.5 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
              Structure planner
            </Typography>
            <ToggleButtonGroup
              exclusive
              value={plannerMode}
              onChange={(_e, next) => { if (next) setPlannerMode(next); }}
              size="small"
              fullWidth
              aria-label="Structure planner mode"
            >
              <ToggleButton value="deepagents" aria-label="Agentic planner">
                Agentic
              </ToggleButton>
              <ToggleButton value="cluster" aria-label="Graph-clustering planner">
                Graph clustering
              </ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
              {plannerMode === 'cluster'
                ? 'Recommended. Builds the outline deterministically from code-graph clusters and scales predictably from a 4-file repo to a large monorepo.'
                : 'For users who prefer agentic loops. The planner agent drafts the outline iteratively, which can over-segment small or doc-light repositories.'}
            </Typography>
            {plannerMode === 'cluster' && (
              <FormControlLabel
                sx={{ mt: 1 }}
                control={
                  <Switch
                    checked={excludeTests}
                    onChange={(e) => setExcludeTests(e.target.checked)}
                    size="small"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2">Smart skip tests</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Excludes test files from the cluster planner to keep the outline focused.
                    </Typography>
                  </Box>
                }
              />
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUpdateDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleGenerateWiki} 
            color="primary" 
            variant="contained" 
            autoFocus={!isWikiFresh()}
          >
            {isWikiFresh() ? 'Update Anyway' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Mermaid Fix Feedback Snackbar */}
      <Snackbar
        open={!!fixFeedback}
        autoHideDuration={fixFeedback?.type === 'error' ? 8000 : null}
        onClose={() => setFixFeedback(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setFixFeedback(null)}
          severity={fixFeedback?.type || 'info'}
          sx={{ width: '100%', alignItems: 'center' }}
          action={
            fixFeedback?.showViewChanges && lastFix ? (
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button 
                  color="inherit" 
                  size="small" 
                  startIcon={<VisibilityIcon />}
                  onClick={() => setShowDiffModal(true)}
                >
                  View Changes
                </Button>
                <Button 
                  color="inherit" 
                  size="small" 
                  startIcon={<UndoIcon />}
                  onClick={handleUndoFix}
                >
                  Undo
                </Button>
              </Box>
            ) : null
          }
        >
          {fixFeedback?.message}
        </Alert>
      </Snackbar>

      {/* Mermaid Fix Diff Modal */}
      <Dialog
        open={showDiffModal}
        onClose={lastFix?.pending ? handleCancelFix : () => setShowDiffModal(false)}
        maxWidth="lg"
        fullWidth
        aria-labelledby="diff-dialog-title"
      >
        <DialogTitle id="diff-dialog-title" sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pb: 0 }}>
          <Typography variant="h6">
            {lastFix?.pending ? '📋 Review Proposed Fix' : 'Diagram Fix Changes'}
          </Typography>
          <IconButton onClick={lastFix?.pending ? handleCancelFix : () => setShowDiffModal(false)} size="small">
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 3 }}>
          <Tabs 
            value={diffModalTab} 
            onChange={(e, newValue) => setDiffModalTab(newValue)}
            aria-label="diff modal tabs"
          >
            <Tab 
              icon={<CompareArrowsIcon />} 
              iconPosition="start" 
              label="Code Diff" 
              id="diff-tab-0"
              aria-controls="diff-tabpanel-0"
            />
            <Tab 
              icon={<VisibilityIcon />} 
              iconPosition="start" 
              label="Preview Fixed" 
              id="diff-tab-1"
              aria-controls="diff-tabpanel-1"
            />
          </Tabs>
        </Box>
        <DialogContent dividers>
          {lastFix && (
            <>
              {/* Code Diff Tab */}
              <Box
                role="tabpanel"
                hidden={diffModalTab !== 0}
                id="diff-tabpanel-0"
                aria-labelledby="diff-tab-0"
              >
                {diffModalTab === 0 && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Changes made by LLM fix:
                    </Typography>
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 1,
                        bgcolor: mode === 'dark' ? 'rgba(30, 30, 30, 0.9)' : 'rgba(250, 250, 250, 0.9)',
                        border: '1px solid',
                        borderColor: 'divider',
                        overflow: 'auto',
                        maxHeight: '60vh',
                        fontSize: '12px',
                        fontFamily: '"Fira Code", "Consolas", monospace',
                      }}
                    >
                      {/* Render unified diff with line-by-line highlighting */}
                      {Diff.diffLines(lastFix.original, lastFix.fixed).map((part, index) => (
                        <Box
                          key={index}
                          component="pre"
                          sx={{
                            m: 0,
                            p: 0,
                            pl: 1,
                            borderLeft: '3px solid',
                            borderLeftColor: part.added 
                              ? 'success.main' 
                              : part.removed 
                                ? 'error.main' 
                                : 'transparent',
                            bgcolor: part.added 
                              ? (mode === 'dark' ? 'rgba(46, 160, 67, 0.15)' : 'rgba(46, 160, 67, 0.1)')
                              : part.removed 
                                ? (mode === 'dark' ? 'rgba(248, 81, 73, 0.15)' : 'rgba(248, 81, 73, 0.1)')
                                : 'transparent',
                            color: part.added 
                              ? 'success.light' 
                              : part.removed 
                                ? 'error.light' 
                                : 'text.primary',
                            fontFamily: 'inherit',
                            fontSize: 'inherit',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            '&::before': {
                              content: part.added ? '"+ "' : part.removed ? '"- "' : '"  "',
                              color: part.added ? 'success.main' : part.removed ? 'error.main' : 'text.secondary',
                              fontWeight: 'bold',
                            },
                          }}
                        >
                          {part.value}
                        </Box>
                      ))}
                    </Box>
                    <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Box sx={{ width: 12, height: 12, bgcolor: 'error.main', borderRadius: 0.5 }} />
                        <Typography variant="caption" color="text.secondary">Removed</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Box sx={{ width: 12, height: 12, bgcolor: 'success.main', borderRadius: 0.5 }} />
                        <Typography variant="caption" color="text.secondary">Added</Typography>
                      </Box>
                    </Box>
                  </Box>
                )}
              </Box>

              {/* Preview Tab */}
              <Box
                role="tabpanel"
                hidden={diffModalTab !== 1}
                id="diff-tabpanel-1"
                aria-labelledby="diff-tab-1"
              >
                {diffModalTab === 1 && (
                  <Box sx={{ minHeight: '300px' }}>
                    <Typography variant="subtitle2" color="success.main" gutterBottom>
                      Fixed Diagram Preview
                    </Typography>
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 1,
                        bgcolor: mode === 'dark' ? 'rgba(100, 255, 100, 0.05)' : 'rgba(0, 255, 0, 0.02)',
                        border: '1px solid',
                        borderColor: 'success.main',
                      }}
                    >
                      <MermaidDiagram 
                        chart={lastFix.fixed} 
                        mode={mode}
                      />
                    </Box>
                  </Box>
                )}
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          {/* Retry button - always available */}
          <Button 
            onClick={handleRetryFix} 
            color="info" 
            startIcon={<RefreshIcon />}
            disabled={isFixingDiagram}
          >
            {isFixingDiagram ? 'Fixing...' : '🔄 Retry'}
          </Button>
          
          {lastFix?.pending ? (
            // Pending fix: show Cancel and Accept & Save
            <>
              <Button 
                onClick={handleCancelFix} 
                color="inherit"
                startIcon={<CloseIcon />}
              >
                ✕ Cancel
              </Button>
              <Button 
                onClick={handleAcceptFix} 
                variant="contained"
                color="success"
                startIcon={<SaveIcon />}
              >
                ✓ Accept & Save
              </Button>
            </>
          ) : (
            // Already applied fix: show Undo and Close
            <>
              <Button 
                onClick={handleUndoFix} 
                color="warning" 
                startIcon={<UndoIcon />}
              >
                Undo Fix
              </Button>
              <Button onClick={() => setShowDiffModal(false)} variant="contained">
                Close
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      {/* Chat Drawer for Ask/Research tools - only show when wiki is generated and not currently generating */}
      {!generating && (
        <ChatDrawer
          open={chatOpen}
          onClose={() => setChatOpen(false)}
          onOpen={() => setChatOpen(true)}
          projectId={projectId}
          toolkitId={toolkitId}
          toolkit={toolkit}
          wikiGenerated={!!wikiStructure || (Array.isArray(artifactsList) && artifactsList.length > 0)}
          mode={mode}
          repoIdentifierOverride={repoIdentifierOverride}
          analysisKeyOverride={analysisKeyOverride}
          authHeaders={authHeaders}
        />
      )}
    </ThemeProvider>
  );
}

export default DeepWikiApp;

const WikiPage = memo(props => {
  const {
    currentPage,
    pageContent,
    mode,
    isFixingDiagram,
    fixingBlockIndex,
    onQuickFix,
    onNavigateToError,
  } = props;

  return (
    <Box>
      <Typography
        variant="h3"
        gutterBottom
        sx={{
          fontFamily: designTokens.typography.fontFamily,
          fontSize: '28px',
          fontWeight: 600,
          lineHeight: '36px',
          color: 'text.primary',
          mb: 4,
        }}
      >
        {currentPage.title}
      </Typography>
      <Box
        sx={{
          fontFamily: designTokens.typography.fontFamily,
          '& h1': {
            fontFamily: designTokens.typography.fontFamily,
            fontSize: '24px',
            fontWeight: 600,
            lineHeight: '32px',
            color: 'text.primary',
            mt: 4,
            mb: 2,
            scrollMarginTop: '80px',
            '&:first-of-type': { mt: 0 },
          },
          '& h2': {
            fontFamily: designTokens.typography.fontFamily,
            fontSize: '20px',
            fontWeight: 600,
            lineHeight: '28px',
            color: 'text.primary',
            mt: 3.5,
            mb: 1.5,
            scrollMarginTop: '80px',
          },
          '& h3': {
            fontFamily: designTokens.typography.fontFamily,
            fontSize: '18px',
            fontWeight: 600,
            lineHeight: '26px',
            color: 'text.primary',
            mt: 3,
            mb: 1.25,
            scrollMarginTop: '80px',
          },
          '& h4, & h5, & h6': {
            fontFamily: designTokens.typography.fontFamily,
            fontSize: '16px',
            fontWeight: 600,
            lineHeight: '24px',
            color: 'text.primary',
            mt: 2.5,
            mb: 1,
            scrollMarginTop: '80px',
          },
          '& p': {
            fontFamily: designTokens.typography.fontFamily,
            ...designTokens.typography.bodyMedium,
            color: 'text.secondary',
            mb: 2,
          },
          '& ul, & ol': {
            ...designTokens.typography.bodyMedium,
            color: 'text.secondary',
            mb: 2,
            pl: 3,
          },
          '& li': {
            ...designTokens.typography.bodyMedium,
            mb: 0.75,
            color: 'text.secondary',
          },
          '& code': {
            backgroundColor: mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(61, 68, 86, 0.08)',
            padding: '3px 8px',
            borderRadius: '6px',
            fontFamily: '"Roboto Mono", monospace',
            fontSize: '13px',
            color: mode === 'dark' ? '#6ae8fa' : '#C428DD',
          },
          '& pre': {
            backgroundColor: mode === 'dark' ? '#181F2A' : '#F8FCFF',
            padding: 2.5,
            borderRadius: 2,
            overflow: 'auto',
            mb: 2.5,
            border: '1px solid',
            borderColor: 'divider',
          },
          '& pre code': {
            backgroundColor: 'transparent',
            padding: 0,
          },
          '& a': {
            color: 'primary.main',
            textDecoration: 'none',
            '&:hover': { textDecoration: 'underline' },
          },
          '& blockquote': {
            borderLeft: '4px solid',
            borderColor: 'primary.main',
            pl: 2,
            ml: 0,
            my: 2,
            color: 'text.secondary',
            fontStyle: 'italic',
          },
          '& table': {
            width: '100%',
            borderCollapse: 'collapse',
            mb: 2,
          },
          '& th, & td': {
            border: '1px solid',
            borderColor: 'divider',
            padding: 1,
            textAlign: 'left',
          },
          '& th': {
            backgroundColor: mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
            fontWeight: 600,
          },
        }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Custom table rendering for proper styling
            table: ({ children }) => (
              <Box
                sx={{
                  overflowX: 'auto',
                  mb: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                }}
              >
                <table
                  style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '14px',
                  }}
                >
                  {children}
                </table>
              </Box>
            ),
            thead: ({ children }) => (
              <thead
                style={{
                  backgroundColor: mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.04)',
                }}
              >
                {children}
              </thead>
            ),
            tbody: ({ children }) => <tbody>{children}</tbody>,
            tr: ({ children }) => (
              <tr
                style={{
                  borderBottom: '1px solid',
                  borderColor: mode === 'dark' ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)',
                }}
              >
                {children}
              </tr>
            ),
            th: ({ children }) => (
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  fontWeight: 600,
                  borderBottom: '2px solid',
                  borderColor: mode === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
                  whiteSpace: 'nowrap',
                }}
              >
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td
                style={{
                  padding: '10px 16px',
                  verticalAlign: 'top',
                }}
              >
                {children}
              </td>
            ),
            // Override pre to prevent wrapping our custom components
            pre: ({ children }) => <>{children}</>,
            // Add IDs to headings for TOC navigation
            h1: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h1 id={id}>{children}</h1>;
            },
            h2: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h2 id={id}>{children}</h2>;
            },
            h3: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h3 id={id}>{children}</h3>;
            },
            h4: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h4 id={id}>{children}</h4>;
            },
            h5: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h5 id={id}>{children}</h5>;
            },
            h6: ({ children }) => {
              const text = String(children);
              const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
              return <h6 id={id}>{children}</h6>;
            },
            code: (() => {
              // Track mermaid block index within the render cycle
              let mermaidBlockIndex = 0;

              return ({ node, inline, className, children, ...props }) => {
                const match = /language-(\w+)/.exec(className || '');
                const language = match ? match[1] : '';
                const codeString = String(children).replace(/\n$/, '');

                // Render Mermaid diagrams with fix capabilities
                if (language === 'mermaid') {
                  const currentBlockIndex = mermaidBlockIndex++;
                  return (
                    <MermaidDiagram
                      chart={codeString}
                      mode={mode}
                      blockIndex={currentBlockIndex}
                      onQuickFix={onQuickFix}
                      onNavigateToError={onNavigateToError}
                      isFixing={isFixingDiagram && fixingBlockIndex === currentBlockIndex}
                    />
                  );
                }

                // Regular code block - check if it's single line or multi-line
                if (!inline) {
                  const isSingleLine = !codeString.includes('\n');

                  // Single-line code blocks - treat as inline code to keep them in the text flow
                  if (isSingleLine) {
                    return (
                      <code
                        style={{
                          backgroundColor:
                            mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)',
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
                  }

                  // Multi-line code blocks - use CodeMirror with syntax highlighting
                  return <CodeBlock code={codeString} language={language} mode={mode} />;
                }

                // Inline code - allow wrapping when needed
                return (
                  <code
                    style={{
                      backgroundColor:
                        mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)',
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
            })(),
          }}
        >
          {pageContent}
        </ReactMarkdown>
      </Box>
    </Box>
  );
});

WikiPage.displayName = 'WikiPage';
