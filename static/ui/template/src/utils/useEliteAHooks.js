/**
 * Custom React hooks for DeepWiki using EliteA Sandbox Client
 * These hooks replace direct fetch calls with the sandbox client
 */

import { useState, useEffect, useCallback } from 'react';
import { getEliteAClient } from '../utils/eliteaClient';

/**
 * Hook to fetch and manage toolkit data
 * @param {number} projectId - Project ID
 * @param {string} toolkitId - Toolkit ID
 * @returns {Object} - Toolkit state and methods
 */
export function useToolkit(projectId, toolkitId) {
  const [toolkit, setToolkit] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchToolkit = useCallback(async () => {
    if (!projectId || !toolkitId) return;

    try {
      setLoading(true);
      setError(null);

      // Using sandbox client (though this endpoint might need custom handling)
      // For now, we'll use native fetch but structure is ready for migration
      const url = `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch toolkit: ${response.statusText}`);
      }

      const data = await response.json();
      setToolkit(data);
      
    } catch (err) {
      console.error('Error fetching toolkit:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [projectId, toolkitId]);

  useEffect(() => {
    fetchToolkit();
  }, [fetchToolkit]);

  return { toolkit, loading, error, refetch: fetchToolkit };
}

/**
 * Hook to manage artifacts using EliteA Sandbox Client
 * @param {string} bucketName - Bucket name for artifacts
 * @returns {Object} - Artifacts state and methods
 */
export function useArtifacts(bucketName) {
  const [artifacts, setArtifacts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const client = getEliteAClient();

  const loadArtifacts = useCallback(async () => {
    if (!bucketName) return;

    try {
      setLoading(true);
      setError(null);

      const artifactManager = await client.artifact(bucketName);
      const result = await artifactManager.list(null, false);
      
      // Transform the result into expected format
      const fileList = result.files || result.rows || [];
      setArtifacts(fileList);
      
    } catch (err) {
      console.error('Error loading artifacts:', err);
      setError(err.message);
      setArtifacts([]);
    } finally {
      setLoading(false);
    }
  }, [bucketName, client]);

  const createArtifact = useCallback(async (fileName, content) => {
    try {
      const artifactManager = await client.artifact(bucketName);
      await artifactManager.create(fileName, content);
      await loadArtifacts(); // Refresh list
      return { success: true };
    } catch (err) {
      console.error('Error creating artifact:', err);
      return { success: false, error: err.message };
    }
  }, [bucketName, client, loadArtifacts]);

  const deleteArtifact = useCallback(async (fileName) => {
    try {
      const artifactManager = await client.artifact(bucketName);
      await artifactManager.delete(fileName);
      await loadArtifacts(); // Refresh list
      return { success: true };
    } catch (err) {
      console.error('Error deleting artifact:', err);
      return { success: false, error: err.message };
    }
  }, [bucketName, client, loadArtifacts]);

  const getArtifact = useCallback(async (fileName) => {
    try {
      const artifactManager = await client.artifact(bucketName);
      const content = await artifactManager.get(fileName);
      return { success: true, content };
    } catch (err) {
      console.error('Error getting artifact:', err);
      return { success: false, error: err.message };
    }
  }, [bucketName, client]);

  const updateArtifact = useCallback(async (fileName, content) => {
    try {
      const artifactManager = await client.artifact(bucketName);
      await artifactManager.overwrite(fileName, content);
      await loadArtifacts(); // Refresh list
      return { success: true };
    } catch (err) {
      console.error('Error updating artifact:', err);
      return { success: false, error: err.message };
    }
  }, [bucketName, client, loadArtifacts]);

  useEffect(() => {
    loadArtifacts();
  }, [loadArtifacts]);

  return {
    artifacts,
    loading,
    error,
    createArtifact,
    deleteArtifact,
    getArtifact,
    updateArtifact,
    refetch: loadArtifacts
  };
}

/**
 * Hook to manage tool invocations with async task tracking
 * @param {string} toolkitName - Toolkit name
 * @param {string} toolName - Tool name
 * @returns {Object} - Invocation state and methods
 */
export function useToolInvocation(toolkitName, toolName) {
  const [invoking, setInvoking] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(null);

  const invoke = useCallback(async (parameters, useAsync = false) => {
    try {
      setInvoking(true);
      setError(null);
      setResult(null);
      setTaskId(null);
      setProgress(null);

      const url = `/api/v2/deepwiki_plugin/invoke/prompt_lib/${toolkitName}/${toolName}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          configuration: { parameters: {} },
          parameters: parameters
        })
      });

      if (!response.ok) {
        throw new Error(`Invocation failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (useAsync && data.task_id) {
        setTaskId(data.task_id);
        // Could use sandbox client to poll for results
        // const client = getEliteAClient();
        // const finalResult = await client.waitForTask(data.task_id, 600);
        // setResult(finalResult);
      } else {
        setResult(data);
      }

      return { success: true, data };
      
    } catch (err) {
      console.error('Error invoking tool:', err);
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setInvoking(false);
    }
  }, [toolkitName, toolName]);

  const cancel = useCallback(async (invocationId) => {
    try {
      const url = `/api/v2/deepwiki_plugin/invocations/prompt_lib/${invocationId}`;
      await fetch(url, { method: 'DELETE' });
      return { success: true };
    } catch (err) {
      console.error('Error canceling invocation:', err);
      return { success: false, error: err.message };
    }
  }, []);

  return {
    invoking,
    result,
    error,
    taskId,
    progress,
    invoke,
    cancel
  };
}

/**
 * Hook to manage MCP toolkits
 * @returns {Object} - MCP toolkits state and methods
 */
export function useMcpToolkits() {
  const [toolkits, setToolkits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const client = getEliteAClient();

  const loadToolkits = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await client.getMcpToolkits();
      setToolkits(data);
      
    } catch (err) {
      console.error('Error loading MCP toolkits:', err);
      setError(err.message);
      setToolkits([]);
    } finally {
      setLoading(false);
    }
  }, [client]);

  const callTool = useCallback(async (params) => {
    try {
      const result = await client.mcpToolCall(params);
      return { success: true, result };
    } catch (err) {
      console.error('Error calling MCP tool:', err);
      return { success: false, error: err.message };
    }
  }, [client]);

  useEffect(() => {
    loadToolkits();
  }, [loadToolkits]);

  return {
    toolkits,
    loading,
    error,
    callTool,
    refetch: loadToolkits
  };
}

/**
 * Hook to manage applications list
 * @returns {Object} - Applications state and methods
 */
export function useApplications() {
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const client = getEliteAClient();

  const loadApplications = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const apps = await client.getListOfApps();
      setApplications(apps);
      
    } catch (err) {
      console.error('Error loading applications:', err);
      setError(err.message);
      setApplications([]);
    } finally {
      setLoading(false);
    }
  }, [client]);

  const getAppDetails = useCallback(async (appId) => {
    try {
      const details = await client.getAppDetails(appId);
      return { success: true, details };
    } catch (err) {
      console.error('Error getting app details:', err);
      return { success: false, error: err.message };
    }
  }, [client]);

  useEffect(() => {
    loadApplications();
  }, [loadApplications]);

  return {
    applications,
    loading,
    error,
    getAppDetails,
    refetch: loadApplications
  };
}
