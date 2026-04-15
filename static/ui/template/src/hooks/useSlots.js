import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * @typedef {Object} SlotsStatus
 * @property {number} available - Number of available generation slots
 * @property {number} total - Total number of slots in the cluster
 * @property {number} active - Number of currently active jobs
 * @property {boolean} canStart - Whether a new generation can be started
 * @property {'jobs' | 'subprocess'} mode - Current backend mode
 */

/**
 * @typedef {Object} UseSlotsResult
 * @property {SlotsStatus | null} slots - Current slots status
 * @property {boolean} loading - Whether the slots are being fetched
 * @property {string | null} error - Error message if fetch failed
 * @property {() => void} refresh - Force refresh the slots status
 * @property {boolean} slotsAvailable - Convenience boolean for checking availability
 */

/**
 * Custom hook for fetching and monitoring DeepWiki cluster slot availability.
 * 
 * This hook provides real-time information about how many wiki generation slots
 * are available across the entire cluster (when using K8s Jobs mode) or on the
 * current pod (when using subprocess mode).
 * 
 * @param {Object} options - Hook options
 * @param {string} options.baseUrl - Base URL for the DeepWiki API
 * @param {number} [options.pollingInterval=5000] - Polling interval in ms (0 to disable)
 * @param {boolean} [options.enabled=true] - Whether to enable polling
 * @returns {UseSlotsResult} Slots status and control functions
 * 
 * @example
 * const { slots, loading, slotsAvailable, refresh } = useSlots({
 *   baseUrl: 'http://deepwiki:8088',
 *   pollingInterval: 5000,
 * });
 * 
 * if (!slotsAvailable) {
 *   return <SlotsFullWarning slots={slots} />;
 * }
 */
const useSlots = ({ baseUrl, pollingInterval = 5000, enabled = true }) => {
  const [slots, setSlots] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);
  const abortControllerRef = useRef(null);

  const fetchSlots = useCallback(async () => {
    if (!baseUrl) {
      setError('No baseUrl provided');
      setLoading(false);
      return;
    }

    // Cancel any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${baseUrl}/slots`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch slots: ${response.status}`);
      }

      const data = await response.json();
      const normalized = {
        ...data,
        canStart: data.canStart ?? data.can_start,
      };
      setSlots(normalized);
      setError(null);
    } catch (err) {
      if (err.name === 'AbortError') {
        // Request was cancelled, ignore
        return;
      }
      setError(err.message || 'Failed to fetch slots');
    } finally {
      setLoading(false);
    }
  }, [baseUrl]);

  // Initial fetch and polling setup
  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchSlots();

    // Set up polling if interval > 0
    if (pollingInterval > 0) {
      intervalRef.current = setInterval(fetchSlots, pollingInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchSlots, pollingInterval, enabled]);

  // Convenience accessor for slot availability
  const slotsAvailable = slots?.canStart ?? false;

  return {
    slots,
    loading,
    error,
    refresh: fetchSlots,
    slotsAvailable,
  };
};

export default useSlots;
