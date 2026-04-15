/**
 * Socket.io hook for DeepWiki UI
 * Provides socket connection and event subscription utilities
 * Based on inventory_plugin implementation
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

// Socket instance (singleton pattern for this app)
let socketInstance = null;

/**
 * Get or create socket.io connection
 * Uses platform's socket.io server with session auth
 */
function getSocket() {
  if (socketInstance && socketInstance.connected) {
    return socketInstance;
  }

  // Socket.io path on the platform
  const socketPath = '/socket.io';

  // Connect to same origin (platform proxy handles routing)
  const socketServer = window.location.origin;

  const ioOptions = {
    path: socketPath,
    reconnectionDelayMax: 2000,
    withCredentials: true, // Include session cookies
  };

  // Add dev token in development mode
  if (import.meta.env.DEV && import.meta.env.VITE_DEV_TOKEN) {
    ioOptions.extraHeaders = {
      Authorization: `Bearer ${import.meta.env.VITE_DEV_TOKEN}`,
    };
  }

  socketInstance = io(socketServer, ioOptions);

  socketInstance.on('connect', () => {
    console.log('[Socket] Connected to platform');
  });

  socketInstance.on('connect_error', (err) => {
    console.warn('[Socket] Connection error:', err.message);
  });

  socketInstance.on('disconnect', () => {
    console.log('[Socket] Disconnected');
  });

  return socketInstance;
}

/**
 * Socket.io event names
 */
export const sioEvents = {
  chat_predict: 'chat_predict',
  chat_leave_rooms: 'chat_leave_rooms',
  chat_enter_room: 'chat_enter_room',
  socket_validation_error: 'socket_validation_error',
  test_toolkit_tool: 'test_toolkit_tool',
  test_toolkit_enter_room: 'test_toolkit_enter_room',
  test_toolkit_leave_room: 'test_toolkit_leave_room',
  application_predict: 'application_predict',
};

/**
 * Socket message types (from platform)
 */
export const SocketMessageType = {
  StartTask: 'start_task',
  Chunk: 'chunk',
  AIMessageChunk: 'AIMessageChunk',
  AgentResponse: 'agent_response',
  AgentStart: 'agent_start',
  AgentToolStart: 'agent_tool_start',
  AgentToolEnd: 'agent_tool_end',
  AgentToolError: 'agent_tool_error',
  AgentLlmStart: 'agent_llm_start',
  AgentLlmChunk: 'agent_llm_chunk',
  AgentLlmEnd: 'agent_llm_end',
  AgentThinkingStep: 'agent_thinking_step',
  AgentThinkingStepUpdate: 'agent_thinking_step_update',
  AgentException: 'agent_exception',
  References: 'references',
  Error: 'error',
  LlmError: 'llm_error',
};

/**
 * Hook to use socket.io connection
 * Automatically subscribes to event on mount and unsubscribes on unmount
 *
 * @param {string} event - Socket event to subscribe to
 * @param {Function} handler - Event handler function
 * @returns {{ emit: Function, connected: boolean }}
 */
export function useSocket(event, handler) {
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const handlerRef = useRef(handler);

  // Keep handler ref up to date
  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    const socket = getSocket();
    socketRef.current = socket;

    // Track connection state
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    setConnected(socket.connected);

    // Subscribe to event
    const eventHandler = (data) => {
      handlerRef.current?.(data);
    };

    if (event && handler) {
      console.log('[Socket] Subscribing to', event);
      socket.on(event, eventHandler);
    }

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);

      if (event && handler) {
        console.log('[Socket] Unsubscribing from', event);
        socket.off(event, eventHandler);
      }
    };
  }, [event, handler]);

  const emit = useCallback((payload) => {
    const socket = socketRef.current;
    if (socket && socket.connected) {
      return socket.emit(event, payload);
    }

    // Try reconnecting
    const newSocket = getSocket();
    if (newSocket.disconnected) {
      newSocket.connect();
    }

    return newSocket?.emit(event, payload);
  }, [event]);

  return { emit, connected };
}

/**
 * Hook for manual socket event control
 * Does NOT auto-subscribe - call subscribe() manually
 *
 * @param {string} event - Socket event name
 * @param {Function} handler - Optional handler to subscribe
 * @returns {{ subscribe, unsubscribe, emit }}
 */
export function useManualSocket(event, handler) {
  const socketRef = useRef(null);
  const handlerRef = useRef(handler);
  // Stable wrapper function that always calls the latest handler
  const stableHandlerRef = useRef(null);
  // Track if we're currently subscribed to avoid duplicate listeners
  const isSubscribedRef = useRef(false);

  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    socketRef.current = getSocket();
    // Create stable wrapper once - it always delegates to handlerRef.current
    stableHandlerRef.current = (data) => {
      if (handlerRef.current) {
        handlerRef.current(data);
      }
    };
  }, []);

  const subscribe = useCallback(() => {
    const socket = socketRef.current || getSocket();
    if (socket && stableHandlerRef.current) {
      // Avoid duplicate subscriptions
      if (isSubscribedRef.current) {
        console.log('[Socket] Already subscribed to', event, '- skipping');
        return;
      }
      console.log('[Socket] Manually subscribing to', event);
      // Use stable wrapper so handler updates are reflected
      socket.on(event, stableHandlerRef.current);
      isSubscribedRef.current = true;
    }
  }, [event]);

  const unsubscribe = useCallback(() => {
    const socket = socketRef.current;
    if (socket && stableHandlerRef.current) {
      console.log('[Socket] Manually unsubscribing from', event);
      // Remove the stable wrapper we subscribed with
      socket.off(event, stableHandlerRef.current);
      isSubscribedRef.current = false;
    }
  }, [event]);

  const emit = useCallback((payload) => {
    const socket = socketRef.current || getSocket();
    if (socket) {
      if (socket.disconnected) {
        socket.connect();
      }
      return socket.emit(event, payload);
    }
    return false;
  }, [event]);

  return { subscribe, unsubscribe, emit };
}

/**
 * Get socket ID (useful for passing to backend for room-based events)
 */
export function getSocketId() {
  const socket = getSocket();
  return socket?.id || null;
}

/**
 * Emit any socket event directly
 * Useful for emitting events that aren't part of the component's main subscription
 * @param {string} eventName - The socket event name
 * @param {object} payload - The payload to send
 * @returns {boolean} - True if emit was called
 */
export function emitSocketEvent(eventName, payload) {
  const socket = getSocket();
  if (socket) {
    if (socket.disconnected) {
      socket.connect();
    }
    socket.emit(eventName, payload);
    return true;
  }
  return false;
}

/**
 * Disconnect socket (cleanup)
 */
export function disconnectSocket() {
  if (socketInstance) {
    socketInstance.disconnect();
    socketInstance = null;
  }
}

export { getSocket };
export default useSocket;
