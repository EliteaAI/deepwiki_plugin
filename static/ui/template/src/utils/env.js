/* global process */
/* eslint-env node */

/**
 * Get environment variable from multiple sources
 * Priority: window.deepwiki_ui_config > import.meta.env > process.env
 * 
 * @param {string} key - Environment variable key
 * @param {*} fallback - Default value if not found
 * @returns {*} Environment variable value or fallback
 */
export function getEnvVar(key, fallback = undefined) {
  // 1. Check deepwiki_ui_config on globalThis or window (production runtime)
  const config =
    typeof globalThis !== 'undefined' && globalThis.deepwiki_ui_config
      ? globalThis.deepwiki_ui_config
      : typeof window !== 'undefined' && window.deepwiki_ui_config
        ? window.deepwiki_ui_config
        : undefined;
  if (config && Object.prototype.hasOwnProperty.call(config, key.toLowerCase())) {
    return config[key.toLowerCase()];
  }

  // 2. Try Vite's import.meta.env (development build-time)
  try {
    if (typeof import.meta !== 'undefined' && import.meta.env && key in import.meta.env) {
      return import.meta.env[key];
    }
  } catch {
    // Ignore if import.meta is not available
  }

  // 3. Check globalThis.__ENV__ (for Jest or custom setups)
  if (typeof globalThis !== 'undefined' && globalThis.__ENV__ && key in globalThis.__ENV__) {
    return globalThis.__ENV__[key];
  }

  // 4. Check process.env (Node, only if process is defined)
  if (typeof process !== 'undefined' && process.env && key in process.env) {
    return process.env[key];
  }

  return fallback;
}
