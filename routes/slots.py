#!/usr/bin/python3
# coding=utf-8

"""
Slots Route - Cluster-wide slot availability for K8s Jobs mode.

This endpoint returns the current availability of wiki generation slots
across the entire cluster (not just per-pod).

In Jobs mode: Counts active K8s Jobs
In Subprocess mode: Returns per-pod worker counts (fallback)
"""

import os
import logging
from pylon.core.tools import web

log = logging.getLogger(__name__)


def _normalize_slots_payload(payload: dict) -> dict:
    """Ensure slots payload contains canStart alias."""
    if "canStart" not in payload and "can_start" in payload:
        payload["canStart"] = payload["can_start"]
    return payload


def _get_k8s_job_slots():
    """Get slot availability from K8s Jobs API"""
    try:
        from kubernetes import client, config
        
        # Load K8s config (in-cluster or local kubeconfig)
        try:
            config.load_incluster_config()
        except config.ConfigException:
            # Fall back to local kubeconfig for development
            config.load_kube_config()
        
        batch_v1 = client.BatchV1Api()
        
        namespace = os.environ.get("DEEPWIKI_NAMESPACE", "deepwiki")
        max_jobs = int(os.environ.get("DEEPWIKI_MAX_CONCURRENT_JOBS", "3"))
        
        # List jobs with our label
        jobs = batch_v1.list_namespaced_job(
            namespace=namespace,
            label_selector="app=deepwiki-worker"
        )
        
        # Count active jobs (status.active > 0)
        active_count = sum(
            1 for job in jobs.items 
            if job.status.active and job.status.active > 0
        )
        
        available = max(0, max_jobs - active_count)
        
        payload = {
            "available": available,
            "total": max_jobs,
            "active": active_count,
            "can_start": active_count < max_jobs,
            "mode": "jobs",
            "namespace": namespace
        }

        return _normalize_slots_payload(payload)
        
    except ImportError as ie:
        log.warning(
            "kubernetes package not importable: %s. "
            "Falling back to subprocess mode.",
            ie,
        )
        return _get_subprocess_slots()
    except Exception as e:
        log.error(f"K8s API error: {e}, falling back to subprocess mode")
        # Don't fail - return subprocess mode as fallback
        return _get_subprocess_slots()


def _get_subprocess_slots():
    """Get slot availability from subprocess worker pool (per-pod)"""
    try:
        # Import the worker pool to get current state
        from plugin_implementation.worker_pool import get_worker_pool_status
        
        status = get_worker_pool_status()
        active = status.get("active_workers", 0)
        total = status.get("max_workers", 1)
        available = max(0, total - active)
        
        payload = {
            "available": available,
            "total": total,
            "active": active,
            "can_start": active < total,
            "mode": "subprocess",
            "note": "Per-pod availability only (subprocess mode)"
        }

        return _normalize_slots_payload(payload)
    except ImportError:
        # Worker pool module doesn't exist yet, use environment variable
        max_workers = int(os.environ.get("DEEPWIKI_MAX_PARALLEL_WORKERS", "1"))
        payload = {
            "available": max_workers,  # Assume all available if we can't check
            "total": max_workers,
            "active": 0,
            "can_start": True,
            "mode": "subprocess",
            "note": "Unable to determine active workers, assuming all available"
        }
        return _normalize_slots_payload(payload)
    except Exception as e:
        log.error(f"Error getting subprocess slots: {e}")
        max_workers = int(os.environ.get("DEEPWIKI_MAX_PARALLEL_WORKERS", "1"))
        payload = {
            "available": max_workers,
            "total": max_workers,
            "active": 0,
            "can_start": True,
            "mode": "subprocess",
            "error": str(e)
        }
        return _normalize_slots_payload(payload)


class Route:
    """Slots availability route"""

    @web.route("/slots", methods=["GET"])
    def slots_route(self):  # pylint: disable=no-self-use
        """
        Return cluster-wide slot availability.
        
        Response:
        {
            "available": 2,      # Slots currently free
            "total": 3,          # Max concurrent jobs
            "active": 1,         # Currently running
            "can_start": true,   # Whether a new job can be started
            "mode": "jobs"       # "jobs" or "subprocess"
        }
        """
        try:
            # Check if Jobs mode is enabled
            jobs_enabled = os.environ.get("DEEPWIKI_JOBS_ENABLED", "false").lower() == "true"
            
            if jobs_enabled:
                return _get_k8s_job_slots()
            else:
                return _get_subprocess_slots()
                
        except Exception as e:
            log.error(f"Error getting slots: {e}")
            payload = {
                "available": 0,
                "total": 0,
                "active": 0,
                "can_start": False,
                "mode": "error",
                "error": str(e)
            }
            return _normalize_slots_payload(payload), 500
