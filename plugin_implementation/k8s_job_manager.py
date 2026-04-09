#!/usr/bin/env python3
# coding=utf-8

"""
K8s Job Manager for DeepWiki.

This module handles creation, monitoring, and cleanup of Kubernetes Jobs
for wiki generation. It provides cluster-wide slot management and log streaming.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


class K8sJobManager:
    """Manages Kubernetes Jobs for wiki generation."""
    
    def __init__(
        self,
        namespace: str = None,
        max_concurrent_jobs: int = None,
        base_path: str = None,
    ):
        """
        Initialize the Job Manager.
        
        Args:
            namespace: K8s namespace for jobs (default: DEEPWIKI_NAMESPACE or 'deepwiki')
            max_concurrent_jobs: Max concurrent jobs (default: DEEPWIKI_MAX_CONCURRENT_JOBS or 3)
            base_path: Base directory for wiki data.  Should come from the
                       plugin YAML config (``self.descriptor.config['base_path']``
                       via ``runtime_config()``).  Falls back to the
                       ``DEEPWIKI_BASE_PATH`` env var or ``/data/wiki_builder``.
        """
        self.namespace = namespace or os.environ.get("DEEPWIKI_NAMESPACE", "deepwiki")
        self.max_concurrent_jobs = max_concurrent_jobs or int(
            os.environ.get("DEEPWIKI_MAX_CONCURRENT_JOBS", "3")
        )
        # Canonical base_path: prefer explicit config value > env var > hardcoded default.
        # On the controller the value comes from deepwiki_plugin.yml (set by Helm).
        self.base_path = base_path or os.environ.get(
            "DEEPWIKI_BASE_PATH", "/data/wiki_builder"
        )
        self.jobs_dir = Path(self.base_path) / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Worker image defaults to current pod's image
        self.worker_image = os.environ.get(
            "DEEPWIKI_WORKER_IMAGE",
            os.environ.get("DEEPWIKI_IMAGE", "pylon:latest")
        )
        # Image pull policy for worker/init containers
        # IfNotPresent works for both registry and local (minikube) images.
        # Set to Never for strictly-local images that must not hit a registry.
        self.image_pull_policy = os.environ.get(
            "DEEPWIKI_WORKER_IMAGE_PULL_POLICY", "IfNotPresent"
        )
        
        # TTL for completed jobs (seconds)
        self.ttl_seconds = int(os.environ.get("DEEPWIKI_JOB_TTL_SECONDS", "300"))
        
        # Resource requests/limits
        self.resources = {
            "requests": {
                "memory": os.environ.get("DEEPWIKI_JOB_MEMORY_REQUEST", "2Gi"),
                "cpu": os.environ.get("DEEPWIKI_JOB_CPU_REQUEST", "1"),
            },
            "limits": {
                "memory": os.environ.get("DEEPWIKI_JOB_MEMORY_LIMIT", "8Gi"),
                "cpu": os.environ.get("DEEPWIKI_JOB_CPU_LIMIT", "4"),
            }
        }
        
        # PVC name for shared data
        self.pvc_name = os.environ.get("DEEPWIKI_PVC_NAME", "deepwiki-data")
        
        # Platform artifact transport (Jobs+API / emptyDir workers)
        # K8s Job mode always uses platform artifact transport: the controller
        # uploads job input and downloads results via the platform API, and
        # workers read/write indexes to the same bucket.  The bucket name is
        # hardcoded ("wiki_artifacts"); per-request llm_settings supply the
        # actual API credentials (base_url, token, project_id).
        from .artifacts_platform_client import ARTIFACT_ENV_VARS
        self._platform_transport = True
        self._artifact_env_var_names = ARTIFACT_ENV_VARS
        self._platform_client = None  # lazy
        
        # Plugin bootstrap credentials (for init container)
        # These are needed to git-clone the plugin and pip-install its deps
        # into the worker pod — replicating what Pylon bootstrap does on the controller.
        self.plugin_branch = os.environ.get(
            "DEEPWIKI_PLUGIN_BRANCH",
            os.environ.get("ELITEA_RELEASE", "main")
        )
        # Fallback chain: DEEPWIKI_LICENSE_* → LICENSE_* → bootstrap.yml config → empty
        # SaaS deployments may only set LICENSE_PASSWORD (from GCP/secret) without the
        # DEEPWIKI_ prefix, and the username may only exist in the bootstrap config file.
        self.license_username = os.environ.get(
            "DEEPWIKI_LICENSE_USERNAME",
            os.environ.get("LICENSE_USERNAME", "")
        )
        self.license_password = os.environ.get(
            "DEEPWIKI_LICENSE_PASSWORD",
            os.environ.get("LICENSE_PASSWORD", "")
        )
        # If username is still empty, try reading from bootstrap config on disk
        if not self.license_username:
            self.license_username = self._read_bootstrap_license_username()
        
        self._k8s_client = None
        self._batch_v1 = None
        self._core_v1 = None
    
    @staticmethod
    def _read_bootstrap_license_username() -> str:
        """Try to read license_username from bootstrap.yml config file.
        
        SaaS deployments mount bootstrap.yml at /data/configs/bootstrap.yml
        with plugin_repo[].license_username but don't always set
        DEEPWIKI_LICENSE_USERNAME as an env var.
        """
        bootstrap_paths = [
            "/data/configs/bootstrap.yml",
            "/data/configs/bootstrap.yaml",
        ]
        for path in bootstrap_paths:
            try:
                import yaml
                with open(path, "r") as f:
                    cfg = yaml.safe_load(f)
                repos = cfg.get("plugin_repo") or []
                for repo in repos:
                    username = repo.get("license_username", "")
                    if username and not username.startswith("${"):
                        log.info("Resolved license_username from %s", path)
                        return username
            except Exception:
                continue
        return ""
    
    def _init_k8s_client(self):
        """Initialize K8s client (lazy loading).
        
        Note: For thread safety during concurrent job monitoring,
        individual methods should create their own API client instances.
        """
        if self._k8s_client is not None:
            return
        
        try:
            from kubernetes import client, config
            
            try:
                config.load_incluster_config()
                log.info("Loaded in-cluster K8s config")
            except config.ConfigException:
                config.load_kube_config()
                log.info("Loaded local kubeconfig")
            
            self._k8s_client = client
            self._batch_v1 = client.BatchV1Api()
            self._core_v1 = client.CoreV1Api()
            
        except ImportError:
            raise RuntimeError("kubernetes package not installed")
    
    def _get_batch_api(self):
        """Get a BatchV1Api instance for this operation.
        
        Creates a new instance to avoid thread safety issues when
        multiple jobs are being monitored concurrently.
        """
        self._init_k8s_client()
        from kubernetes import client
        return client.BatchV1Api()
    
    def _get_core_api(self):
        """Get a CoreV1Api instance for this operation.
        
        Creates a new instance to avoid thread safety issues.
        """
        self._init_k8s_client()
        from kubernetes import client
        return client.CoreV1Api()
    
    # ------------------------------------------------------------------
    # Platform artifact transport helpers (Jobs+API)
    # ------------------------------------------------------------------
    
    def _uses_platform_transport(self) -> bool:
        """True when platform artifact transport is configured (emptyDir workers)."""
        return self._platform_transport
    
    def _get_platform_client(self):
        """Lazy-create platform artifact client from env vars."""
        if self._platform_client is None and self._platform_transport:
            from .artifacts_platform_client import create_platform_client_from_env
            self._platform_client = create_platform_client_from_env()
        return self._platform_client
    
    def _get_platform_client_from_llm_settings(self, llm_settings: Dict[str, Any]):
        """Create platform artifact client from llm_settings (per-request)."""
        from .artifacts_platform_client import create_platform_client_from_llm_settings
        return create_platform_client_from_llm_settings(llm_settings)
    
    def _upload_job_input(self, job_id: str, input_data: Dict[str, Any],
                          platform_client=None) -> bool:
        """Upload job input to platform bucket (Jobs+API mode)."""
        client = platform_client or self._get_platform_client()
        if not client:
            log.error("Cannot upload job input — no platform client")
            return False
        try:
            from .artifacts_platform_client import get_artifact_bucket
            bucket = get_artifact_bucket()
            payload = json.dumps(input_data, indent=2, default=str)
            client.upload_artifact(bucket, f"jobs/{job_id}/input.json", payload)
            log.info("Uploaded job input to platform bucket: jobs/%s/input.json", job_id)
            return True
        except Exception as exc:
            log.error("Failed to upload job input to platform bucket: %s", exc)
            return False
    
    def _download_job_result_from_platform(self, job_id: str,
                                           platform_client=None) -> Optional[Dict[str, Any]]:
        """Download result.json from platform bucket (Jobs+API mode)."""
        client = platform_client or self._get_platform_client()
        if not client:
            return None
        try:
            from .artifacts_platform_client import get_artifact_bucket
            bucket = get_artifact_bucket()
            data = client.download_artifact(bucket, f"jobs/{job_id}/result.json")
            result = json.loads(data)
            # Cache locally for potential re-reads
            result_file = self.jobs_dir / job_id / "result.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            log.info("Downloaded result from platform bucket: jobs/%s/result.json", job_id)
            return result
        except Exception as exc:
            log.warning("Failed to download result from platform for job %s: %s", job_id, exc)
            return None
    
    def _cleanup_platform_job_objects(self, job_id: str, platform_client=None) -> None:
        """Delete job I/O objects from platform bucket."""
        client = platform_client or self._get_platform_client()
        if not client:
            return
        try:
            from .artifacts_platform_client import get_artifact_bucket
            bucket = get_artifact_bucket()
            for key in [f"jobs/{job_id}/input.json", f"jobs/{job_id}/result.json"]:
                try:
                    client.delete_artifact(bucket, key)
                except Exception:
                    pass
            log.debug("Cleaned up platform bucket objects for job %s", job_id)
        except Exception as exc:
            log.debug("Platform bucket cleanup failed for job %s: %s", job_id, exc)

    def generate_job_id(self) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"{timestamp}-{unique}"
    
    def get_slot_availability(self) -> Dict[str, Any]:
        """
        Get current slot availability.
        
        Returns:
            {
                "available": int,
                "total": int,
                "active": int,
                "can_start": bool
            }
        """
        batch_v1 = self._get_batch_api()
        
        jobs = batch_v1.list_namespaced_job(
            namespace=self.namespace,
            label_selector="app=deepwiki-worker"
        )
        
        active_count = sum(
            1 for job in jobs.items 
            if job.status.active and job.status.active > 0
        )
        
        available = max(0, self.max_concurrent_jobs - active_count)
        
        return {
            "available": available,
            "total": self.max_concurrent_jobs,
            "active": active_count,
            "can_start": active_count < self.max_concurrent_jobs
        }
    
    def _build_init_container(self, client):
        """
        Build an init container that bootstraps plugin code + dependencies.
        
        This replicates what Pylon's bootstrap plugin does on the controller:
        1. Git-clone the deepwiki_plugin from GitHub (EliteaAI org)
        2. Pip-install the plugin's requirements.txt into the standard path
        
        After the init container completes, the worker container finds the code
        and deps at the same paths the controller uses:
        - /data/plugins/deepwiki_plugin/        (plugin source)
        - /data/requirements/deepwiki_plugin/... (pip packages)
        
        Credentials come from the controller's env vars, which are set by
        the Helm chart from values.yaml (config.licenseUsername, etc.).
        
        Returns:
            V1Container spec for the init container, or None if credentials
            are not configured (PVC-based fallback).
        """
        plugin_path = "/data/plugins/deepwiki_plugin"
        requirements_path = "/data/requirements/deepwiki_plugin/lib/python3.12/site-packages"
        # Pre-baked deps live at this path inside the worker image
        # (built via Dockerfile.worker).  NOT masked by the emptyDir
        # volume at /data.
        prebaked_path = "/opt/deepwiki/lib/python3.12/site-packages"
        prebaked_marker = "/opt/deepwiki/.prebaked"
        
        # If using a pre-baked local image (Dockerfile.worker), the init
        # container is redundant — code + deps are already in the image.
        # DEEPWIKI_SKIP_WORKER_INIT=1 lets callers (e.g. values-minikube)
        # opt out even when license credentials are configured.
        if os.environ.get("DEEPWIKI_SKIP_WORKER_INIT", "").strip() == "1":
            log.info(
                "DEEPWIKI_SKIP_WORKER_INIT=1 — skipping Job init container. "
                "Worker will use baked-in code at /app/deepwiki_plugin "
                "and pre-installed deps at /opt/deepwiki/."
            )
            return None

        # If no credentials configured, skip init container.
        # Pre-baked image: code is at /app/deepwiki_plugin — no clone needed.
        # Vanilla pylon image + PVC: relies on PVC having plugin code.
        if not self.license_username or not self.license_password:
            if self._uses_platform_transport():
                log.info(
                    "Plugin bootstrap credentials not configured. "
                    "Worker will use baked-in code at /app/deepwiki_plugin. "
                    "Set DEEPWIKI_LICENSE_USERNAME / DEEPWIKI_LICENSE_PASSWORD "
                    "to enable code override via init container."
                )
            else:
                log.warning(
                    "Plugin bootstrap credentials not configured "
                    "(DEEPWIKI_LICENSE_USERNAME / DEEPWIKI_LICENSE_PASSWORD). "
                    "Init container will be skipped — worker relies on PVC "
                    "having plugin code pre-populated by the controller."
                )
            return None
        
        # Shell script that clones plugin code and optionally installs deps.
        #
        # Pre-baked image (Dockerfile.worker):
        #   Deps already at /opt/deepwiki/... — only git-clone needed.
        #   Detected via marker file at /opt/deepwiki/.prebaked.
        #
        # Vanilla Pylon image (no pre-baked deps):
        #   Full pip install with -U, retries, and increased timeout.
        #   This is the FALLBACK path — slow (~5-10min) and network-
        #   dependent.  Use the pre-baked image in production.
        bootstrap_script = f"""set -e
echo "[init] DeepWiki plugin bootstrap starting..."
echo "[init] Branch: $PLUGIN_BRANCH"

# --- 1. Clone plugin source code --------------------------------
if [ -d "{plugin_path}/plugin_implementation" ]; then
    echo "[init] Plugin code already exists at {plugin_path}, skipping clone"
else
    echo "[init] Cloning deepwiki_plugin..."
    git clone --depth 1 --branch "$PLUGIN_BRANCH" \\
        "https://$LICENSE_USERNAME:$LICENSE_PASSWORD@github.com/EliteaAI/deepwiki_plugin.git" \\
        "{plugin_path}"
    echo "[init] Clone complete"
fi

# --- 2. Install pip dependencies --------------------------------
if [ -f "{prebaked_marker}" ]; then
    echo "[init] Pre-baked deps detected ({prebaked_marker}), skipping pip install"
    cat "{prebaked_marker}"
elif [ -d "{requirements_path}" ] && [ "$(ls -A {requirements_path} 2>/dev/null)" ]; then
    echo "[init] Requirements already installed at {requirements_path}, skipping pip install"
else
    echo "[init] No pre-baked deps — installing via pip (this may take several minutes)..."
    mkdir -p "{requirements_path}"
    # Use CPU-only PyTorch index to avoid downloading ~1.5GB of CUDA
    # binaries (nvidia_cublas, nvidia_cudnn, cuda_bindings, etc.).
    # DeepWiki uses faiss-cpu + sentence-transformers — no GPU needed.
    pip install --no-cache-dir -U \\
        --retries 3 --timeout 300 \\
        --extra-index-url https://download.pytorch.org/whl/cpu \\
        --target "{requirements_path}" \\
        -r "{plugin_path}/requirements.txt"
    echo "[init] Pip install complete"
fi

echo "[init] Bootstrap finished successfully"
"""
        
        return client.V1Container(
            name="plugin-bootstrap",
            image=self.worker_image,
            image_pull_policy=self.image_pull_policy,
            command=["/bin/sh", "-c", bootstrap_script],
            env=[
                client.V1EnvVar(name="PLUGIN_BRANCH", value=self.plugin_branch),
                client.V1EnvVar(name="LICENSE_USERNAME", value=self.license_username),
                client.V1EnvVar(name="LICENSE_PASSWORD", value=self.license_password),
            ],
            security_context=client.V1SecurityContext(
                run_as_user=33,
                run_as_group=33,
            ),
            volume_mounts=[
                client.V1VolumeMount(
                    name="data",
                    mount_path="/data"
                )
            ],
            # Pre-baked image: init is lightweight (git clone only, ~10s).
            # Vanilla image: pip install needs memory for torch wheel
            # extraction (~2GB wheel → needs 3-4Gi during install).
            resources=client.V1ResourceRequirements(
                requests={"memory": "512Mi", "cpu": "500m"},
                limits={"memory": "4Gi", "cpu": "2"}
            )
        )
    
    def create_job(self, job_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a K8s Job for wiki generation.
        
        Args:
            job_id: Unique job identifier
            input_data: Input parameters for wiki generation
            
        Returns:
            Job creation result with status
        """
        # Use kubernetes library for spec building (thread-safe)
        from kubernetes import client
        
        # Check slot availability first
        slots = self.get_slot_availability()
        if not slots["can_start"]:
            return {
                "success": False,
                "error": f"[SERVICE_BUSY] All {slots['total']} generation slots are in use",
                "active_workers": slots["active"],
                "max_workers": slots["total"]
            }
        
        # Create job directory and write input
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_file = job_dir / "input.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=2)
        
        log.info(f"Created job input at {input_file}")
        
        # In platform-transport mode (Jobs+API), derive artifact settings
        # from llm_settings and upload input.json to platform bucket so
        # the worker can download it (emptyDir starts empty).
        platform_client = None
        if self._uses_platform_transport():
            from .artifacts_platform_client import (
                extract_artifact_settings, inject_artifact_env_vars,
                get_artifact_bucket,
            )
            llm_settings = input_data.get("llm_settings", {})
            platform_client = self._get_platform_client_from_llm_settings(llm_settings)
            if not platform_client:
                # Fallback to env-based client
                platform_client = self._get_platform_client()
            if not self._upload_job_input(job_id, input_data, platform_client=platform_client):
                return {
                    "success": False,
                    "error": "Failed to upload job input to platform bucket",
                    "error_category": "platform_upload_failed",
                }
        
        # Build Job spec
        job_name = f"deepwiki-worker-{job_id}"
        
        # Plugin code paths:
        #   /data/plugins/deepwiki_plugin  — init container git-clone (override)
        #   /app/deepwiki_plugin           — baked into worker image (fallback)
        # The shell command tries /data/ first, falls back to /app/.
        plugin_path = "/data/plugins/deepwiki_plugin"
        baked_plugin_path = "/app/deepwiki_plugin"
        worker_module = "plugin_implementation/wiki_job_worker.py"
        
        # Requirements paths:
        #   /data/requirements/...  — init container pip install (fallback)
        #   /opt/deepwiki/...       — pre-baked in worker image (preferred)
        requirements_path = "/data/requirements/deepwiki_plugin/lib/python3.12/site-packages"
        prebaked_path = "/opt/deepwiki/lib/python3.12/site-packages"
        
        # Build environment variables - inherit key settings from controller
        # PYTHONPATH priority:
        #   pre-baked deps (correct arch) > runtime pip deps > clone code > baked code
        # Pre-baked deps from worker image are architecture-matched;
        # PVC requirements may be from a different arch (e.g. ARM init container).
        # Non-existent paths are silently ignored by Python.
        env_vars = [
            client.V1EnvVar(name="DEEPWIKI_JOB_ID", value=job_id),
            client.V1EnvVar(name="DEEPWIKI_BASE_PATH", value=self.base_path),
            client.V1EnvVar(
                name="PYTHONPATH",
                value=(
                    f"{prebaked_path}:{requirements_path}"
                    f":{plugin_path}:{baked_plugin_path}:/data/plugins"
                )
            ),
            client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
        ]
        
        # Pass through ALL DEEPWIKI_* environment variables from the controller
        # to the worker pod.  This ensures feature flags (DOC_SEPARATE_INDEX,
        # ENABLE_FTS5, AGENTIC_*, etc.) are inherited without maintaining a
        # manual whitelist.  Variables explicitly set above (JOB_ID, BASE_PATH)
        # are skipped to avoid duplicates.
        _already_set = {ev.name for ev in env_vars}
        for var_name, var_value in os.environ.items():
            if var_name.startswith("DEEPWIKI_") and var_name not in _already_set:
                env_vars.append(client.V1EnvVar(name=var_name, value=var_value))
        
        # In platform-transport mode, inject artifact API credentials
        # so the worker can download input.json and upload results.
        # These are derived per-request from llm_settings (the platform
        # provides fresh credentials for each invocation).
        if self._uses_platform_transport() and platform_client is not None:
            llm_settings = input_data.get("llm_settings", {})
            art_settings = extract_artifact_settings(llm_settings)
            art_env = inject_artifact_env_vars(art_settings, get_artifact_bucket())
            for var_name, var_value in art_env.items():
                if var_value:
                    env_vars.append(client.V1EnvVar(name=var_name, value=var_value))
        
        # Build init container for plugin bootstrap (clone + pip install)
        init_container = self._build_init_container(client)
        init_containers = [init_container] if init_container else None
        
        # Determine volume type: platform transport → emptyDir, else PVC
        uses_platform = self._uses_platform_transport()
        
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "deepwiki-worker",
                    "job-id": job_id,
                    "created-by": "deepwiki-controller"
                }
            ),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=self.ttl_seconds,
                backoff_limit=0,  # No retries - fail fast
                # active_deadline_seconds=3600,  # no need since there might be the runs for the big repos which need way more than 1 hour.
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "deepwiki-worker",
                            "job-id": job_id
                        }
                    ),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        # Worker Jobs make zero K8s API calls — no RBAC needed.
                        # Use a minimal service account (default: "deepwiki-worker").
                        # The RBAC-enabled SA is only for the controller.
                        service_account_name=os.environ.get(
                            "DEEPWIKI_WORKER_SERVICE_ACCOUNT", "deepwiki-worker"
                        ),
                        security_context=client.V1PodSecurityContext(
                            run_as_user=33,
                            run_as_group=33,
                            fs_group=33,
                        ),
                        # Init container bootstraps plugin code + pip deps
                        # into /data/plugins/ and /data/requirements/.
                        # Skipped if credentials aren't configured (PVC fallback).
                        init_containers=init_containers,
                        containers=[
                            client.V1Container(
                                name="worker",
                                image=self.worker_image,
                                image_pull_policy=self.image_pull_policy,
                                # Smart path resolution: prefer init-container
                                # clone at /data/plugins/, fall back to baked-in
                                # code at /app/.  Works with both vanilla Pylon
                                # and the custom worker image.
                                command=["/bin/sh", "-c"],
                                args=[
                                    f'W={plugin_path}/{worker_module}; '
                                    f'[ -f "$W" ] || W={baked_plugin_path}/{worker_module}; '
                                    f'exec python "$W" --job-id={job_id}'
                                ],
                                env=env_vars,
                                security_context=client.V1SecurityContext(
                                    run_as_user=33,
                                    run_as_group=33,
                                ),
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="data",
                                        mount_path="/data"
                                    )
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests=self.resources["requests"],
                                    limits=self.resources["limits"]
                                )
                            )
                        ],
                        volumes=[
                            # Platform-transport mode: emptyDir (worker is stateless).
                            # PVC mode: shared PVC (controller + worker share filesystem).
                            client.V1Volume(
                                name="data",
                                empty_dir=client.V1EmptyDirVolumeSource()
                            ) if uses_platform else
                            client.V1Volume(
                                name="data",
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                    claim_name=self.pvc_name
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        try:
            batch_v1 = self._get_batch_api()
            created_job = batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            log.info(f"Created K8s Job: {job_name}")
            
            return {
                "success": True,
                "job_id": job_id,
                "job_name": job_name,
                "namespace": self.namespace,
                "status": "created"
            }
            
        except Exception as e:
            log.error(f"Failed to create K8s Job: {e}")
            # Clean up input file on failure
            try:
                input_file.unlink()
                job_dir.rmdir()
            except Exception:
                pass
            
            return {
                "success": False,
                "error": f"Failed to create K8s Job: {e}"
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status including phase, pod status, etc.
        """
        batch_v1 = self._get_batch_api()
        
        job_name = f"deepwiki-worker-{job_id}"
        
        try:
            job = batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            # Determine job phase
            if job.status.succeeded and job.status.succeeded > 0:
                phase = "succeeded"
            elif job.status.failed and job.status.failed > 0:
                phase = "failed"
            elif job.status.active and job.status.active > 0:
                phase = "running"
            else:
                phase = "pending"
            
            return {
                "success": True,
                "job_id": job_id,
                "job_name": job_name,
                "phase": phase,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "start_time": job.status.start_time.isoformat() if job.status.start_time else None,
                "completion_time": job.status.completion_time.isoformat() if job.status.completion_time else None
            }
            
        except Exception as e:
            if "NotFound" in str(e):
                return {
                    "success": False,
                    "job_id": job_id,
                    "phase": "not_found",
                    "error": f"Job {job_name} not found"
                }
            log.error(f"Error getting job status: {e}")
            return {
                "success": False,
                "job_id": job_id,
                "phase": "error",
                "error": str(e)
            }
    
    def get_job_failure_info(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed failure information for a failed job.
        
        Extracts error details from multiple K8s sources:
        1. Pod termination message (/dev/termination-log)
        2. Container exit code and reason
        3. Job conditions (BackoffLimitExceeded, DeadlineExceeded, etc.)
        4. Pod events for OOM kills, evictions, etc.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict with error details:
            {
                "error": str,           # Human-readable error message
                "error_category": str,  # Categorized error type
                "exit_code": int,       # Container exit code
                "termination_reason": str,  # K8s termination reason
                "termination_message": str, # Full termination message from pod
            }
        """
        batch_v1 = self._get_batch_api()
        core_v1 = self._get_core_api()
        
        job_name = f"deepwiki-worker-{job_id}"
        result = {
            "error": "Unknown job failure",
            "error_category": "unknown_error",
            "exit_code": None,
            "termination_reason": None,
            "termination_message": None,
        }
        
        try:
            # 1. Check job conditions for high-level failure reasons
            job = batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            if job.status.conditions:
                for condition in job.status.conditions:
                    if condition.type == "Failed" and condition.status == "True":
                        reason = condition.reason or "Unknown"
                        message = condition.message or ""
                        
                        if reason == "BackoffLimitExceeded":
                            result["error"] = "Job exceeded retry limit"
                            result["error_category"] = "backoff_limit_exceeded"
                        elif reason == "DeadlineExceeded":
                            result["error"] = "Job exceeded deadline"
                            result["error_category"] = "deadline_exceeded"
                        else:
                            result["error"] = f"Job failed: {reason}"
                            result["error_category"] = reason.lower()
                        
                        if message:
                            result["error"] = f"{result['error']}: {message}"
            
            # 2. Get pod details for container-level failure info
            pods = core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )
            
            if pods.items:
                pod = pods.items[0]
                
                # Check container statuses
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state.terminated:
                            terminated = container_status.state.terminated
                            
                            result["exit_code"] = terminated.exit_code
                            result["termination_reason"] = terminated.reason
                            
                            # Read termination message from pod (our custom message)
                            if terminated.message:
                                result["termination_message"] = terminated.message
                                
                                # Parse structured error from termination message
                                if "DeepWiki" in terminated.message:
                                    # Our structured format
                                    result["error"] = terminated.message.split("\n")[0]
                                    for line in terminated.message.split("\n"):
                                        if line.startswith("Category: "):
                                            result["error_category"] = line[10:].strip()
                                        elif line.startswith("Error: "):
                                            result["error"] = line[7:].strip()
                            
                            # Map K8s termination reasons to categories
                            if terminated.reason == "OOMKilled":
                                result["error"] = "Job ran out of memory (OOMKilled)"
                                result["error_category"] = "out_of_memory"
                            elif terminated.reason == "Error" and terminated.exit_code == 137:
                                result["error"] = "Job was killed (possibly OOM or resource limits)"
                                result["error_category"] = "killed"
                            elif terminated.reason == "Error" and terminated.exit_code == 143:
                                result["error"] = "Job was terminated (SIGTERM)"
                                result["error_category"] = "terminated"
                
                # 3. Check pod events for additional context (eviction, etc.)
                try:
                    events = core_v1.list_namespaced_event(
                        namespace=self.namespace,
                        field_selector=f"involvedObject.name={pod.metadata.name}"
                    )
                    
                    for event in events.items:
                        if event.reason in ("Evicted", "Preempted", "OutOfmemory", "OOMKilling"):
                            result["error"] = f"Pod was {event.reason}: {event.message or ''}"
                            result["error_category"] = event.reason.lower()
                            break
                        elif event.reason == "FailedScheduling":
                            result["error"] = f"Pod scheduling failed: {event.message or ''}"
                            result["error_category"] = "scheduling_failed"
                            break
                except Exception as e:
                    log.debug(f"Failed to fetch pod events: {e}")
                    
        except Exception as e:
            log.warning(f"Error getting job failure info: {e}")
            result["error"] = f"Failed to get job details: {e}"
        
        return result
    
    def get_job_pod_name(self, job_id: str, timeout: int = 60) -> Optional[str]:
        """
        Get the pod name for a job (waits for pod creation).
        
        Args:
            job_id: Job identifier
            timeout: Max seconds to wait for pod
            
        Returns:
            Pod name or None if not found
        """
        core_v1 = self._get_core_api()
        
        job_name = f"deepwiki-worker-{job_id}"
        
        for _ in range(timeout):
            try:
                pods = core_v1.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=f"job-name={job_name}"
                )
                
                if pods.items:
                    return pods.items[0].metadata.name
                
            except Exception as e:
                log.warning(f"Error listing pods: {e}")
            
            import time
            time.sleep(1)
        
        return None
    
    def stream_job_logs(
        self,
        job_id: str,
        callback: Callable[[str], None],
        timeout: int = 60
    ):
        """
        Stream job pod logs asynchronously.
        
        Args:
            job_id: Job identifier
            callback: Function to call for each log line
            timeout: Max seconds to wait for pod to start
        """
        self._init_k8s_client()
        
        # Use a dedicated CoreV1Api per stream to avoid concurrency issues
        core_v1 = self._k8s_client.CoreV1Api()

        # Wait for pod to be created
        pod_name = self.get_job_pod_name(job_id, timeout=timeout)
        if not pod_name:
            callback(f"[ERROR] Pod for job {job_id} not created within {timeout}s")
            return
        
        # Wait for pod to be running
        import time
        for _ in range(timeout):
            try:
                pod = core_v1.read_namespaced_pod(
                    name=pod_name,
                    namespace=self.namespace
                )
                if pod.status.phase in ("Running", "Succeeded", "Failed"):
                    break
            except Exception:
                pass
            time.sleep(1)
        
        # Stream logs
        try:
            from kubernetes import watch
            
            w = watch.Watch()
            for line in w.stream(
                core_v1.read_namespaced_pod_log,
                name=pod_name,
                namespace=self.namespace,
                follow=True
            ):
                callback(line)
                
        except Exception as e:
            callback(f"[ERROR] Log streaming failed: {e}")
    
    def read_job_result(self, job_id: str,
                        llm_settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Read the result of a completed job.
        
        Tries local PVC first (Jobs+PVC mode), then falls back to
        downloading from platform bucket (Jobs+API / emptyDir mode).
        
        Args:
            job_id: Job identifier
            llm_settings: Optional llm_settings dict for creating a platform
                client on the controller (which lacks artifact env vars).
            
        Returns:
            Job result dict or None if not found
        """
        # Try local file first (PVC mode, or previously downloaded)
        result_file = self.jobs_dir / job_id / "result.json"
        
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Error reading local job result: {e}")
        
        # In platform-transport mode, try downloading from platform bucket.
        # The controller pod does NOT have DEEPWIKI_ARTIFACT_BASE_URL etc.,
        # so we build a per-request client from llm_settings when available.
        if self._uses_platform_transport():
            platform_client = None
            if llm_settings:
                platform_client = self._get_platform_client_from_llm_settings(llm_settings)
            if not platform_client:
                platform_client = self._get_platform_client()  # env fallback
            return self._download_job_result_from_platform(job_id, platform_client=platform_client)
        
        return None
    
    def cleanup_job(self, job_id: str, delete_k8s_job: bool = False,
                    llm_settings: Optional[Dict[str, Any]] = None):
        """
        Clean up job files, bucket objects, and optionally the K8s Job.
        
        Args:
            job_id: Job identifier
            delete_k8s_job: Whether to delete the K8s Job resource
            llm_settings: Optional llm_settings dict for creating a platform
                client on the controller (which lacks artifact env vars).
        """
        # Clean up local files
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            import shutil
            try:
                shutil.rmtree(job_dir)
                log.info(f"Cleaned up job directory: {job_dir}")
            except Exception as e:
                log.warning(f"Failed to clean up job directory: {e}")
        
        # Clean up platform bucket objects (Jobs+API mode)
        if self._uses_platform_transport():
            platform_client = None
            if llm_settings:
                platform_client = self._get_platform_client_from_llm_settings(llm_settings)
            self._cleanup_platform_job_objects(job_id, platform_client=platform_client)
        
        # Optionally delete K8s Job
        if delete_k8s_job:
            batch_v1 = self._get_batch_api()
            job_name = f"deepwiki-worker-{job_id}"
            
            try:
                batch_v1.delete_namespaced_job(
                    name=job_name,
                    namespace=self.namespace,
                    propagation_policy="Background"
                )
                log.info(f"Deleted K8s Job: {job_name}")
            except Exception as e:
                if "NotFound" not in str(e):
                    log.warning(f"Failed to delete K8s Job: {e}")


# Singleton instance
_job_manager: Optional[K8sJobManager] = None


def get_job_manager(base_path: str = None) -> K8sJobManager:
    """Get or create the singleton job manager.

    Args:
        base_path: Base directory from the plugin YAML config
                   (``runtime_config()["base_path"]``).  Only used on the
                   first call that creates the singleton; ignored afterwards.
    """
    global _job_manager
    if _job_manager is None:
        _job_manager = K8sJobManager(base_path=base_path)
    return _job_manager
