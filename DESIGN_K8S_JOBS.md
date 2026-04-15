# DeepWiki K8s Jobs-Based Scaling Architecture

## Overview

This document describes the architecture for running wiki generation as Kubernetes Jobs instead of in-process subprocess workers. This approach provides:

1. **Cluster-wide slot visibility** - Users see true availability (`2/3 slots`) not per-pod state
2. **Resource isolation** - Each generation runs in its own pod
3. **Auto-scaling potential** - K8s can manage resources and scheduling
4. **Better failure handling** - K8s restarts failed jobs automatically

## Tools Analysis

| Tool | Current Behavior | Uses Slots? | Should Use Jobs? | Notes |
|------|------------------|-------------|------------------|-------|
| **generate_wiki** | Subprocess worker, heavy (clone, index, LLM calls) | YES | YES | Primary candidate - slot-limited, heavy resource usage |
| **ask** | Subprocess worker, uses existing vector store | NO | NO | Lightweight - queries existing index, no slot limit |
| **deep_research** | Subprocess worker, LLM-heavy multi-iteration | NO (currently) | YES (recommended) | Heavy LLM calls, multiple iterations, should be slot-limited |

### generate_wiki
- **Weight**: Heavy (clone repo, build vector store, build graph, multiple LLM calls)
- **Slot-limited**: YES (current `DEEPWIKI_MAX_PARALLEL_WORKERS`)
- **Subprocess**: `wiki_subprocess_worker.py`
- **Output**: Wiki pages, manifest, structure JSON

### ask
- **Weight**: Light (queries existing vector store, single LLM call)
- **Slot-limited**: NO (runs immediately)
- **Subprocess**: `ask_subprocess_worker.py` (via `_run_ask_subprocess`)
- **Output**: Answer text
- **Decision**: Keep as subprocess - lightweight, uses cached data

### deep_research
- **Weight**: Heavy (multiple LLM iterations, web search, thinking steps)
- **Slot-limited**: NO (currently runs without limits)
- **Subprocess**: `deep_research_subprocess_worker.py`
- **Output**: Report, findings, todos
- **Decision**: Should be Jobs-based OR add slot limits - currently unbound

## UI Impact

### Current UI Behavior (DeepWikiApp.jsx)

```javascript
// Error detection for slots full
const isServiceBusyMarker =
  contentStr.includes('[SERVICE_BUSY]') || 
  contentStr.includes('slots taken');

// Displays per-pod worker count
const activeWorkers = contentObj?.active_workers;  // e.g., 1
const maxWorkers = contentObj?.max_workers;        // e.g., 1

// Shows: "1/1 slots taken" - but this is per-pod, not cluster-wide!
message: `Max parallel wiki generations reached: ${activeWorkers}/${maxWorkers} slots taken.`
```

### Required UI Changes

1. **New `/slots` endpoint call** - UI should check availability BEFORE starting generation
2. **Cluster-wide display** - Show `2/3 slots available` not per-pod
3. **Pre-check on Generate button** - Disable button or show warning if no slots
4. **Progress from Job logs** - Parse K8s pod logs for progress events

```javascript
// New: Check slots before generation
async function checkSlots() {
  const response = await fetch('/slots');
  const { available, total, can_start } = await response.json();
  return { available, total, can_start };
}

// New: Display cluster-wide availability
<span>{available}/{total} slots available</span>
```

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Platform (elitea_core)                                                      │
│   ↓ invoke RPC                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ DeepWiki Plugin Pod (invoke.py)                                             │
│   ├─ Check slot availability (Python global - per-pod only)                │
│   ├─ Acquire slot                                                           │
│   ├─ Fork subprocess (wiki_subprocess_worker.py)                            │
│   │   ├─ Clone/update repo                                                  │
│   │   ├─ Build vector store + graph (→ PVC)                                │
│   │   ├─ Run LangGraph agent                                                │
│   │   ├─ Generate wiki (in memory)                                          │
│   │   ├─ Export artifacts via ArtifactExporter                              │
│   │   ├─ Build manifest JSON                                                │
│   │   └─ Write result to --output file                                      │
│   ├─ Read result, stream progress via socket.io                             │
│   ├─ Build result_objects[] from artifacts                                  │
│   ├─ Release slot                                                           │
│   └─ Return result_objects to platform                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Platform (elitea_core)                                                      │
│   ├─ Parse result_objects from result JSON                                  │
│   ├─ For each artifact: PUT to artifacts API / S3                           │
│   └─ Update toolkit run status                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What stays on disk (PVC - for caching/reuse):
- Vector storage (FAISS index)
- Code graph (NetworkX pickle)
- Repository analysis JSON (for Ask tool reuse)
- BM25 index

### What goes to platform (in-memory → result_objects):
- Wiki pages (markdown) - generated in memory by ArtifactExporter
- Wiki structure (JSON) - generated in memory by ArtifactExporter
- Wiki manifest (JSON) - built by wiki_subprocess_worker.py
- Repository context (dual: cached on disk + passed as result_object)

## Proposed Jobs-Based Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DeepWiki Controller Pod (Deployment, always running)                        │
│                                                                             │
│  Endpoints:                                                                 │
│  ├─ GET /slots → Query K8s API for active jobs, return availability        │
│  ├─ POST /invoke → Pre-check slots, create Job if available                │
│  ├─ GET /job/{id}/logs → Stream job pod logs via K8s API                   │
│  └─ GET /job/{id}/result → Read result from shared PVC                     │
│                                                                             │
│  Flow:                                                                      │
│  1. Receive invoke request                                                  │
│  2. Query: kubectl get jobs -l app=deepwiki-worker --field-selector=...    │
│  3. If activeJobs >= maxConcurrentJobs → Return SERVICE_BUSY error         │
│  4. Create K8s Job with unique ID                                           │
│  5. Return job_id to caller, start streaming logs                           │
│  6. Poll job status until complete                                          │
│  7. Read result from /data/jobs/{job_id}/result.json                        │
│  8. Build result_objects, return to platform                                │
│  9. Cleanup: delete result file (Job auto-deleted via TTL)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Creates
┌─────────────────────────────────────────────────────────────────────────────┐
│ DeepWiki Worker Job (K8s Job, ephemeral)                                    │
│                                                                             │
│  Spec:                                                                      │
│  - Same image as controller                                                 │
│  - Command: python wiki_job_worker.py --job-id=xxx --input=/data/jobs/...   │
│  - Mounts shared PVC at /data                                               │
│  - TTL after completion: 300s (auto-cleanup)                                │
│  - Resource requests/limits for proper scheduling                           │
│                                                                             │
│  Flow:                                                                      │
│  1. Read input from /data/jobs/{job_id}/input.json                          │
│  2. Execute wiki generation (same as wiki_subprocess_worker.py)             │
│     ├─ Clone/update repo                                                    │
│     ├─ Build vector store + graph (→ /data/cache/...)                       │
│     ├─ Run LangGraph agent                                                  │
│     ├─ Generate wiki (in memory)                                            │
│     ├─ Export artifacts via ArtifactExporter                                │
│     └─ Build manifest JSON                                                  │
│  3. Write result to /data/jobs/{job_id}/result.json                         │
│  4. Write progress to stdout (for log streaming)                            │
│  5. Exit with code 0 on success, 1 on failure                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Shared storage
┌─────────────────────────────────────────────────────────────────────────────┐
│ Shared PVC (/data)                                                          │
│                                                                             │
│  /data/                                                                     │
│  ├─ wiki_builder/          # Existing structure                             │
│  │   ├─ repos/             # Cloned repositories                            │
│  │   ├─ cache/             # Vector stores, graphs, BM25                    │
│  │   └─ analysis/          # Repository analysis JSON                       │
│  │                                                                          │
│  ├─ jobs/                  # NEW: Job I/O directory                         │
│  │   ├─ {job_id}/                                                           │
│  │   │   ├─ input.json     # Job parameters (written by controller)        │
│  │   │   ├─ result.json    # Job output (written by worker)                 │
│  │   │   └─ progress.json  # Optional: progress updates                     │
│  │   └─ ...                                                                 │
│  │                                                                          │
│  └─ plugins/               # Bootstrap plugins (existing)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components to Implement

### 1. Slot Availability Endpoint

```python
# routes/slots.py
@web.route("/slots", methods=["GET"])
def get_slots(self):
    """Return cluster-wide slot availability"""
    from kubernetes import client, config
    
    # Load in-cluster config (or kubeconfig for local dev)
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    
    batch_v1 = client.BatchV1Api()
    namespace = os.environ.get("DEEPWIKI_NAMESPACE", "deepwiki")
    max_jobs = int(os.environ.get("DEEPWIKI_MAX_CONCURRENT_JOBS", "3"))
    
    # Count active jobs
    jobs = batch_v1.list_namespaced_job(
        namespace=namespace,
        label_selector="app=deepwiki-worker"
    )
    
    active_count = sum(1 for j in jobs.items if j.status.active)
    
    return {
        "available": max_jobs - active_count,
        "total": max_jobs,
        "active": active_count,
        "can_start": active_count < max_jobs
    }
```

### 2. Job Creation

```python
# plugin_implementation/k8s_job_manager.py
def create_wiki_job(job_id: str, input_data: dict) -> str:
    """Create a K8s Job for wiki generation"""
    
    # Write input to shared PVC
    job_dir = Path(f"/data/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / "input.json", "w") as f:
        json.dump(input_data, f)
    
    # Create Job spec
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=f"deepwiki-worker-{job_id}",
            labels={"app": "deepwiki-worker", "job-id": job_id}
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=300,  # Auto-cleanup
            backoff_limit=0,  # No retries (handled at higher level)
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="worker",
                            image=os.environ["DEEPWIKI_WORKER_IMAGE"],
                            command=["python", "wiki_job_worker.py"],
                            args=[f"--job-id={job_id}"],
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="data",
                                    mount_path="/data"
                                )
                            ],
                            resources=client.V1ResourceRequirements(
                                requests={"memory": "2Gi", "cpu": "1"},
                                limits={"memory": "8Gi", "cpu": "4"}
                            )
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="data",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name="deepwiki-data"
                            )
                        )
                    ]
                )
            )
        )
    )
    
    batch_v1.create_namespaced_job(namespace=namespace, body=job)
    return job_id
```

### 3. Log Streaming

```python
# plugin_implementation/k8s_job_manager.py
async def stream_job_logs(job_id: str, socket_emit_fn):
    """Stream job pod logs to socket.io"""
    core_v1 = client.CoreV1Api()
    
    # Wait for pod to be created
    pod_name = None
    for _ in range(60):  # 60s timeout
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name=deepwiki-worker-{job_id}"
        )
        if pods.items:
            pod_name = pods.items[0].metadata.name
            break
        await asyncio.sleep(1)
    
    if not pod_name:
        raise TimeoutError("Job pod not created")
    
    # Stream logs
    w = watch.Watch()
    for line in w.stream(
        core_v1.read_namespaced_pod_log,
        name=pod_name,
        namespace=namespace,
        follow=True
    ):
        # Parse and emit progress events
        socket_emit_fn(parse_log_line(line))
```

### 4. Job Worker Entry Point

```python
# plugin_implementation/wiki_job_worker.py
"""
K8s Job worker for wiki generation.
Similar to wiki_subprocess_worker.py but designed for Job execution.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    
    job_dir = Path(f"/data/jobs/{args.job_id}")
    
    # Read input
    with open(job_dir / "input.json") as f:
        input_data = json.load(f)
    
    try:
        # Execute wiki generation (reuse existing logic)
        result = run_wiki_generation(input_data)
        result["success"] = True
    except Exception as e:
        result = {"success": False, "error": str(e)}
    
    # Write result
    with open(job_dir / "result.json", "w") as f:
        json.dump(result, f)
    
    return 0 if result["success"] else 1
```

## Streaming Strategy

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **K8s logs API** | Stream via `kubectl logs -f` equivalent | Native, no extra deps | Slight latency |
| **Progress file** | Worker writes to `/data/jobs/{id}/progress.json` | Simple polling | Not real-time |
| **Redis pub/sub** | Worker publishes, controller subscribes | Real-time | Extra dependency |

**Recommendation**: Use K8s logs API for simplicity. Worker writes progress to stdout, controller streams logs and parses progress events.

## Configuration (Helm values.yaml)

```yaml
# K8s Jobs mode configuration
jobs:
  enabled: true  # Enable Jobs-based scaling (vs subprocess mode)
  maxConcurrentJobs: 3  # Cluster-wide limit
  ttlSecondsAfterFinished: 300  # Auto-cleanup after completion
  workerImage: ""  # Defaults to controller image
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "8Gi"
      cpu: "4"
```

## RBAC Requirements

Controller pod needs permissions to manage Jobs:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: deepwiki-controller
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "delete", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
```

## Migration Path

1. **Phase 1**: Implement Job-based mode behind feature flag (`jobs.enabled: false` by default)
2. **Phase 2**: Test with `jobs.enabled: true` in staging
3. **Phase 3**: Make Jobs mode default, deprecate subprocess mode
4. **Phase 4**: Remove subprocess mode

## Impact Analysis

### What stays the same:
- ArtifactExporter - generates artifacts in memory, returns list
- Manifest JSON building - same logic in wiki_job_worker.py
- Result object structure - platform sees identical result_objects
- PVC structure for caching - same paths and formats

### What changes:
- Slot tracking: Per-pod Python global → K8s Job count
- Process isolation: Subprocess → Separate pod
- Streaming: Direct stdout → K8s logs API
- Error handling: Process exit code → Job status

### Risk areas:
1. **Streaming latency** - K8s logs API may have slight delay
2. **PVC access** - Multiple pods accessing shared PVC (ReadWriteMany required)
3. **RBAC** - Controller needs Job management permissions
4. **Resource contention** - Jobs compete for cluster resources

## Files to Create/Modify

### New files:
- `plugin_implementation/k8s_job_manager.py` - Job creation, status, log streaming
- `plugin_implementation/wiki_job_worker.py` - Job worker entry point
- `routes/slots.py` - Slot availability endpoint

### Modified files:
- `methods/invoke.py` - Route to Job-based execution when enabled
- `routes/health.py` - Add slots info to health response
- `static/ui/template/src/DeepWikiApp.jsx` - Add slots pre-check, cluster-wide display
- `static/ui/template/src/hooks/useSlots.js` - New hook for slots API
- Helm chart - Add RBAC, Job mode config

## UI Implementation Details

### New Hook: useSlots

```javascript
// hooks/useSlots.js
import { useState, useEffect, useCallback } from 'react';

export function useSlots(serviceUrl) {
  const [slots, setSlots] = useState({ available: null, total: null, loading: true });
  
  const refreshSlots = useCallback(async () => {
    try {
      const response = await fetch(`${serviceUrl}/slots`);
      const data = await response.json();
      setSlots({ 
        available: data.available, 
        total: data.total, 
        canStart: data.can_start,
        loading: false 
      });
    } catch (error) {
      setSlots({ available: null, total: null, loading: false, error });
    }
  }, [serviceUrl]);
  
  useEffect(() => {
    refreshSlots();
    const interval = setInterval(refreshSlots, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, [refreshSlots]);
  
  return { ...slots, refreshSlots };
}
```

### DeepWikiApp Changes

```javascript
// In DeepWikiApp.jsx

// 1. Import and use slots hook
const { available, total, canStart, refreshSlots } = useSlots(serviceUrl);

// 2. Show cluster-wide slot availability in header
<div className="slots-indicator">
  {available !== null && (
    <span className={available === 0 ? 'slots-full' : 'slots-available'}>
      {available}/{total} slots available
    </span>
  )}
</div>

// 3. Disable Generate button if no slots
<button 
  onClick={handleGenerate}
  disabled={!canStart || isGenerating}
  title={!canStart ? 'No slots available' : 'Generate Wiki'}
>
  Generate Wiki
</button>

// 4. Pre-check before generation
const handleGenerate = async () => {
  // Refresh slots right before starting
  await refreshSlots();
  if (!canStart) {
    setError('All generation slots are currently in use. Please wait.');
    return;
  }
  // Proceed with generation...
};
```

### Error Message Updates

```javascript
// Old: Per-pod message (confusing with multiple pods)
message: `${activeWorkers}/${maxWorkers} slots taken`

// New: Cluster-wide message (clear)
message: `All ${total} generation slots are currently in use. Please try again in a few minutes.`
```

## Testing Plan

1. **Unit tests**: Mock K8s API, test job creation/status logic
2. **Integration tests**: Run in minikube with real Jobs
3. **Load tests**: Saturate slots, verify queuing/rejection
4. **Streaming tests**: Verify log streaming works end-to-end
5. **Failure tests**: Kill job pod, verify error handling
6. **UI tests**: Verify slots display, button disable, pre-check logic

## Implementation Order

1. **Backend: `/slots` endpoint** - Returns cluster-wide availability
2. **Backend: Job manager** - Create/list/stream jobs
3. **Backend: Job worker** - Entry point for wiki generation in Job
4. **Backend: invoke.py** - Route to Jobs when enabled
5. **UI: useSlots hook** - Fetch and poll slots
6. **UI: Slots display** - Show availability in header
7. **UI: Generate button** - Pre-check and disable when full
8. **Helm: RBAC** - ServiceAccount, Role, RoleBinding
9. **Helm: Values** - Jobs configuration section
10. **Testing** - Integration tests in minikube

