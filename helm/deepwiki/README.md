# DeepWiki Helm Chart

This Helm chart deploys DeepWiki - an AI-powered wiki generation service that analyzes code repositories and generates comprehensive documentation.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (for persistence)

## Architecture

DeepWiki deploys as a **standalone plugin** with no external dependencies (no Redis, no message queue). Wiki generation runs as K8s Jobs for proper resource isolation.

```
┌─────────────────────────────────────────────────────────────────┐
│  deepwiki (Deployment)                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Pod (replicas: 1-5)                                       │  │
│  │  - HTTP API (/health, /descriptor, /invoke)               │  │
│  │  - UI Serving                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  K8s Jobs (wiki generation)                                │  │
│  │  - One Job per generation request                         │  │
│  │  - Cluster-wide concurrency via maxConcurrentJobs         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  PersistentVolumeClaim                                     │  │
│  │  - Caches, cloned repos, generated wikis                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Quick Start (Development)

```bash
# Create namespace
kubectl create namespace deepwiki

# Install with default values
helm install deepwiki ./deepwiki -n deepwiki

# Port-forward to access locally
kubectl port-forward -n deepwiki svc/deepwiki 8090:8080

# Test health endpoint
curl http://localhost:8090/health
```

### Production Installation

```bash
helm install deepwiki ./deepwiki -n deepwiki \
  -f values-production.yaml \
  --set config.serviceLocationUrl=http://deepwiki.deepwiki.svc.cluster.local:8080
```

Or with custom values:

```bash
helm install deepwiki ./deepwiki -n deepwiki \
  --set combined.replicaCount=3 \
  --set combined.resources.limits.memory=16Gi \
  --set combined.maxParallelWorkers=3 \
  --set storage.persistence.size=100Gi \
  --set storage.persistence.accessModes[0]=ReadWriteMany
```

## Configuration

See `values.yaml` for the full list of configurable parameters.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Pylon container image | `getcarrier/pylon` |
| `image.tag` | Pylon image tag | `1.2.7` |
| `combined.replicaCount` | Number of pod replicas | `1` |
| `combined.resources.requests.memory` | Memory request | `1536Mi` |
| `combined.resources.limits.memory` | Memory limit | `2Gi` |
| `combined.terminationGracePeriodSeconds` | Grace period for generation completion | `3600` |
| `jobs.enabled` | Enable K8s Jobs-based scaling | `false` |
| `jobs.maxConcurrentJobs` | Max concurrent generation jobs | `3` |
| `config.aiRunPlatform.url` | Autoregistration endpoint URL | `""` |
| `config.aiRunPlatform.token` | Bearer token for registration | `""` |
| `storage.persistence.enabled` | Enable persistent storage | `true` |
| `storage.persistence.size` | PVC size | `50Gi` |
| `podDisruptionBudget.enabled` | Protect running generations | `true` |

### Sizing Guide

| Repo Size | Memory (Job) | Max Concurrent Jobs |
|-----------|--------------|--------------------|
| 500 files | 2Gi | 3 |
| 3,000 files | 4Gi | 2 |
| 10,000 files | 8Gi | 1-2 |

### DeepWiki Feature Flags

All DeepWiki feature flags are set via the `env` section in `values.yaml`. Key flags include:

| Flag | Description | Default |
|------|-------------|---------|
| `DEEPWIKI_STRUCTURE_PLANNER` | Structure planning algorithm | `deepagents` |
| `DEEPWIKI_DEEPAGENTS_COVERAGE_CHECK` | Enable coverage validation | `1` |
| `DEEPWIKI_USE_STRUCTURED_REPO_ANALYSIS` | Use JSON analysis format | `1` |
| `DEEPWIKI_DOC_SEPARATE_INDEX` | Separate doc/code indexing | `1` |

## Integration with Elitea Platform

DeepWiki integrates with the Elitea platform (elitea_core) via provider registration:

1. **Provider Registration**: Admin registers DeepWiki in elitea_core with the `service_location_url`

Or enable autoregistration via Helm values:

```yaml
config:
  aiRunPlatform:
    url: "http://elitea-core.elitea.svc.cluster.local/api/v1/providers/register"
    token: "your-bearer-token"
```

2. **Health Checks**: elitea_core periodically checks `/health` endpoint
3. **UI Proxying**: elitea_core proxies UI requests via `/ui_host/deepwiki/ui/<project_id>/`
4. **Tool Invocation**: Platform calls `/tools/<toolkit>/<tool>/invoke` for wiki generation

The `config.serviceLocationUrl` must be accessible from elitea_core. In Kubernetes, use the internal service DNS name:

```yaml
config:
  serviceLocationUrl: http://deepwiki.deepwiki.svc.cluster.local:8080
```

## Persistence

The chart mounts a PersistentVolumeClaim at `/data` containing:

- `/data/pylon.db` - Pylon database
- `/data/wiki_builder/` - Generated wikis, caches, cloned repos
- `/data/cache/` - Pip cache, model cache
- `/data/plugins/` - Plugin code

For multi-replica setups, use `ReadWriteMany` access mode:

```yaml
storage:
  persistence:
    accessModes:
      - ReadWriteMany
```

## Scaling

### Horizontal Scaling

With Jobs-based scaling enabled, generation capacity is controlled by `jobs.maxConcurrentJobs` regardless of API pod count. API pods handle routing only.

Enable HPA for API pod autoscaling:

```yaml
combined:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 5
    targetMemoryUtilizationPercentage: 75
```

### Job Resources

Configure per-job resources for wiki generation:

```yaml
jobs:
  enabled: true
  maxConcurrentJobs: 3
  resources:
    requests:
      memory: 2Gi
      cpu: "1"
    limits:
      memory: 8Gi
      cpu: "4"
```

## Graceful Shutdown

Wiki generation can take 10-60 minutes. The chart configures:

- `terminationGracePeriodSeconds: 3600` - Allow up to 1 hour for completion
- `podDisruptionBudget.minAvailable: 1` - Protect running generations during updates

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n deepwiki
kubectl describe pod -n deepwiki <pod-name>
```

### View logs
```bash
kubectl logs -n deepwiki deployment/deepwiki -f
```

### Test health endpoint
```bash
kubectl port-forward -n deepwiki svc/deepwiki 8090:8080 &
curl http://localhost:8090/health
curl http://localhost:8090/descriptor
```

### Check persistent volume
```bash
kubectl get pvc -n deepwiki
kubectl exec -n deepwiki deployment/deepwiki -- ls -la /data
kubectl exec -n deepwiki deployment/deepwiki -- du -sh /data/wiki_builder/*
```

## Uninstallation

```bash
helm uninstall deepwiki -n deepwiki
kubectl delete pvc -n deepwiki --all  # Optional: remove persistent data
kubectl delete namespace deepwiki
```

## Values Files

- `values.yaml` - Default development values
- `values-production.yaml` - Production-ready configuration
- `values-development.yaml` - Minimal resources for local testing
