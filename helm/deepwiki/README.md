# DeepWiki Helm Chart

This Helm chart deploys DeepWiki - an AI-powered wiki generation service that analyzes code repositories and generates comprehensive documentation.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (for persistence)

## Architecture

DeepWiki deploys as a **standalone plugin** with no external dependencies (no Redis, no message queue). Each pod manages its own wiki generation workers.

```
┌─────────────────────────────────────────────────────────────────┐
│  deepwiki (Deployment)                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Pod (replicas: 1-5)                                       │  │
│  │  - HTTP API (/health, /descriptor, /invoke)               │  │
│  │  - Wiki Generation (DEEPWIKI_MAX_PARALLEL_WORKERS per pod)│  │
│  │  - UI Serving                                              │  │
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
| `combined.maxParallelWorkers` | Max concurrent wiki generations per pod | `2` |
| `combined.resources.requests.memory` | Memory request | `4Gi` |
| `combined.resources.limits.memory` | Memory limit | `8Gi` |
| `combined.terminationGracePeriodSeconds` | Grace period for generation completion | `3600` |
| `storage.persistence.enabled` | Enable persistent storage | `true` |
| `storage.persistence.size` | PVC size | `50Gi` |
| `podDisruptionBudget.enabled` | Protect running generations | `true` |

### Sizing Guide

| Repo Size | Memory/Pod | Workers/Pod | Recommended Replicas |
|-----------|------------|-------------|---------------------|
| 500 files | 4Gi | 2 | 1 |
| 3,000 files | 8Gi | 2 | 2 |
| 10,000 files | 16Gi | 2-3 | 2-3 |

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

Each pod manages its own worker slots. Scaling horizontally increases total cluster capacity:

| Replicas | Workers/Pod | Total Concurrent | Memory/Pod | Total Memory |
|----------|-------------|------------------|------------|--------------|
| 1 | 2 | 2 | 8Gi | 8Gi |
| 2 | 2 | 4 | 8Gi | 16Gi |
| 3 | 2 | 6 | 8Gi | 24Gi |
| 5 | 3 | 15 | 16Gi | 80Gi |

Enable HPA for automatic scaling:

```yaml
combined:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 5
    targetMemoryUtilizationPercentage: 75
```

### Vertical Scaling

Increase resources for larger repositories:

```yaml
combined:
  resources:
    requests:
      cpu: 2
      memory: 8Gi
    limits:
      cpu: 8
      memory: 16Gi
  maxParallelWorkers: 3  # More memory = more workers
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
