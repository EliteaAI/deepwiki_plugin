#!/bin/bash
# DeepWiki Minikube Testing Script
# See SCALABILITY_KUBERNETES.md for full documentation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
NAMESPACE="deepwiki"
RELEASE_NAME="deepwiki"
CHART_DIR="$(dirname "$0")/deepwiki"

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    if ! command -v minikube &> /dev/null; then
        echo_error "minikube is not installed. Please install it first."
        echo "  brew install minikube"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        echo_error "helm is not installed. Please install it first."
        echo "  brew install helm"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo_error "kubectl is not installed. Please install it first."
        echo "  brew install kubectl"
        exit 1
    fi
    
    echo_info "All prerequisites satisfied."
}

# Start minikube
start_minikube() {
    echo_info "Starting minikube..."
    
    if minikube status | grep -q "Running"; then
        echo_info "Minikube is already running."
    else
        minikube start --cpus=4 --memory=8192 --disk-size=50g
        echo_info "Minikube started."
    fi
    
    # Enable ingress addon
    echo_info "Enabling ingress addon..."
    minikube addons enable ingress
}

# Build dependencies
build_dependencies() {
    echo_info "Validating Helm chart..."
    cd "$CHART_DIR"
    
    # Lint the chart
    helm lint .
    
    echo_info "Chart validation complete."
}

# Deploy DeepWiki
deploy() {
    echo_info "Deploying DeepWiki..."
    
    # Create namespace if not exists
    kubectl create namespace "$NAMESPACE" 2>/dev/null || true
    
    # Install or upgrade
    helm upgrade --install "$RELEASE_NAME" "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        --set combined.resources.requests.memory=2Gi \
        --set combined.resources.limits.memory=4Gi \
        --set storage.persistence.size=10Gi \
        --wait \
        --timeout 5m
    
    echo_info "DeepWiki deployed successfully."
}

# Wait for pods to be ready
wait_for_ready() {
    echo_info "Waiting for pods to be ready..."
    
    kubectl wait --namespace "$NAMESPACE" \
        --for=condition=ready pod \
        --selector="app.kubernetes.io/name=deepwiki" \
        --timeout=300s
    
    echo_info "All pods are ready."
}

# Show deployment status
show_status() {
    echo_info "Deployment status:"
    echo ""
    kubectl get pods -n "$NAMESPACE"
    echo ""
    kubectl get svc -n "$NAMESPACE"
    echo ""
    kubectl get pvc -n "$NAMESPACE"
}

# Port forward for local access
port_forward() {
    echo_info "Setting up port-forward..."
    
    # Kill any existing port-forward
    pkill -f "port-forward.*deepwiki" 2>/dev/null || true
    
    # Start port-forward in background
    kubectl port-forward -n "$NAMESPACE" svc/"$RELEASE_NAME" 8090:8080 &
    
    sleep 2
    
    echo_info "Port-forward active. Access DeepWiki at: http://localhost:8090"
}

# Test health endpoint
test_health() {
    echo_info "Testing health endpoint..."
    
    # Set up port-forward if not running
    if ! curl -s http://localhost:8090/health > /dev/null 2>&1; then
        port_forward
        sleep 2
    fi
    
    response=$(curl -s http://localhost:8090/health)
    echo "$response" | python3 -m json.tool
    
    if echo "$response" | grep -q '"status": "UP"'; then
        echo_info "Health check passed!"
    else
        echo_error "Health check failed!"
        exit 1
    fi
}

# Test descriptor endpoint
test_descriptor() {
    echo_info "Testing descriptor endpoint..."
    
    response=$(curl -s http://localhost:8090/descriptor)
    echo "$response" | python3 -m json.tool | head -50
    
    if echo "$response" | grep -q '"name": "deepwiki"'; then
        echo_info "Descriptor endpoint works!"
    else
        echo_error "Descriptor endpoint failed!"
        exit 1
    fi
}

# View logs
view_logs() {
    echo_info "Viewing logs..."
    kubectl logs -n "$NAMESPACE" deployment/"$RELEASE_NAME" --tail=100 -f
}

# Cleanup
cleanup() {
    echo_info "Cleaning up..."
    
    helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete pvc -n "$NAMESPACE" --all 2>/dev/null || true
    kubectl delete namespace "$NAMESPACE" 2>/dev/null || true
    
    # Kill port-forward
    pkill -f "port-forward.*deepwiki" 2>/dev/null || true
    
    echo_info "Cleanup complete."
}

# Stop minikube
stop_minikube() {
    echo_info "Stopping minikube..."
    minikube stop
    echo_info "Minikube stopped."
}

# Help
show_help() {
    echo "DeepWiki Minikube Testing Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start       Start minikube and prepare environment"
    echo "  deploy      Deploy DeepWiki to minikube"
    echo "  status      Show deployment status"
    echo "  forward     Set up port-forward for local access"
    echo "  test        Run health and descriptor tests"
    echo "  logs        View deployment logs"
    echo "  cleanup     Remove DeepWiki deployment"
    echo "  stop        Stop minikube"
    echo "  all         Run full setup (start, deploy, test)"
    echo ""
    echo "Examples:"
    echo "  $0 all           # Full setup and test"
    echo "  $0 deploy        # Just deploy (minikube must be running)"
    echo "  $0 logs          # View logs"
    echo "  $0 cleanup       # Remove deployment"
}

# Main
case "$1" in
    start)
        check_prerequisites
        start_minikube
        ;;
    deploy)
        check_prerequisites
        build_dependencies
        deploy
        wait_for_ready
        show_status
        ;;
    status)
        show_status
        ;;
    forward)
        port_forward
        ;;
    test)
        test_health
        test_descriptor
        ;;
    logs)
        view_logs
        ;;
    cleanup)
        cleanup
        ;;
    stop)
        cleanup
        stop_minikube
        ;;
    all)
        check_prerequisites
        start_minikube
        build_dependencies
        deploy
        wait_for_ready
        show_status
        port_forward
        sleep 3
        test_health
        test_descriptor
        echo ""
        echo_info "DeepWiki is ready! Access at: http://localhost:8090"
        echo_info "UI: http://localhost:8090/ui"
        echo_info "Health: http://localhost:8090/health"
        echo_info "Descriptor: http://localhost:8090/descriptor"
        ;;
    *)
        show_help
        ;;
esac
