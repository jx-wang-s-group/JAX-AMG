#!/usr/bin/env bash
# JAX-AMG Installation Script
# Usage: bash scripts/install.sh [OPTIONS]
#
# Options:
#   --mpi         Install with MPI support
#   --dev         Include development dependencies
#   --help        Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  JAX-AMG Installation Script${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

show_help() {
    echo "JAX-AMG Installation Script"
    echo
    echo "Usage: bash scripts/install.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --mpi       Install with MPI support"
    echo "  --dev       Include development dependencies"
    echo "  --help      Show this help message"
    echo
    echo "Environment Variables:"
    echo "  CUDA_HOME   Path to CUDA toolkit"
    echo "  AMGX_ROOT   Path to AmgX source directory"
    echo "  AMGX_BUILD  Path to AmgX build directory (default: \$AMGX_ROOT/build)"
    echo "  MPI_HOME    Path to MPI installation (optional, for custom MPI path)"
    echo
    echo "Examples:"
    echo "  bash scripts/install.sh                    # Single-GPU installation"
    echo "  bash scripts/install.sh --mpi              # MPI installation"
    echo "  bash scripts/install.sh --mpi --dev        # MPI + dev tools"
}

# Parse command line arguments
INSTALL_MPI=false
INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mpi)
            INSTALL_MPI=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run 'bash scripts/install.sh --help' for usage."
            exit 1
            ;;
    esac
done

print_header

# Step 1: Check Python
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    print_error "Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python $PYTHON_VERSION"

# Step 2: Check/Set CUDA_HOME
print_info "Checking CUDA installation..."
if [[ -z "$CUDA_HOME" ]]; then
    # Try to auto-detect
    if command -v nvcc &> /dev/null; then
        CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export CUDA_HOME
    elif [[ -d "/usr/local/cuda" ]]; then
        CUDA_HOME="/usr/local/cuda"
        export CUDA_HOME
    else
        print_error "CUDA not found. Please set CUDA_HOME environment variable."
        exit 1
    fi
fi

if [[ ! -f "$CUDA_HOME/bin/nvcc" ]]; then
    print_error "CUDA_HOME set to '$CUDA_HOME' but nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi
print_success "CUDA_HOME: $CUDA_HOME"

# Step 3: Check/Set AMGX_ROOT and AMGX_BUILD
print_info "Checking AmgX installation..."
if [[ -z "$AMGX_ROOT" ]]; then
    # Try common locations
    for loc in "$HOME/amgx" "$HOME/AMGX" "/opt/amgx" "/usr/local/amgx"; do
        if [[ -f "$loc/include/amgx_c.h" ]]; then
            AMGX_ROOT="$loc"
            export AMGX_ROOT
            break
        fi
    done
fi

if [[ -z "$AMGX_ROOT" ]] || [[ ! -f "$AMGX_ROOT/include/amgx_c.h" ]]; then
    print_error "AmgX not found. Please set AMGX_ROOT environment variable."
    echo
    echo "To build AmgX from source, see https://github.com/NVIDIA/AMGX"
    exit 1
fi
print_success "AMGX_ROOT: $AMGX_ROOT"

if [[ -z "$AMGX_BUILD" ]]; then
    AMGX_BUILD="$AMGX_ROOT/build"
    export AMGX_BUILD
fi

if [[ ! -f "$AMGX_BUILD/libamgxsh.so" ]]; then
    print_error "AmgX library not found at $AMGX_BUILD/libamgxsh.so"
    echo "Please build AmgX or set AMGX_BUILD correctly."
    exit 1
fi
print_success "AMGX_BUILD: $AMGX_BUILD"

# Step 4: Check MPI if needed
if [[ "$INSTALL_MPI" == true ]]; then
    print_info "Checking MPI installation..."

    # Check for MPI_HOME environment variable first
    if [[ -n "$MPI_HOME" ]]; then
        if [[ -f "$MPI_HOME/bin/mpicc" ]]; then
            export PATH="$MPI_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"
            print_success "Using MPI_HOME: $MPI_HOME"
        else
            print_error "MPI_HOME set to '$MPI_HOME' but mpicc not found at $MPI_HOME/bin/mpicc"
            exit 1
        fi
    fi

    # Check for mpicc (required for building mpi4py/mpi4jax)
    if ! command -v mpicc &> /dev/null; then
        print_error "MPI compiler (mpicc) not found. Please install MPI development packages."
        echo
        echo "On Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev"
        echo "On CentOS/RHEL:   sudo yum install openmpi openmpi-devel"
        echo "With conda:       conda install -c conda-forge openmpi-mpicc"
        echo
        echo "Or set MPI_HOME to your MPI installation directory:"
        echo "  export MPI_HOME=/path/to/mpi"
        exit 1
    fi

    MPI_VERSION=$(mpicc --version 2>&1 | head -n1)
    print_success "MPI: $MPI_VERSION ($(which mpicc))"
    # Note: MPI linkage is auto-detected by setup.py based on AmgX build
fi

# Step 5: Detect CUDA version and install JAX with CUDA support
print_info "Detecting CUDA version..."
CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1/p')

if [[ "$CUDA_VERSION" == "12" ]]; then
    JAX_CUDA_EXTRA="jax[cuda12]>=0.4.35"
    print_success "CUDA 12 detected - will install jax[cuda12]"
elif [[ "$CUDA_VERSION" == "13" ]]; then
    JAX_CUDA_EXTRA="jax[cuda13]>=0.4.35"
    print_success "CUDA 13 detected - will install jax[cuda13]"
else
    print_warning "Unknown CUDA version $CUDA_VERSION, defaulting to cuda12"
    JAX_CUDA_EXTRA="jax[cuda12]>=0.4.35"
fi

# Step 6: Install the package
echo
print_info "Installing JAX-AMG..."

# First install JAX with CUDA support
print_info "Installing JAX with CUDA support..."
pip install "$JAX_CUDA_EXTRA"

# Determine pip install command
PIP_EXTRAS=""
if [[ "$INSTALL_MPI" == true ]]; then
    # Build mpi4py from source to ensure it links against the same MPI as AmgX
    print_info "Building mpi4py from source (for MPI compatibility)..."
    pip install mpi4py --no-binary mpi4py --force-reinstall

    # Install mpi4jax with --no-build-isolation for CUDA compatibility
    # mpi4jax uses CUDA_ROOT (not CUDA_HOME) to find CUDA
    # (per mpi4jax docs: https://github.com/mpi4jax/mpi4jax)
    print_info "Installing mpi4jax (with cython for CUDA backend)..."
    pip install cython
    CUDA_ROOT="$CUDA_HOME" pip install mpi4jax --no-build-isolation --no-cache-dir

    PIP_EXTRAS="[mpi]"
fi
if [[ "$INSTALL_DEV" == true ]]; then
    if [[ -n "$PIP_EXTRAS" ]]; then
        PIP_EXTRAS="[mpi,dev]"
    else
        PIP_EXTRAS="[dev]"
    fi
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run pip install (mpi4py already installed from source, pip will skip it)
cd "$PROJECT_ROOT"
pip install -e ".$PIP_EXTRAS"

# Step 6: Verify installation
export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo
print_info "Verifying installation..."
python3 -c "import jaxamg; print('JAX-AMG imported successfully')" || {
    print_error "Installation verification failed"
    exit 1
}

# Test CUDA availability
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    print_warning "JAX CUDA devices not available"
}

# Success message
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "For runtime, add AmgX and CUDA libraries to your path:"
echo "  export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"

if [[ "$INSTALL_MPI" == true ]]; then
    echo
    echo "For GPU-aware MPI, also set:"
    echo "  export OMPI_MCA_opal_cuda_support=true  # if using OpenMPI"
    echo "  export MPI4JAX_USE_CUDA_MPI=1           # mpi4jax"
fi

echo
echo "Quick test:"
echo "python -c 'from jaxamg import solve; print(\"Ready!\")'"
