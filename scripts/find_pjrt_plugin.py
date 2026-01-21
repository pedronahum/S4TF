#!/usr/bin/env python3
"""
Find PJRT Plugin Utility

This script searches for pre-built PJRT plugins from Python packages like JAX
and TensorFlow. It helps configure Swift for TensorFlow (X10) to use
pre-existing PJRT libraries instead of building from source.

Usage:
    python find_pjrt_plugin.py [platform]

Examples:
    python find_pjrt_plugin.py cpu
    python find_pjrt_plugin.py cuda
    python find_pjrt_plugin.py gpu
    python find_pjrt_plugin.py tpu
"""

import os
import sys
import site
import glob
from pathlib import Path
from typing import List, Optional, Dict

# Known PJRT plugin library names for different backends
PLUGIN_REGISTRY: Dict[str, Dict] = {
    "cpu": {
        "env_var": "PJRT_CPU_LIBRARY_PATH",
        "lib_names": [
            "pjrt_c_api_cpu_plugin.so",
            "libpjrt_c_api_cpu.so",
            "pjrt_plugin_xla_cpu.so",
            "xla_cpu_plugin.so",
        ],
        "package_hint": "pip install jax[cpu] or pip install tensorflow",
    },
    "cuda": {
        "env_var": "PJRT_CUDA_LIBRARY_PATH",
        "lib_names": [
            "pjrt_c_api_gpu_plugin.so",
            "libpjrt_c_api_gpu.so",
            "pjrt_plugin_xla_cuda.so",
            "libxla_cuda.so",
            "xla_cuda_plugin.so",
        ],
        "package_hint": "pip install jax[cuda12] or pip install tensorflow[and-cuda]",
    },
    "gpu": {
        "env_var": "PJRT_GPU_LIBRARY_PATH",
        "lib_names": [
            "pjrt_c_api_gpu_plugin.so",
            "libpjrt_c_api_gpu.so",
            "pjrt_plugin_xla_cuda.so",
            "libxla_cuda.so",
            "xla_cuda_plugin.so",
        ],
        "package_hint": "pip install jax[cuda12] or pip install tensorflow[and-cuda]",
    },
    "tpu": {
        "env_var": "PJRT_TPU_LIBRARY_PATH",
        "lib_names": [
            "libtpu.so",
            "pjrt_c_api_tpu_plugin.so",
        ],
        "package_hint": "pip install jax[tpu] (on Google Cloud TPU)",
    },
    "rocm": {
        "env_var": "PJRT_ROCM_LIBRARY_PATH",
        "lib_names": [
            "pjrt_c_api_rocm_plugin.so",
            "libpjrt_c_api_rocm.so",
            "xla_rocm_plugin.so",
        ],
        "package_hint": "pip install jax[rocm]",
    },
}

# Directories to search within site-packages
SEARCH_SUBDIRS = [
    "",
    "jaxlib",
    "jaxlib/xla_extension",
    "jax_plugins",
    "jax_plugins/xla_cpu",
    "jax_plugins/xla_cuda",
    "jax_plugins/xla_rocm",
    "jax_plugins/xla_tpu",
    "tensorflow",
    "tensorflow/compiler/tf2xla/python",
    "tensorflow/python/_pywrap_tfe",
    "xla",
    "xla/pjrt",
    "xla_extension",
]


def get_site_packages() -> List[Path]:
    """Get all Python site-packages directories."""
    paths = []

    # Get site-packages from site module
    try:
        paths.extend(Path(p) for p in site.getsitepackages())
    except AttributeError:
        pass  # Not available in all Python environments

    # Add user site-packages
    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.append(Path(user_site))
    except AttributeError:
        pass

    # Check CONDA_PREFIX
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for py_version in ["3.10", "3.11", "3.12", "3.13"]:
            conda_site = Path(conda_prefix) / "lib" / f"python{py_version}" / "site-packages"
            if conda_site.exists():
                paths.append(conda_site)

    # Check VIRTUAL_ENV
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        for py_version in ["3.10", "3.11", "3.12", "3.13"]:
            venv_site = Path(virtual_env) / "lib" / f"python{py_version}" / "site-packages"
            if venv_site.exists():
                paths.append(venv_site)

    return [p for p in paths if p.exists()]


def find_pjrt_plugin(platform: str) -> Optional[Path]:
    """Find a PJRT plugin for the given platform."""
    if platform not in PLUGIN_REGISTRY:
        print(f"Unknown platform: {platform}")
        print(f"Supported platforms: {', '.join(PLUGIN_REGISTRY.keys())}")
        return None

    info = PLUGIN_REGISTRY[platform]

    # Check environment variable override first
    env_path = os.environ.get(info["env_var"])
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check generic override
    generic_path = os.environ.get("PJRT_PLUGIN_LIBRARY_PATH")
    if generic_path and Path(generic_path).exists():
        return Path(generic_path)

    # Search in site-packages
    site_packages = get_site_packages()
    found_plugins = []

    for site_pkg in site_packages:
        for subdir in SEARCH_SUBDIRS:
            search_dir = site_pkg / subdir if subdir else site_pkg
            if not search_dir.exists():
                continue

            for lib_name in info["lib_names"]:
                plugin_path = search_dir / lib_name
                if plugin_path.exists():
                    found_plugins.append(plugin_path)

                # Also try glob patterns
                for match in search_dir.glob(f"*{lib_name}*"):
                    if match.is_file() and match.suffix == ".so":
                        found_plugins.append(match)

    # Search in LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for dir_path in ld_path.split(":"):
        if not dir_path:
            continue
        dir_path = Path(dir_path)
        if not dir_path.exists():
            continue
        for lib_name in info["lib_names"]:
            plugin_path = dir_path / lib_name
            if plugin_path.exists():
                found_plugins.append(plugin_path)

    # Remove duplicates while preserving order
    seen = set()
    unique_plugins = []
    for p in found_plugins:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_plugins.append(p)

    return unique_plugins[0] if unique_plugins else None


def search_all_plugins() -> Dict[str, List[Path]]:
    """Search for all PJRT plugins on the system."""
    results = {}

    site_packages = get_site_packages()
    print(f"Searching in {len(site_packages)} site-packages directories...")

    for platform, info in PLUGIN_REGISTRY.items():
        found = []

        # Check environment variable
        env_path = os.environ.get(info["env_var"])
        if env_path and Path(env_path).exists():
            found.append(("env:" + info["env_var"], Path(env_path)))

        # Search site-packages
        for site_pkg in site_packages:
            for subdir in SEARCH_SUBDIRS:
                search_dir = site_pkg / subdir if subdir else site_pkg
                if not search_dir.exists():
                    continue

                for lib_name in info["lib_names"]:
                    plugin_path = search_dir / lib_name
                    if plugin_path.exists():
                        found.append((str(search_dir), plugin_path))

        results[platform] = found

    return results


def print_export_commands(platform: str, plugin_path: Path) -> None:
    """Print shell commands to set up the environment."""
    info = PLUGIN_REGISTRY[platform]
    print("\n# Add these to your shell profile (~/.bashrc or ~/.zshrc):")
    print(f"export {info['env_var']}=\"{plugin_path}\"")
    print(f"export XLA_PLATFORM=\"{platform}\"")

    # Add library directory to LD_LIBRARY_PATH
    lib_dir = plugin_path.parent
    print(f"export LD_LIBRARY_PATH=\"{lib_dir}:$LD_LIBRARY_PATH\"")


def main():
    if len(sys.argv) < 2:
        # Search for all plugins
        print("PJRT Plugin Discovery Tool")
        print("=" * 60)
        print()

        results = search_all_plugins()

        found_any = False
        for platform, plugins in results.items():
            if plugins:
                found_any = True
                print(f"\n{platform.upper()} plugins found:")
                for source, path in plugins:
                    print(f"  - {path}")
                    print(f"    (from: {source})")
            else:
                print(f"\n{platform.upper()}: No plugins found")
                print(f"  Install: {PLUGIN_REGISTRY[platform]['package_hint']}")

        if not found_any:
            print("\nNo PJRT plugins found. Install JAX or TensorFlow:")
            print("  pip install jax[cpu]        # For CPU")
            print("  pip install jax[cuda12]     # For CUDA GPU")
            print("  pip install tensorflow      # TensorFlow (includes XLA)")

        return

    platform = sys.argv[1].lower()

    if platform == "--help" or platform == "-h":
        print(__doc__)
        return

    if platform == "--all":
        results = search_all_plugins()
        for platform, plugins in results.items():
            if plugins:
                for source, path in plugins:
                    print(f"{platform}: {path}")
        return

    plugin_path = find_pjrt_plugin(platform)

    if plugin_path:
        print(f"Found PJRT plugin for {platform}: {plugin_path}")
        print_export_commands(platform, plugin_path)
    else:
        info = PLUGIN_REGISTRY.get(platform, {})
        print(f"No PJRT plugin found for {platform}")
        print(f"\nTo install, try: {info.get('package_hint', 'pip install jax or tensorflow')}")
        print(f"\nOr manually set: export {info.get('env_var', 'PJRT_PLUGIN_LIBRARY_PATH')}=/path/to/plugin.so")
        sys.exit(1)


if __name__ == "__main__":
    main()
