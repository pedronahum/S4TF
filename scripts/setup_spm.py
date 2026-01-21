#!/usr/bin/env python3
"""
Swift Package Manager Setup Script for X10

This script sets up the environment for building Swift for TensorFlow with SPM.
It supports both pre-built PJRT plugins and building from source.

Usage:
    python scripts/setup_spm.py [options]

Options:
    --prefix PATH       Installation prefix (default: ~/.local)
    --platform NAME     Target platform: cpu, cuda, gpu, tpu, rocm (default: cpu)
    --use-prebuilt      Use pre-built PJRT plugins from JAX/TensorFlow
    --build-x10         Build X10 library from source
    --install-deps      Install Python dependencies (JAX/TensorFlow)
    --generate-pc       Generate pkg-config file only
    --help              Show this help message

Examples:
    # Quick setup with pre-built plugins
    python scripts/setup_spm.py --use-prebuilt --platform cpu

    # Setup for CUDA with JAX
    python scripts/setup_spm.py --use-prebuilt --platform cuda --install-deps

    # Generate pkg-config file for custom installation
    python scripts/setup_spm.py --generate-pc --prefix /opt/x10
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, List

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


def run_command(cmd: List[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def find_pjrt_plugin(platform: str) -> Optional[Path]:
    """Find PJRT plugin using the find_pjrt_plugin.py script."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "find_pjrt_plugin.py"), platform],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        # Parse the output to find the plugin path
        for line in result.stdout.split('\n'):
            if line.startswith("Found PJRT plugin"):
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    return Path(parts[1].strip())
    return None


def install_python_deps(platform: str) -> bool:
    """Install Python dependencies for the specified platform."""
    print(f"\n=== Installing Python dependencies for {platform} ===")

    packages = {
        "cpu": ["jax[cpu]"],
        "cuda": ["jax[cuda12]"],
        "gpu": ["jax[cuda12]"],
        "rocm": ["jax[rocm]"],
        "tpu": ["jax[tpu]"],
    }

    pkgs = packages.get(platform, ["jax[cpu]"])
    try:
        run_command([sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs)
        return True
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to install {pkgs}")
        return False


def generate_pkgconfig(
    prefix: Path,
    version: str = "0.13.0",
    extra_libs: str = "",
    extra_cflags: str = ""
) -> Path:
    """Generate pkg-config file for X10."""
    print(f"\n=== Generating pkg-config file ===")

    # Read template
    template_path = SCRIPT_DIR / "x10.pc.in"
    if not template_path.exists():
        # Create inline if template doesn't exist
        template = """# pkg-config file for X10 library
prefix=@PREFIX@
exec_prefix=${prefix}
libdir=${prefix}/lib
includedir=${prefix}/include

Name: x10
Description: X10 Tensor Library with OpenXLA/PJRT backend
Version: @VERSION@
URL: https://github.com/tensorflow/swift-apis

Libs: -L${libdir} -lx10 @EXTRA_LIBS@
Libs.private: -ldl -lpthread -lstdc++
Cflags: -I${includedir} -I${includedir}/x10 @EXTRA_CFLAGS@
"""
    else:
        template = template_path.read_text()

    # Replace placeholders
    content = template.replace("@PREFIX@", str(prefix))
    content = content.replace("@VERSION@", version)
    content = content.replace("@EXTRA_LIBS@", extra_libs)
    content = content.replace("@EXTRA_CFLAGS@", extra_cflags)

    # Write pkg-config file
    pkgconfig_dir = prefix / "lib" / "pkgconfig"
    pkgconfig_dir.mkdir(parents=True, exist_ok=True)
    pc_file = pkgconfig_dir / "x10.pc"
    pc_file.write_text(content)

    print(f"  Generated: {pc_file}")
    return pc_file


def setup_prebuilt(prefix: Path, platform: str) -> bool:
    """Setup using pre-built PJRT plugins."""
    print(f"\n=== Setting up with pre-built PJRT plugins ===")

    # Find PJRT plugin
    plugin_path = find_pjrt_plugin(platform)
    if not plugin_path:
        print(f"Error: No PJRT plugin found for platform '{platform}'")
        print(f"Install JAX or TensorFlow first: pip install jax[{platform}]")
        return False

    print(f"  Found PJRT plugin: {plugin_path}")

    # Create directories
    lib_dir = prefix / "lib"
    include_dir = prefix / "include" / "x10"
    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link headers
    cx10_dir = PROJECT_ROOT / "Sources" / "CX10"
    headers = ["device_wrapper.h", "xla_tensor_wrapper.h", "xla_tensor_tf_ops.h"]
    for header in headers:
        src = cx10_dir / header
        dst = include_dir / header
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied: {header}")

    # Create a stub library that loads PJRT dynamically
    # For now, we create a symlink to indicate the plugin path
    plugin_link = lib_dir / "pjrt_plugin.so"
    if plugin_link.exists():
        plugin_link.unlink()
    plugin_link.symlink_to(plugin_path)
    print(f"  Linked plugin: {plugin_link} -> {plugin_path}")

    # Generate pkg-config
    extra_libs = f"-Wl,-rpath,{plugin_path.parent}"
    generate_pkgconfig(prefix, extra_libs=extra_libs)

    return True


def build_x10(prefix: Path, platform: str) -> bool:
    """Build X10 from source using Bazel."""
    print(f"\n=== Building X10 from source ===")

    # Check for Bazel
    if not shutil.which("bazel") and not shutil.which("bazelisk"):
        print("Error: Bazel not found. Install Bazel or Bazelisk first.")
        return False

    bazel = shutil.which("bazelisk") or shutil.which("bazel")

    # Determine build config
    configs = ["--config=openxla"]
    if platform in ("cuda", "gpu"):
        configs.append("--config=cuda")
    elif platform == "tpu":
        configs.append("--config=tpu")
    elif platform == "rocm":
        configs.append("--config=rocm")

    # Build
    try:
        os.chdir(PROJECT_ROOT)
        run_command([bazel, "build"] + configs + ["//xla_tensor:x10"])

        # Install
        bazel_bin = PROJECT_ROOT / "bazel-bin"
        lib_dir = prefix / "lib"
        include_dir = prefix / "include" / "x10"
        lib_dir.mkdir(parents=True, exist_ok=True)
        include_dir.mkdir(parents=True, exist_ok=True)

        # Copy library
        for lib_name in ["libx10.so", "libx10.dylib"]:
            lib_src = bazel_bin / "tensorflow" / "compiler" / "tf2xla" / "xla_tensor" / lib_name
            if lib_src.exists():
                shutil.copy2(lib_src, lib_dir / lib_name)
                print(f"  Installed: {lib_name}")

        # Copy headers
        cx10_dir = PROJECT_ROOT / "Sources" / "CX10"
        for header in cx10_dir.glob("*.h"):
            shutil.copy2(header, include_dir / header.name)
            print(f"  Copied: {header.name}")

        # Generate pkg-config
        generate_pkgconfig(prefix)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Build failed: {e}")
        return False


def print_setup_instructions(prefix: Path, platform: str):
    """Print post-setup instructions."""
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)

    env_vars = {
        "X10_LIBRARY_PATH": str(prefix / "lib"),
        "X10_INCLUDE_PATH": str(prefix / "include"),
        "XLA_PLATFORM": platform,
        "PKG_CONFIG_PATH": str(prefix / "lib" / "pkgconfig") + ":$PKG_CONFIG_PATH",
        "LD_LIBRARY_PATH": str(prefix / "lib") + ":$LD_LIBRARY_PATH",
    }

    print("\nAdd these to your shell profile (~/.bashrc or ~/.zshrc):\n")
    for var, value in env_vars.items():
        print(f'export {var}="{value}"')

    print("\nThen build with Swift Package Manager:")
    print("  swift build")
    print("\nOr with explicit paths:")
    print(f"  swift build -Xcc -I{prefix}/include -Xlinker -L{prefix}/lib")

    # Create a setup script
    setup_script = prefix / "setup_env.sh"
    with open(setup_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Source this file: source setup_env.sh\n\n")
        for var, value in env_vars.items():
            f.write(f'export {var}="{value}"\n')
    setup_script.chmod(0o755)
    print(f"\nOr source the generated script:")
    print(f"  source {setup_script}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Swift Package Manager for X10 with OpenXLA/PJRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path.home() / ".local",
        help="Installation prefix (default: ~/.local)"
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "cuda", "gpu", "tpu", "rocm"],
        default="cpu",
        help="Target platform (default: cpu)"
    )
    parser.add_argument(
        "--use-prebuilt",
        action="store_true",
        help="Use pre-built PJRT plugins from JAX/TensorFlow"
    )
    parser.add_argument(
        "--build-x10",
        action="store_true",
        help="Build X10 library from source"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies (JAX/TensorFlow)"
    )
    parser.add_argument(
        "--generate-pc",
        action="store_true",
        help="Generate pkg-config file only"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("X10 Swift Package Manager Setup")
    print("=" * 60)
    print(f"  Prefix: {args.prefix}")
    print(f"  Platform: {args.platform}")

    # Generate pkg-config only
    if args.generate_pc:
        generate_pkgconfig(args.prefix)
        print("\nDone!")
        return 0

    # Install dependencies if requested
    if args.install_deps:
        install_python_deps(args.platform)

    # Setup based on mode
    success = False
    if args.use_prebuilt:
        success = setup_prebuilt(args.prefix, args.platform)
    elif args.build_x10:
        success = build_x10(args.prefix, args.platform)
    else:
        # Default: try pre-built first, then offer to build
        success = setup_prebuilt(args.prefix, args.platform)
        if not success:
            print("\nPre-built plugins not available.")
            print("Run with --install-deps to install JAX, or --build-x10 to build from source.")
            return 1

    if success:
        print_setup_instructions(args.prefix, args.platform)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
