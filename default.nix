{pkgs ? import <nixpkgs> {config.allowUnfree = true;}}: let
  cuda128 = pkgs.stdenv.mkDerivation rec {
    name = "cudatoolkit-12.8.0";
    version = "12.8.0";
    src = ./cuda_12.8.0_570.86.10_linux.run;
    nativeBuildInputs = [pkgs.autoPatchelfHook pkgs.makeWrapper pkgs.coreutils pkgs.bash];
    buildInputs = [
      pkgs.stdenv.cc.cc.lib
      pkgs.libxml2
      pkgs.cudaPackages.cuda_cupti
      pkgs.rdma-core
    ];
    autoPatchelfIgnoreMissingDeps = [
      "libmlx5.so.1"
      "libcuda.so.1"
      "libnvidia-ml.so.1"
      "libwayland-client.so.0"
      "libwayland-cursor.so.0"
      "libxkbcommon.so.0"
      "libGLX.so.0"
      "libOpenGL.so.0"
      "libQt6WlShellIntegration.so.6"
    ];
    unpackPhase = ''
      echo "Unpacking Makeself CUDA 12.8.0 archive from $src"
      cp $src cuda.run
      chmod +x cuda.run
      mkdir -p $out
      ./cuda.run --tar xvf -C $out || {
        echo "Warning: --tar xvf failed or incomplete, attempting full extraction"
        mkdir -p tmp
        ./cuda.run --silent --extract=tmp
        mv tmp/* $out/
        rm -rf tmp
      }

      # Create standard CUDA layout
      mkdir -p $out/bin $out/lib $out/include $out/nvvm/bin $out/targets/x86_64-linux/lib $out/targets/x86_64-linux/include

      # Move CUDA binaries
      mv $out/builds/cuda_nvcc/bin/* $out/bin/ 2>/dev/null || true

      # Collect libraries from all relevant subdirs
      for dir in $out/builds/*/lib $out/builds/*/lib64 $out/builds/*/targets/x86_64-linux/lib; do
        if [ -d "$dir" ]; then
          mv "$dir"/* $out/targets/x86_64-linux/lib/ 2>/dev/null || true
        fi
      done

      # Collect headers from all relevant subdirs
      for dir in $out/builds/*/include $out/builds/*/targets/x86_64-linux/include; do
        if [ -d "$dir" ]; then
          mv "$dir"/* $out/targets/x86_64-linux/include/ 2>/dev/null || true
        fi
      done

      # Move NVVM files (ensure cicc is included)
      if [ -d "$out/builds/cuda_nvcc/nvvm/bin" ]; then
        mv $out/builds/cuda_nvcc/nvvm/bin/* $out/nvvm/bin/ 2>/dev/null || true
      fi
      mv $out/builds/cuda_nvcc/nvvm/* $out/nvvm/ 2>/dev/null || true

      # Fix symlinks
      ln -sf $out/targets/x86_64-linux/lib $out/lib/lib64 2>/dev/null || true
      ln -sf $out/targets/x86_64-linux/include $out/include/include 2>/dev/null || true

      # Remove problematic nested symlinks
      rm -rf $out/targets/x86_64-linux/include/include 2>/dev/null || true
      rm -rf $out/targets/x86_64-linux/lib/lib64 2>/dev/null || true

      # Clean up
      rm -rf $out/builds
    '';
    installPhase = ''
      echo "Installing CUDA 12.8.0"
      for bin in $out/bin/*; do
        if [ -f "$bin" ] && [ -x "$bin" ]; then
          wrapProgram "$bin" --prefix LD_LIBRARY_PATH : "$out/targets/x86_64-linux/lib:/run/opengl-driver/lib"
        fi
      done
    '';
    postFixup = ''
      echo "Patching libraries:"
      for lib in $out/targets/x86_64-linux/lib/*.so; do
        patchelf --set-rpath "$out/targets/x86_64-linux/lib:/run/opengl-driver/lib" $lib 2>/dev/null || true
      done
      for lib in $out/nvvm/lib64/*.so; do
        patchelf --set-rpath "$out/targets/x86_64-linux/lib:$out/nvvm/lib64:/run/opengl-driver/lib" $lib 2>/dev/null || true
      done
    '';
  };
in
  pkgs.mkShell {
    buildInputs = [
      cuda128
      pkgs.cudaPackages.cuda_gdb # Explicitly added cuda-gdb if the run file does not include it or it does not work
    ];
    shellHook = ''
      export CUDA_PATH=${cuda128}
      export LD_LIBRARY_PATH=/run/opengl-driver/lib:${cuda128}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
    '';
  }
