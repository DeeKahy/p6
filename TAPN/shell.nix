{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    gcc
    cudaPackages.cudatoolkit
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export CPATH=${pkgs.cudaPackages.cudatoolkit}/include:$CPATH

    # Function to recompile and run TAPN
    function rebuild_tapn() {
      local build_dir="build"
      local start_dir=$(pwd)

      # Find project root (where CMakeLists.txt is)
      while [[ ! -f "CMakeLists.txt" && $(pwd) != "/" ]]; do
        cd ..
      done

      if [[ ! -f "CMakeLists.txt" ]]; then
        echo "Error: Could not find CMakeLists.txt in parent directories"
        cd "$start_dir"
        return 1
      fi

      # Create build directory if it doesn't exist
      if [[ ! -d "$build_dir" ]]; then
        mkdir -p "$build_dir"
      fi

      # Enter build directory and build
      cd "$build_dir"
      echo "Building TAPN..."
      cmake .. && make

      # Run if build was successful
      if [[ $? -eq 0 ]]; then
        echo "Running TAPN..."
        ./TAPN "$@"
      else
        echo "Build failed."
      fi

      # Return to starting directory
      cd "$start_dir"
    }

    echo "TAPN development environment ready."
    echo "Use 'rebuild_tapn' to recompile and run the program."
  '';
}

