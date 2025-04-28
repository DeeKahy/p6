{ pkgs ? import <nixpkgs> {} }:

let
in
pkgs.mkShell {
  shellHook = ''
  
    run() {
      root="$(realpath TAPN)"
      CURRENT_PATH="$(pwd)"
      if [ ! -d "$root/build" ]; then
        cd "$root"
        mkdir build
      fi
      cd "$root/build"
      cmake ..
      make
      ./TAPN
      cd "$CURRENT_PATH"
    }
    cmakeInstall(){
      sudo snap install cmake --classic
    }
  '';
}
