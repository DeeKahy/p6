{ pkgs ? import <nixpkgs> { system = "x86_64-darwin"; } }:

let
  pkgsLinux = pkgs.pkgsCross.musl64; # Cross-compile for Linux
in
pkgsLinux.dockerTools.buildImage {
  name = "cpp-dev-env";
  tag = "latest";
  contents = pkgsLinux.buildEnv {
    name = "cpp-dev-env";
    paths = with pkgsLinux; [
      clang-tools
      cmake
      codespell
      conan
      cppcheck
      doxygen
      gtest
      lcov
      vcpkg
      vcpkg-tool
      cudatoolkit
      copycat
      git
      gh
    ] ++ (if pkgsLinux.stdenv.hostPlatform.system == "aarch64-darwin" then [ ] else [ gdb ]);
  };
  config = {
    Cmd = [ "bash" ];
  };
}
