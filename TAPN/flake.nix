{
  description = "PNML Parser";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        packages = rec {
          pnml-parser = pkgs.stdenv.mkDerivation {
            pname = "pnml-parser";
            version = "0.1.0";

            src = ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
            ];

            buildInputs = with pkgs; [
              pugixml
            ];

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
            ];

            installPhase = ''
              mkdir -p $out/bin
              cp pnml_parser $out/bin/
            '';
          };

          default = pnml-parser;
        };

        apps = rec {
          pnml-parser = flake-utils.lib.mkApp {
            drv = self.packages.${system}.pnml-parser;
            name = "pnml-parser";
          };
          default = pnml-parser;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            pugixml
            gcc
          ];
        };
      }
    );
}