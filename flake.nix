{
  description = "folio — section-scoped semantic search over a markdown corpus";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          # One nixpkgs supplies all four, so clippy and rustfmt are the ones
          # built against the rustc that compiled the dependencies. Mixing two
          # installations is what E0514 reports, and the shell is where that
          # stops. mkShell's stdenv carries the C compiler `rusqlite`'s bundled
          # SQLite needs. jujutsu and git are how the work is recorded.
          #
          # No embedding server is declared. folio needs one at runtime and not
          # to build or test, and which of the two measured servers this project
          # settles on is undecided.
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.clippy
            pkgs.rustfmt
            pkgs.jujutsu
            pkgs.git
          ];

          shellHook = ''
            echo >&2 "folio — rustc $(rustc --version | cut -d' ' -f2), cargo $(cargo --version | cut -d' ' -f2)"
            echo >&2 "  cargo test               no endpoint needed"
            echo >&2 "  cargo clippy --all-targets"
          '';
        };
      });
    };
}
