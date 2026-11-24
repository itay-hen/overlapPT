#!/bin/bash
# Build the overlap-mode binary for one model directory (made by tools/make_model.py).
#
#   ./build_overlap.sh MODEL_DIR [extra compiler flags...]
#
# Example:
#   ./build_overlap.sh models/3x3_v2 -DBeta=4 -DMBAR_N_BETA=16 -DMBAR_T_EQUIL=20000 -DMBAR_N_SAMPLES=20000
#
# Steps: compile prepare.cpp, run it on MODEL_DIR/H1.txt (permutations, cycles),
# generate hamiltonian2.hpp with tools/make_tables.py (H1 and H2 on that list),
# copy the sources into MODEL_DIR/build and compile there, so the model's own
# hamiltonian.hpp is the only one the quoted #include can find.
#
# Compiler: uses mpicxx if available; otherwise g++ with the single-process MPI
# stand-in serial/serial_mpi.h (laptop testing only -- one rank, all rungs in one process).
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
MODEL="$(cd "$1" && pwd)"; shift
BUILD="$MODEL/build"
mkdir -p "$BUILD"

g++ -O3 -o "$BUILD/prepare" "$SRC/prepare.cpp"
(cd "$BUILD" && ./prepare "$MODEL/H1.txt" >/dev/null)
python3 "$SRC/tools/make_tables.py" "$BUILD/hamiltonian.hpp" "$MODEL/H1.txt" "$MODEL/H2.txt" > "$BUILD/hamiltonian2.hpp"
cp "$SRC"/{mbarpt_mpi.cpp,mainQMC.hpp,replica.hpp,divdiff.hpp,mbar_common.hpp,parameters.hpp} "$BUILD/"

source "$SRC/common_flags.sh"
FLAGS=(-O3 "${OVERLAP_BASE_FLAGS[@]}" "$@")
if command -v mpicxx >/dev/null 2>&1; then
    mpicxx "${FLAGS[@]}" -o "$BUILD/overlap.bin" "$BUILD/mbarpt_mpi.cpp"
else
    echo "mpicxx not found: building single-process binary with serial/serial_mpi.h" >&2
    g++ "${FLAGS[@]}" -DSERIAL_MPI -I "$SRC/serial" -o "$BUILD/overlap.bin" "$BUILD/mbarpt_mpi.cpp"
fi
echo "built $BUILD/overlap.bin  (run from a separate directory per pair: overlap.bin 11 | 12 | 22)"
