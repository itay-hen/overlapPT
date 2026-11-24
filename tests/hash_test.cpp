// Part of tests/run_small_checks.sh: prints overlap_tables_checksum() for the tables compiled in.
#include "serial_mpi.h"
#include "mainQMC.hpp"
#include "mbar_common.hpp"
#include <cstdio>
int main(){ printf("%016llx\n", (unsigned long long)overlap_tables_checksum()); }
