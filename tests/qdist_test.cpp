// Full walk-length distribution P(q) of the production move kernel at ONE rung s (single replica,
// no PT), to be compared with the exact P(q) of tools/walk_sum_exact.py.
// Compile inside a model build directory with the production flags (see tests/run_qdist.sh).
//   ./qdist_test s n_updates seed  > hist.txt      (prints "q count" lines, then a summary)
#include "serial_mpi.h"   // single-process MPI stand-in (serial/serial_mpi.h)
#include "mainQMC.hpp"
#include <cstdio>
int main(int argc, char** argv){
  if(argc < 4){ fprintf(stderr, "usage: qdist_test s n_updates seed\n"); return 1; }
  divdiff_init(); init_rng();
  double curr=1; factorial[0]=1; for(int k=1;k<qmax;k++){curr*=k; factorial[k]=curr;}
  for(int i=0;i<Ncycles;i++) cycle_len[i]=cycles[i].count();
  cycle_min_len=64; for(int i=0;i<Ncycles;i++) cycle_min_len=std::min(cycle_min_len,cycle_len[i]);
  cycle_max_len=0; for(int i=0;i<Ncycles;i++) cycle_max_len=std::max(cycle_max_len,cycle_len[i]);
  for(int i=0;i<Nop;i++) for(int j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
  const double s = atof(argv[1]); const long nsteps = atol(argv[2]); const unsigned seed = (unsigned)atol(argv[3]);
  RenyiReplica r(2);
  r.init_renyi(s, 0);
  r.local_rng.seed(seed);   // after init_renyi, which reseeds from process entropy
  std::vector<double> h(qmax + 1, 0); double n = 0, sq = 0;
  const long burn = nsteps / 10;
  for(long t = 0; t < nsteps; t++){
    r.update();
    if(t >= burn && t % 5 == 0){ n++; h[r.q]++; sq += r.q; }
  }
  for(int q = 0; q <= qmax; q++) if(h[q] > 0) printf("%d %.0f\n", q, h[q]);
  printf("# s=%.6f n=%.0f mean_q=%.6f\n", s, n, sq / n);
}
