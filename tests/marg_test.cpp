// Single-replica chain (no PT) at one rung s of the OVERLAP_PLUS ladder, any model size.
// Prints, per block, the mean of q and q^2 and the histogram of the walk start y0, so that
// tools/plus_observables.py's exact <q>, var(q) and y0 marginal can be compared with block errors.
// Build like tests/hist_test.cpp, inside a model build directory.
//   ./marg_test S NSTEPS NBLOCKS SEED [nohist]
// Also prints sup = fraction of samples in the support of the last rung (W(s=1) > 0, i.e. no
// severed-bond hop), to compare with tools/plus_support.py.
#include "serial_mpi.h"
#include "mainQMC.hpp"
#include <cstdio>
#include <vector>
#include <string>
int main(int argc, char** argv){
  divdiff_init(); init_rng();
  double curr=1; factorial[0]=1; for(int k=1;k<qmax;k++){curr*=k; factorial[k]=curr;}
  for(int i=0;i<Ncycles;i++) cycle_len[i]=cycles[i].count();
  cycle_min_len=64; for(int i=0;i<Ncycles;i++) cycle_min_len=std::min(cycle_min_len,cycle_len[i]);
  cycle_max_len=0; for(int i=0;i<Ncycles;i++) cycle_max_len=std::max(cycle_max_len,cycle_len[i]);
  for(int i=0;i<Nop;i++) for(int j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
  double s=atof(argv[1]); long nsteps=atol(argv[2]); int nb=atoi(argv[3]); unsigned seed=atoi(argv[4]); bool hist = !(argc > 5 && std::string(argv[5]) == "nohist");
  RenyiReplica r(2); r.init_renyi(s,0); r.local_rng.seed(seed);
  long equil=nsteps/10, per=(nsteps-equil)/nb;
  for(long t=0;t<equil;t++) r.update();
  for(int b=0;b<nb;b++){
    std::vector<double> hy(1<<N,0); double n=0,sq=0,sq2=0,sup=0;
    for(long t=0;t<per;t++){ r.update(); if(t%5==0){ n++; sq+=r.q; sq2+=(double)r.q*r.q; hy[r.lattice[0].to_ulong()]++;
      ExExFloat w1=r.OvWeight(1.0); w1.normalize(); if(w1.get_mantissa()>0) sup++; } }
    printf("block %d q=%.6f q2=%.6f sup=%.6f\n",b,sq/n,sq2/n,sup/n);
    if(!hist) continue;
    printf("hist %d",b); for(int y=0;y<(1<<N);y++) printf(" %.0f",hy[y]); printf("\n");
  }
}
