// Part of tests/run_small_checks.sh -- compiled inside a model build directory (see that script).
// Brute-force check of RenyiReplica::OvWeight: sum over y0 and all walks of length <= QMAX
// should converge to dim * O(lam_a, lam_b) for |+>  (endpoint factor 1/dim dropped).
#include "serial_mpi.h"   // single-process MPI stand-in (serial/serial_mpi.h)
#include "mainQMC.hpp"
#include <cstdio>
int main(int argc, char** argv){
  divdiff_init(); init_rng();
  double curr=1; factorial[0]=1; for(int k=1;k<qmax;k++){curr*=k; factorial[k]=curr;}
  int QMAX = atoi(argv[1]); double s = atof(argv[2]);
  RenyiReplica r(2);
  r.init_renyi(s, 0);
  double total=0;
  for(int y=0;y<(1<<N);y++){
    std::vector<int> seq;
    // iterate all sequences of length q via odometer
    for(int q=0;q<=QMAX;q++){
      std::vector<int> w(q,0);
      while(true){
        r.lattice[0]=std::bitset<N>(y); r.lattice[1]=r.lattice[0]; r.q=q;
        for(int i=0;i<q;i++) r.Sq[i]=w[i];
        r.InvalidateCache();
        total += r.GetRenyiWeight(s, r.beta_pow_factorial).get_double();
        int i=q-1; while(i>=0 && w[i]==Nop-1){w[i]=0;i--;} if(i<0) break; w[i]++;
      }
    }
  }
  printf("QMAX=%d s=%.3f  sum=%.10f  ln(sum/dim)=%.10f\n", QMAX, s, total, log(total/(1<<N)));
}
