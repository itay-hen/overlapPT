// Part of tests/run_small_checks.sh -- compiled inside a model build directory (see that script).
// Exact P(q) at one rung by enumeration (q <= QE) vs. the single-replica sampler (no PT).
#include "serial_mpi.h"   // single-process MPI stand-in (serial/serial_mpi.h)
#include "mainQMC.hpp"
#include <cstdio>
int main(int argc, char** argv){
  divdiff_init(); init_rng();
  double curr=1; factorial[0]=1; for(int k=1;k<qmax;k++){curr*=k; factorial[k]=curr;}
  for(int i=0;i<Ncycles;i++) cycle_len[i]=cycles[i].count();
  cycle_min_len=64; for(int i=0;i<Ncycles;i++) cycle_min_len=std::min(cycle_min_len,cycle_len[i]);
  cycle_max_len=0; for(int i=0;i<Ncycles;i++) cycle_max_len=std::max(cycle_max_len,cycle_len[i]);
  for(int i=0;i<Nop;i++) for(int j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
  double s=atof(argv[1]); double Ztot=atof(argv[2]); int QE=atoi(argv[3]); long nsteps=atol(argv[4]);
  RenyiReplica r(2); r.init_renyi(s,0);
  std::vector<double> ex(QE+1,0);
  for(int y=0;y<(1<<N);y++) for(int q=0;q<=QE;q++){ std::vector<int> w(q,0);
    while(true){ r.lattice[0]=std::bitset<N>(y); r.lattice[1]=r.lattice[0]; r.q=q; for(int i=0;i<q;i++) r.Sq[i]=w[i];
      r.InvalidateCache(); ex[q]+=r.GetRenyiWeight(s,r.beta_pow_factorial).get_double();
      int i=q-1; while(i>=0&&w[i]==Nop-1){w[i]=0;i--;} if(i<0)break; w[i]++; } }
  r.init_renyi(s,0);
  std::vector<double> h(QE+2,0), hy(1<<N,0); double n=0;
  for(long t=0;t<nsteps;t++){ r.update(); if(t>nsteps/10 && t%5==0){ n++; hy[r.lattice[0].to_ulong()]++; if(r.q<=QE) h[r.q]++; else h[QE+1]++; } }
  double se=0, ss=0;
  for(int q=0;q<=QE;q++){ double pe=ex[q]/Ztot, ps=h[q]/n; se+=pe; ss+=ps; printf("q=%d exact=%.5f sampled=%.5f\n",q,pe,ps); }
  printf("sum<=%d exact=%.5f sampled=%.5f\n",QE,se,ss);
  for(int y=0;y<(1<<N);y++) printf("y0=%d sampled=%.6f\n",y,hy[y]/n);
}
