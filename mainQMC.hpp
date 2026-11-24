//
// This program implements Permutation Matrix Representation Quantum Monte Carlo for arbitrary spin-1/2 Hamiltonians.
//
// This program is introduced in the paper:
// Lev Barash, Arman Babakhani, Itay Hen, A quantum Monte Carlo algorithm for arbitrary spin-1/2 Hamiltonians, Physical Review Research 6, 013281 (2024).
//
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//
// ExExFloat datatype and calculation of divided differences are described in the paper:
// L. Gupta, L. Barash, I. Hen, Calculating the divided differences of the exponential function by addition and removal of inputs, Computer Physics Communications 254, 107385 (2020)
//

#include<iostream>
#include<fstream>
#include<iomanip>
#include<complex>
#include<random>
#include<cstdlib>
#include<algorithm>
#include<csignal>
#include<bitset>
#include<unordered_map>
#include"divdiff.hpp"
#include"hamiltonian.hpp" // use a header file, which defines the Hamiltonian and the custom observables
#include"parameters.hpp"  // parameters of the simulation such as the number of Monte-Carlo updates
#ifdef OVERLAP_MODE
#include"hamiltonian2.hpp" // [OVERLAP] H1 and H2 tables on the shared P_matrix (tools/make_tables.py)
// [OVERLAP] Hamiltonian index (0 = H1, 1 = H2) of copy 0 (acts first, segment [0,c))
// and copy 1 (segment [c,q)). Set by the driver before any RenyiReplica is built.
int g_ov_ham[2] = {0, 1};
#endif

#define measurements (steps/stepsPerMeasurement)

#ifdef EXHAUSTIVE_CYCLE_SEARCH
	#define rmin 0                            // length r of sub-sequence is chosen randomly between rmin and rmax
	#define rmax cycle_max_len
	#define lmin r                            // cycle lengths must be between lmin and lmax
	#define lmax cycle_max_len
#else
	#define rmin (cycle_min_len-1)/2
	#define rmax (cycle_max_len+1)/2
	#define lmin 2*r-1
	#define lmax 2*r+1
#endif

static std::random_device rd;
static std::mt19937 rng;
static std::uniform_int_distribution<> dice2(0,1);
static std::uniform_int_distribution<> diceN(0,N-1);
static std::uniform_int_distribution<> diceNop(0,Nop-1);
static std::uniform_real_distribution<> val(0.0,1.0);
static std::geometric_distribution<> geometric_int(GAPS_GEOMETRIC_PARAMETER);

double factorial[qmax]; // contains the values q!
int cycle_len[Ncycles];
int cycle_min_len, cycle_max_len;

#ifndef MEASURE_CUSTOM_OBSERVABLES
#define Nobservables 0
#endif

const int N_all_observables = Nobservables + 9; //gabe
int valid_observable[N_all_observables];

unsigned long long bin_length = measurements / Nbins, bin_length_old;
double in_bin_sum[N_all_observables];
double bin_mean[N_all_observables][Nbins];
double in_bin_sum_sgn;
double bin_mean_sgn[Nbins];

std::bitset<Ncycles> P_in_cycles[Nop];

#ifdef ABS_WEIGHTS
#define REALW    std::abs
#else
#define REALW    std::real
#endif

ExExFloat zero; int TstepsFinished = 0;
unsigned long long step = 0;
unsigned long long measurement_step = 0;

unsigned int rng_seed;
double meanq = 0;
double maxq = 0;
double start_time;
int save_data_flag = 0, mpi_rank = 0, mpi_size = 0, resume_calc = 0;

void init_rng(){
#ifdef EXACTLY_REPRODUCIBLE
	rng_seed = 1000 + mpi_rank;
#else
	rng_seed = rd();
#endif
	rng.seed(rng_seed);
}

#include"replica.hpp"

Replica replica;

void init_basic(){
	double curr=1; factorial[0] = curr; zero -= zero;
	for(int k=1;k<qmax;k++){ curr*=k; factorial[k] = curr;}
	replica.set_beta(0.00787402, 0.015748);
}

void init(){
	int i,j; double curr=1; factorial[0] = curr; zero -= zero;
	for(int k=1;k<qmax;k++){ curr*=k; factorial[k] = curr;}
	replica.init(0.00787402, 0.015748);
	for(i=0;i<Ncycles;i++) cycle_len[i] = cycles[i].count();
	cycle_min_len = 64; for(i=0;i<Ncycles;i++) cycle_min_len = min(cycle_min_len,cycle_len[i]);
	cycle_max_len = 0; for(i=0;i<Ncycles;i++) cycle_max_len = max(cycle_max_len,cycle_len[i]);
	for(i=0;i<Nop;i++) for(j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
	for(i=0;i<N_all_observables;i++) in_bin_sum[i] = 0; in_bin_sum_sgn = 0;
	for(i=0;i<N_all_observables;i++) valid_observable[i] = 0;
#ifdef MEASURE_CUSTOM_OBSERVABLES
	for(i=0;i<Nobservables;i++) valid_observable[i] = 1;
#endif
#ifdef MEASURE_H
	valid_observable[Nobservables] = 1;
#endif
#ifdef MEASURE_H2
	valid_observable[Nobservables + 1] = 1;
#endif
#ifdef MEASURE_HDIAG
	valid_observable[Nobservables + 2] = 1;
#endif
#ifdef MEASURE_HDIAG2
	valid_observable[Nobservables + 3] = 1;
#endif
#ifdef MEASURE_HOFFDIAG
	valid_observable[Nobservables + 4] = 1;
#endif
#ifdef MEASURE_HOFFDIAG2
	valid_observable[Nobservables + 5] = 1;
#endif
#ifdef MEASURE_Z_MAGNETIZATION
	valid_observable[Nobservables + 6] = 1;
#endif
#ifdef MEASURE_RATIO 
	valid_observable[Nobservables + 8] = 1; //gabe
#endif
}

void init2(double beta_smaller, double beta_bigger){
	int i,j; double curr=1; factorial[0] = curr; zero -= zero;
	//gabe 
	replica.m_step = 0; replica.stp = 0; replica.TstepsFin = 0;
	for(int k=1;k<qmax;k++){ curr*=k; factorial[k] = curr;}
	replica.init(beta_smaller, beta_bigger);
	for(i=0;i<Ncycles;i++) cycle_len[i] = cycles[i].count();
	cycle_min_len = 64; for(i=0;i<Ncycles;i++) cycle_min_len = min(cycle_min_len,cycle_len[i]);
	cycle_max_len = 0; for(i=0;i<Ncycles;i++) cycle_max_len = max(cycle_max_len,cycle_len[i]);
	for(i=0;i<Nop;i++) for(j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
	for(i=0;i<N_all_observables;i++) in_bin_sum[i] = 0; in_bin_sum_sgn = 0;
	for(i=0;i<N_all_observables;i++) valid_observable[i] = 0;
#ifdef MEASURE_CUSTOM_OBSERVABLES
	for(i=0;i<Nobservables;i++) valid_observable[i] = 1;
#endif
#ifdef MEASURE_H
	valid_observable[Nobservables] = 1;
#endif
#ifdef MEASURE_H2
	valid_observable[Nobservables + 1] = 1;
#endif
#ifdef MEASURE_HDIAG
	valid_observable[Nobservables + 2] = 1;
#endif
#ifdef MEASURE_HDIAG2
	valid_observable[Nobservables + 3] = 1;
#endif
#ifdef MEASURE_HOFFDIAG
	valid_observable[Nobservables + 4] = 1;
#endif
#ifdef MEASURE_HOFFDIAG2
	valid_observable[Nobservables + 5] = 1;
#endif
#ifdef MEASURE_Z_MAGNETIZATION
	valid_observable[Nobservables + 6] = 1;
#endif
#ifdef MEASURE_RATIO 
	valid_observable[Nobservables + 8] = 1; //gabe
#endif
}

std::string name_of_observable(int n){
	std::string s;
	if(n < Nobservables){
#ifdef MEASURE_CUSTOM_OBSERVABLES
		s = Mnames[n];
#endif
	} else switch(n-Nobservables){
			case 0: s = "H";             break;
			case 1: s = "H^2";           break;
			case 2: s = "H_{diag}";      break;
			case 3: s = "H_{diag}^2";    break;
			case 4: s = "H_{offdiag}";   break;
			case 5: s = "H_{offdiag}^2"; break;
			case 6: s = "Z_magnetization"; break;
			case 7: s = "Z_magnetization^2"; break;
			case 8: s = "ratio"; break; //gabe
	}
	return s;
}

double mean_O[N_all_observables], stdev_O[N_all_observables], mean_O_backup[N_all_observables];

const int N_derived_observables = 2;  // we define number of derived observables

std::string name_of_derived_observable(int n){ // we define names of derived observables
	std::string s;
	switch(n){
		case 0 : s = "specific heat"; break;
		case 1 : s = "magnetic susceptibility"; break;
	}
	return s;
}

int valid_derived_observable(int n){ // we define which observables are needed for each derived observable
	int r = 0;
	switch(n){
		case 0: r = valid_observable[Nobservables+0] && valid_observable[Nobservables+1]; break;
		case 1: r = valid_observable[Nobservables+6] && valid_observable[Nobservables+7]; break;
	}
	return r;
}

double compute_derived_observable(int n){ // we compute the derived observables
	double R = 0;
	switch(n){
		case 0 : R = replica.beta*replica.beta*(mean_O[Nobservables+1] - mean_O[Nobservables+0]*mean_O[Nobservables+0]); break;
		case 1 : R = replica.beta*(mean_O[Nobservables+7] - mean_O[Nobservables+6]*mean_O[Nobservables+6]); break;
	}
	return R;
}
