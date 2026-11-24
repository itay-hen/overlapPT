class Replica{

	public:
	//denom
	double beta;
	ExExFloat currWeight;
	//Gabe
	//num
	double beta2;
	ExExFloat currWeight2;
	
	int q; 
	int Sq[qmax]; // sequence of permutations 
	std::bitset<N> lattice; // single configuration of spins
	std::bitset<N> z; // reference configuration of spins
	int qmax_achieved=0; // maximum number of operators in the sequence
	int cycles_used[Ncycles];
	// added by gabriel for telescope
	int TstepsFin = 0;
	unsigned long long stp = 0;
	unsigned long long m_step = 0;
	
	//private:
	
	divdiff d;
	ExExFloat beta_pow_factorial[qmax]; // contains the values (-beta)^q / q!
	//Gabe 
	divdiff d2;
	ExExFloat beta2_pow_factorial[qmax]; // for the other beta (prob better to change to num and denom for clarity)
	
	int cycles_used_backup[Ncycles];
	int found_cycles, min_index, max_index;
	int q_saved;
	std::bitset<N> lattice_backup;
	std::bitset<Nop> P;
	int Sq_backup[qmax];
	int Sq_saved[qmax];
	int Sq_subseq[qmax];
	int Sq_gaps[qmax];
	double Energies[qmax+1];
	double Energies_backup[qmax+1];
	int eoccupied[qmax+1];
	double currEnergy;
	std::complex<double> old_currD, currD;
	std::complex<double> currD_partial[qmax];
	int distance_from_identity = 0;
	int saved_distance_from_identity;
	
	#define fout(obj) { foutput.write((char *)& obj, sizeof(obj)); }
	#define fin(obj)  { finput.read((char *)& obj, sizeof(obj)); }
	
	public:
	
	void save_QMC_data(int printout = 1){
		if(printout) std::cout<<"SIGTERM signal detected. Saving unfinished calculation...";
		char fname[100]; if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
		std::ofstream foutput(fname,std::ios::binary); double elapsed;
		fout(rng); fout(bin_length);
		fout(cycle_len); fout(cycles_used); fout(cycles_used_backup); fout(cycle_min_len); fout(cycle_max_len);
		fout(found_cycles); fout(min_index); fout(max_index); fout(TstepsFin);
		fout(in_bin_sum); fout(bin_mean); fout(in_bin_sum_sgn); fout(bin_mean_sgn); 
		fout(q); fout(qmax_achieved); fout(lattice); fout(z); fout(P); fout(P_in_cycles);
		fout(Sq); fout(Sq_backup); fout(Sq_subseq); fout(Sq_gaps);
		fout(Energies); fout(Energies_backup); fout(eoccupied); fout(currEnergy); fout(valid_observable);
		fout(currD); fout(old_currD); fout(currD_partial); fout(zero); fout(currWeight); fout(stp); fout(m_step); 
		fout(rng_seed); fout(meanq); fout(maxq);
	#ifdef MPI_VERSION
		elapsed = MPI_Wtime() - start_time;
	#else
		elapsed = (double)clock() / CLOCKS_PER_SEC - start_time;
	#endif
		fout(elapsed);
		foutput.close(); if(printout) std::cout<<"done"<<std::endl; fflush(stdout);
	}
	
	int check_QMC_data(){
	#ifdef RESUME_CALCULATION
		int i,r,g; char fname[100];
		if(mpi_rank==0 && mpi_size>0){
			r = 1;
			for(i=0;i<mpi_size;i++){
				sprintf(fname,"qmc_data_%d.dat",mpi_rank);
				std::ifstream finput(fname,std::ios::binary);
				g = finput.good(); if(g) finput.close(); r = r && g;
			}
		} else{
			if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
			std::ifstream finput(fname,std::ios::binary);
			r = finput.good(); if(r) finput.close();
		}
	#else
		int r = 0;
	#endif
		return r;
	}
	
	void load_QMC_data(){
		char fname[100]; if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
		std::ifstream finput(fname,std::ios::binary); double elapsed;
		if(finput.good()){
			fin(rng); fin(bin_length_old);
			fin(cycle_len); fin(cycles_used); fin(cycles_used_backup); fin(cycle_min_len); fin(cycle_max_len);
			fin(found_cycles); fin(min_index); fin(max_index); fin(TstepsFin);
			fin(in_bin_sum); fin(bin_mean); fin(in_bin_sum_sgn); fin(bin_mean_sgn); 
			fin(q); fin(qmax_achieved); fin(lattice); fin(z); fin(P); fin(P_in_cycles);
			fin(Sq); fin(Sq_backup); fin(Sq_subseq); fin(Sq_gaps);
			fin(Energies); fin(Energies_backup); fin(eoccupied); fin(currEnergy); fin(valid_observable);
			fin(currD); fin(old_currD); fin(currD_partial); fin(zero); fin(currWeight); fin(stp); fin(m_step); 
			fin(rng_seed); fin(meanq); fin(maxq); fin(elapsed); if(finput.gcount()==0) elapsed = 0;
			finput.close(); start_time -= elapsed;
			if(mpi_size > 0){
				if(TstepsFin){
					std::cout<<"MPI process No. "<< mpi_rank <<" loaded unfinished calculation: all Tsteps completed, main steps completed: "
						<<m_step*stepsPerMeasurement<<" of "<<steps<<"."<<std::endl;
				} else std::cout <<"MPI process No. "<< mpi_rank <<" loaded unfinished calculation: Tsteps completed: "<<stp<<" of "<<Tsteps<<"."<<std::endl;
			} else{
				if(TstepsFin){
					std::cout<<"Loaded unfinished calculation: all Tsteps completed, main steps completed: "
						<<m_step*stepsPerMeasurement<<" of "<<steps<<"."<<std::endl;
				} else std::cout <<"Loaded unfinished calculation: Tsteps completed: "<<stp<<" of "<<Tsteps<<"."<<std::endl;
			}
		} else{
			if(mpi_size>0) std::cout<<"MPI process No. "<<mpi_rank<<": error opening file "<<fname<<std::endl;
			else std::cout<<"Error opening file "<<fname<<std::endl; fflush(stdout);
	#ifdef MPI_VERSION
			MPI_Abort(MPI_COMM_WORLD,1);
	#else
			exit(1);
	#endif
		}
		if(bin_length != bin_length_old){ // It is allowed to increase steps by an integer number of times for a completed calculation
			if(bin_length > 0 && bin_length % bin_length_old == 0){ // All other parameters should remain unchanged
				double sum; int i, j, o, m = bin_length / bin_length_old, curr_bins = Nbins/m; // merging bins together
				for(o=0;o<N_all_observables;o++) for(i=0;i<curr_bins;i++){
					sum = 0; for(j=0;j<m;j++) sum += bin_mean[o][m*i+j]; sum /= m;
					bin_mean[o][i] = sum;
				}
				for(i=0;i<curr_bins;i++){
					sum = 0; for(j=0;j<m;j++) sum += bin_mean_sgn[m*i+j]; sum /= m;
					bin_mean_sgn[i] = sum;
				}
				for(o=0;o<N_all_observables;o++){
					sum = 0; for(i=curr_bins*m;i<Nbins;i++) sum += bin_mean[o][i]*bin_length_old;
					in_bin_sum[o] += sum;
				}
				sum = 0; for(i=curr_bins*m;i<Nbins;i++) sum += bin_mean_sgn[i]*bin_length_old;
				in_bin_sum_sgn += sum;
			} else{
				std::cout << "Error: bin_length = " << bin_length <<" is not divisible by bin_length_old = " << bin_length_old << std::endl; fflush(stdout);
	#ifdef MPI_VERSION
				MPI_Abort(MPI_COMM_WORLD,1);
	#else
				exit(1);
	#endif
			}
		}
	}
	
	public:
	
	double CalcEnergy(){ // calculate the energy <z | D_0 | z> of a given configuration of spins
		std::complex<double> sum = 0;
		for(int i=0;i<D0_size;i++) sum -= double(2*(int((D0_product[i] & (~lattice)).count())%2)-1) * D0_coeff[i];
		return sum.real();
	}
	
	std::complex<double> calc_d(int k){ // calculate d_k = <z | D_k | z> for the current configuration of spins,    diagonal matricx elemenets 
		std::complex<double> sum = 0;
		for(int i=0;i<D_size[k];i++) sum -= double(2*(int((D_product[k][i] & (~lattice)).count())%2)-1) * D_coeff[k][i];
		return sum;
	}
	
	void ApplyOperator(int k){
		lattice ^= P_matrix[k];
	}
	
	void GetEnergies(){
	#ifdef WORM_UPDATE
		z = lattice;
	#endif
		currD = currD_partial[0] = 1;
		for(int i=0;i<q;i++){
			Energies[i] = CalcEnergy();
			ApplyOperator(Sq[i]);
			currD *= calc_d(Sq[i]);
			currD_partial[i+1] = currD;
		}
		Energies[q] = CalcEnergy(); currEnergy = Energies[0];
	#ifdef WORM_UPDATE
		lattice ^= z; distance_from_identity = lattice.count();
		lattice = z;
	#endif
	}
	
	public:
	
	ExExFloat GetWeight(){
		d.CurrentLength=0;
		GetEnergies();
		for(int i=0;i<=q;i++) d.AddElement(-beta*Energies[i]);
		return d.divdiffs[q] * beta_pow_factorial[q] * REALW(currD);
	}
	
	ExExFloat GetWeight2(){
		d2.CurrentLength=0;
		GetEnergies();
		for(int i=0;i<=q;i++) d2.AddElement(-beta2*Energies[i]);
		return d2.divdiffs[q] * beta2_pow_factorial[q] * REALW(currD);	
	}
	
	ExExFloat UpdateWeight(){
		int i, j, notfound, n=d.CurrentLength; double value;
		GetEnergies(); memset(eoccupied,0,(q+1)*sizeof(int));
		for(i=0;i<n;i++){
				notfound = 1; value = d.z[i];
			for(j=0;j<=q;j++) if(eoccupied[j]==0 && value == -beta*Energies[j]){ notfound = 0; break; }
			if(notfound) break; eoccupied[j] = 1;
		}
		if(i==0) d.CurrentLength=0; else while(n>i){ d.RemoveElement(); n--; }
		j=0; while(i<=q){ while(eoccupied[j]) j++; d.AddElement(-beta*Energies[j++]); i++; }
			return d.divdiffs[q] * beta_pow_factorial[q] * REALW(currD);
	}
	
	ExExFloat UpdateWeight2(){
		int i, j, notfound, n=d2.CurrentLength; double value;
		GetEnergies(); memset(eoccupied,0,(q+1)*sizeof(int));
		for(i=0;i<n;i++){
				notfound = 1; value = d2.z[i];
			for(j=0;j<=q;j++) if(eoccupied[j]==0 && value == -beta2*Energies[j]){ notfound = 0; break; }
			if(notfound) break; eoccupied[j] = 1;
		}
		if(i==0) d2.CurrentLength=0; else while(n>i){ d2.RemoveElement(); n--; }
		j=0; while(i<=q){ while(eoccupied[j]) j++; d2.AddElement(-beta2*Energies[j++]); i++; }
			return d2.divdiffs[q] * beta2_pow_factorial[q] * REALW(currD);
	}
	
	private:
	
	ExExFloat UpdateWeightReplace(double removeEnergy, double addEnergy){
		if(removeEnergy != addEnergy){
			if(d.RemoveValue(-beta*removeEnergy)) d.AddElement(-beta*addEnergy); else{
				std::cout << "Error: energy not found1" << std::endl; exit(1);
			}
		}
		return d.divdiffs[q] * beta_pow_factorial[q] * REALW(currD); // use this value only when the values of q and currD are correct
	}
	
	ExExFloat UpdateWeightDel(double removeEnergy1, double removeEnergy2){
		if(d.RemoveValue(-beta*removeEnergy1) && d.RemoveValue(-beta*removeEnergy2))
			return d.divdiffs[q] * beta_pow_factorial[q] * REALW(currD);  // use this value only when the values of q and currD are correct
		else{
			std::cout << "Error: energy not found1" << std::endl; exit(1);
		}
	}
	
	ExExFloat UpdateWeightIns(double addEnergy1, double addEnergy2){
		d.AddElement(-beta*addEnergy1); d.AddElement(-beta*addEnergy2);
		return d.divdiffs[q] * beta_pow_factorial[q] * REALW(currD);    // use this value only when the values of q and currD are correct
	}
	
	
	int NoRepetitionCheck(int* sequence, int r){ // check for absence of repetitions in a sequence of length r
		int i,j,rep = 1;
		for(i=0;i<r && rep;i++) for(j=0;j<i;j++) if(sequence[j]==sequence[i]){ rep = 0; break;}
		return rep;
	}
	
	void PickSubsequence(int r){ // randomly picks a sequential sub-sequence of length r from Sq
		int i,m; m = int(val(rng)*(q-r+1)); // m is random integer between 0 and q-r
		for(i=0;i<r;i++) Sq_subseq[i] = Sq[i+m];
		min_index = m; max_index = m+r-1;
	}
	
	int FindCycles(int r){  // find all cycles of length between lmin and lmax, each containing all operators of the array Sq_subseq of length r.
		int i,k,sum; std::bitset<Ncycles> curr; curr.set();
		for(i=0;i<r;i++) curr &= P_in_cycles[Sq_subseq[i]];
	#ifndef EXHAUSTIVE_CYCLE_SEARCH
		for(i=0;i<Ncycles;i++) if(curr[i]) if(cycle_len[i]<lmin || cycle_len[i]>lmax) curr.reset(i);
	#endif
		found_cycles = curr.count();
		if(found_cycles > 0){
			k = int(val(rng)*found_cycles);
			i=sum=0; while(sum <= k) sum += curr[i++]; i--;
			return i;
		} else return -1;
	}
	
	double Metropolis(ExExFloat newWeight){
		return min(1.0,fabs((newWeight/currWeight).get_double()));
	}
	
	public:
	
	#ifdef WORM_UPDATE
	
	#define WORM_BREAK_PROBABILITY 0.1
	
	int worm_exit_check(){
		int r = save_data_flag;
	#ifdef WORM_BREAK_PROBABILITY
		if(val(rng) < WORM_BREAK_PROBABILITY) r = 1;
	#endif
		return r;
	}
	
	void worm_update(){  // employing worm update in addition to other updates
		int i,m,p,r,u; double oldE, oldE2, v = Nop>0 ? val(rng) : 1; ExExFloat newWeight;
		if(v < 0.3){ //local move
			if(v<0.1 && q>=2){ // attempt to swap Sq[m] and Sq[m+1]
				m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
				if(Sq[m]!=Sq[m+1]){
					oldE = Energies[m+1]; old_currD = currD;
					p = Sq[m]; Sq[m] = Sq[m+1]; Sq[m+1] = p; GetEnergies(); newWeight = UpdateWeightReplace(oldE,Energies[m+1]);
					if(val(rng) < Metropolis(newWeight)) currWeight = newWeight; else {
						Sq[m+1] = Sq[m]; Sq[m] = p; currD = old_currD;
						UpdateWeightReplace(Energies[m+1],oldE); Energies[m+1] = oldE;
					}
				}
			} else if(v<0.2){ // attempt to delete Sq[m] and Sq[m+1]
				if(q>=2){
					m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
					if(Sq[m]==Sq[m+1]){
						oldE = Energies[m]; oldE2 = Energies[m+1]; old_currD = currD;
						memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
						for(i=m;i<q-2;i++) Sq[i] = Sq[i+2]; q-=2;
						GetEnergies(); newWeight = UpdateWeightDel(oldE,oldE2);
						if(val(rng) < Metropolis(newWeight)/Nop) currWeight = newWeight; else{
							q+=2; memcpy(Sq,Sq_backup,q*sizeof(int)); memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
							currD = old_currD; UpdateWeightIns(oldE,oldE2);
						}
					}
				}
			} else{
				if(q+2<qmax){ // attempt to insert Sq[m] and Sq[m+1]
					m = int(val(rng)*(q+1)); // m is between 0 and q
					memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
					old_currD = currD; p = diceNop(rng);
					for(i=q-1;i>=m;i--) Sq[i+2] = Sq[i]; q+=2; Sq[m] = Sq[m+1] = p;
					GetEnergies(); newWeight = UpdateWeightIns(Energies[m],Energies[m+1]);
					if(val(rng) < Metropolis(newWeight)) currWeight = newWeight; else{
						q-=2; memcpy(Sq,Sq_backup,q*sizeof(int)); memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
						currD = old_currD; d.RemoveElement(); d.RemoveElement();
					}
				} else qmax_achieved = 1;
			}
		} else if(v < 0.4){ // attempt to insert a randomly chosen single permutation operator
			if(q+1<qmax){
				saved_distance_from_identity = distance_from_identity;
				m = int(val(rng)*(q+1)); // m is between 0 and q
				memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
				old_currD = currD; p = diceNop(rng);
				for(i=q-1;i>=m;i--) Sq[i+1] = Sq[i]; q++; Sq[m] = p; newWeight = UpdateWeight();
				if(saved_distance_from_identity && distance_from_identity && worm_exit_check()){
					q=q_saved; memcpy(Sq,Sq_saved,q_saved*sizeof(int));
					currWeight = UpdateWeight();
				} else if(val(rng) < Metropolis(newWeight)){
					currWeight = newWeight;
					if(saved_distance_from_identity == 0){
						memcpy(Sq_saved,Sq_backup,(q-1)*sizeof(int)); q_saved = q-1;
					}				
				} else{
					q--; memcpy(Sq,Sq_backup,q*sizeof(int)); memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
					currD = old_currD; UpdateWeight();
				}
			} else qmax_achieved = 1;
		} else if(v < 0.5){ // attempt to remove a randomly chosen single permutation operator
			if(q>0){
				saved_distance_from_identity = distance_from_identity;
				m = int(val(rng)*q); p = Sq[m]; // m is between 0 and (q-1)
				old_currD = currD; memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
				for(i=m;i<q-1;i++) Sq[i] = Sq[i+1]; q--; newWeight = UpdateWeight();
				if(saved_distance_from_identity && distance_from_identity && worm_exit_check()){
					q=q_saved; memcpy(Sq,Sq_saved,q_saved*sizeof(int));
					currWeight = UpdateWeight();
				} else if(val(rng) < Metropolis(newWeight)/Nop){
					currWeight = newWeight;
					if(saved_distance_from_identity == 0){
						memcpy(Sq_saved,Sq_backup,(q+1)*sizeof(int)); q_saved = q+1;
					}
				} else{
					q++; memcpy(Sq,Sq_backup,q*sizeof(int)); memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
					currD = old_currD; UpdateWeight();
				}
			}
		} else if(v < 0.6){ // attempting a fundamental cycle completion
			int oldq, j = 0, inv_pr; double wfactor;
			u = geometric_int(rng); // a random integer u is picked according to geometric distribution
			if(q >= u+rmin){
				inv_pr = min(rmax,q-u)-(rmin)+1;
				r = int(val(rng)*inv_pr) + (rmin);  // r is random integer between rmin and min(rmax,q-u)
				PickSubsequence(r+u); // indexes of the subsequence are min_index, min_index+1,..., max_index=min_index+r+u-1
				std::shuffle(Sq_subseq,Sq_subseq+r+u,rng);
				if(NoRepetitionCheck(Sq_subseq,r)){
					for(i=0;i<u;i++) Sq_gaps[i] = Sq_subseq[i+r];
					m = FindCycles(r);
					if(found_cycles > 0){ // cycles[m] is one of the found cycles, containing all the operators of Sq_subseq
						P = cycles[m]; for(i=0;i<r;i++) P.reset(Sq_subseq[i]);
						p = P.count(); // here, p is length of the complement sequence S'
						if(q+p-r < qmax){
							memcpy(Sq_backup,Sq,q*sizeof(int)); oldq = q;
							if(r<p)	     for(i=q-1;i>max_index;i--) Sq[i+p-r] = Sq[i]; // shift the values to the right
							else if(r>p) for(i=max_index+1;i<q;i++) Sq[i+p-r] = Sq[i]; // shift the values to the left
							for(i=0;i<p;i++){ while(!P.test(j)) j++; Sq_subseq[i] = Sq[min_index+i] = j++;}
							for(i=0;i<u;i++) Sq[min_index+p+i] = Sq_gaps[i]; // S' contains the remaining operators
							std::shuffle(Sq+min_index,Sq+min_index+p+u,rng);
							q += p-r; // the length q may have changed
							newWeight = UpdateWeight();
							wfactor = found_cycles; FindCycles(p); wfactor /= found_cycles;
							wfactor *= factorial[p]/factorial[r];
							wfactor *= inv_pr; inv_pr = min(rmax,q-u)-(rmin)+1; wfactor /= inv_pr;
							if(val(rng) < Metropolis(newWeight*wfactor)){
								currWeight = newWeight; cycles_used[m] = 1;
							} else{
								q = oldq; memcpy(Sq,Sq_backup,q*sizeof(int)); currWeight = UpdateWeight();
							}
						} else qmax_achieved = 1;
					}
				}
			}
		} else if(v < 0.9){ // attempting a block swap
			if(q>=2 && distance_from_identity==0){
				m = q==2 ? 0 : int(val(rng)*(q-1)); // m is between 0 and (q-2)
				lattice_backup = lattice;
				oldE = currEnergy; for(i=0;i<=m;i++) ApplyOperator(Sq[i]);
				for(i=0;i<=m;i++)  Sq_backup[q-1-m+i] = Sq[i];
				for(i=m+1;i<q;i++) Sq_backup[i-m-1] = Sq[i];
				for(i=0;i<q;i++) { p = Sq[i]; Sq[i] = Sq_backup[i]; Sq_backup[i] = p;}
				memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
				GetEnergies(); newWeight = UpdateWeightReplace(oldE,currEnergy);
				if(val(rng) < Metropolis(newWeight)) currWeight = newWeight; else{
					UpdateWeightReplace(currEnergy,oldE); currEnergy = oldE;
					lattice = lattice_backup; memcpy(Sq,Sq_backup,q*sizeof(int));
					memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
				}
			}
		} else if(distance_from_identity==0){ // flip of a random spin
			p = diceN(rng);	lattice.flip(p); newWeight = UpdateWeight();
			if(val(rng) < Metropolis(newWeight)) currWeight = newWeight;
				else { lattice.flip(p); currWeight = UpdateWeight();}
		}
	}
	
	void update(){
		if(save_data_flag){ save_QMC_data(); exit(0); }
		do worm_update(); while(distance_from_identity);
	}
	
	#else
	
	void update(){
		if(save_data_flag){ save_QMC_data(); exit(0); };
		int i,m,p,r,u,oldq,cont; double oldE, oldE2, v = Nop>0 ? val(rng) : 1; ExExFloat newWeight; double Rfactor;
		if(v < 0.8){ // composite update
			Rfactor = 1; oldq = q; memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(cycles_used_backup,cycles_used,Ncycles*sizeof(int));
			newWeight = currWeight;
			do{
				cont = 0; v = val(rng);
				if(v < 0.25){ // attempt to swap Sq[m] and Sq[m+1]
					if(q>=2){
						m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
						if(Sq[m]!=Sq[m+1]){
							oldE = Energies[m+1]; old_currD = currD;
							p = Sq[m]; Sq[m] = Sq[m+1]; Sq[m+1] = p;
							GetEnergies();
							newWeight = UpdateWeightReplace(oldE,Energies[m+1]);
							if(REALW(currD) == 0) cont = 1;
						}
					}
				} else if(v < 0.5){ // attempt to delete Sq[m] and Sq[m+1]
					if(q>=2){
						m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
						if(Sq[m]==Sq[m+1]){
							oldE = Energies[m]; oldE2 = Energies[m+1]; old_currD = currD;
							for(i=m;i<q-2;i++) Sq[i] = Sq[i+2]; q-=2;
							GetEnergies(); Rfactor /= Nop;
							newWeight = UpdateWeightDel(oldE,oldE2);
							if(REALW(currD) == 0) cont = 1;
						}
					}
				} else if(v < 0.75){
					if(q+2<qmax){ // attempt to insert Sq[m] and Sq[m+1]
						m = int(val(rng)*(q+1)); // m is between 0 and q
						old_currD = currD; p = diceNop(rng);
						for(i=q-1;i>=m;i--) Sq[i+2] = Sq[i]; q+=2; Sq[m] = Sq[m+1] = p;
						GetEnergies(); Rfactor *= Nop;
						newWeight = UpdateWeightIns(Energies[m],Energies[m+1]);
						if(REALW(currD) == 0) cont = 1;
					} else qmax_achieved = 1;
				} else{ // attempting a fundamental cycle completion
					int j = 0, inv_pr; double wfactor;
					u = geometric_int(rng); // a random integer u is picked according to geometric distribution
					if(q >= u+rmin){
						inv_pr = min(rmax,q-u)-(rmin)+1;
						r = int(val(rng)*inv_pr) + (rmin);  // r is random integer between rmin and min(rmax,q-u)
						PickSubsequence(r+u); // indexes of the subsequence are min_index, min_index+1,..., max_index=min_index+r+u-1
						std::shuffle(Sq_subseq,Sq_subseq+r+u,rng);
						if(NoRepetitionCheck(Sq_subseq,r)){
							for(i=0;i<u;i++) Sq_gaps[i] = Sq_subseq[i+r];
							m = FindCycles(r);
							if(found_cycles > 0){ // cycles[m] is one of the found cycles, containing all the operators of Sq_subseq
								P = cycles[m]; for(i=0;i<r;i++) P.reset(Sq_subseq[i]);
								p = P.count(); // here, p is length of the complement sequence S'
								if(q+p-r < qmax){
									if(r<p)	     for(i=q-1;i>max_index;i--) Sq[i+p-r] = Sq[i]; // shift the values to the right
									else if(r>p) for(i=max_index+1;i<q;i++) Sq[i+p-r] = Sq[i]; // shift the values to the left
									for(i=0;i<p;i++){ while(!P.test(j)) j++; Sq_subseq[i] = Sq[min_index+i] = j++;}
									for(i=0;i<u;i++) Sq[min_index+p+i] = Sq_gaps[i]; // S' contains the remaining operators
									std::shuffle(Sq+min_index,Sq+min_index+p+u,rng);
									q += p-r; // the length q may have changed
									newWeight = UpdateWeight();
									wfactor = found_cycles; FindCycles(p); wfactor /= found_cycles;
									wfactor *= factorial[p]/factorial[r];
									wfactor *= inv_pr; inv_pr = min(rmax,q-u)-(rmin)+1; wfactor /= inv_pr;
									Rfactor *= wfactor;
									cycles_used[m] = 1;
									if(REALW(currD) == 0) cont = 1;
								} else qmax_achieved = 1;
							}
						}
					}
				}
			} while(cont &&
	#ifdef HURRY_ON_SIGTERM
				!save_data_flag &&
	#endif
				val(rng) > COMPOSITE_UPDATE_BREAK_PROBABILITY);
			if(REALW(currD) != 0 && val(rng) < Metropolis(newWeight*Rfactor)){
				currWeight = newWeight;
			} else{
				q = oldq;
				memcpy(Sq,Sq_backup,q*sizeof(int));
				memcpy(cycles_used,cycles_used_backup,Ncycles*sizeof(int));
				currWeight = UpdateWeight();
			}
		} else if(v < 0.9 && q>=2){ // attempting a block swap
			m = q==2 ? 0 : int(val(rng)*(q-1)); // m is between 0 and (q-2)
			oldE = currEnergy; for(i=0;i<=m;i++) ApplyOperator(Sq[i]);
			for(i=0;i<=m;i++)  Sq_backup[q-1-m+i] = Sq[i];
			for(i=m+1;i<q;i++) Sq_backup[i-m-1] = Sq[i];
			for(i=0;i<q;i++) { p = Sq[i]; Sq[i] = Sq_backup[i]; Sq_backup[i] = p;}
			memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
			GetEnergies(); newWeight = UpdateWeightReplace(oldE,currEnergy);
			if(val(rng) < Metropolis(newWeight)){
				z = lattice; currWeight = newWeight;
			} else{ UpdateWeightReplace(currEnergy,oldE); currEnergy = oldE;
				lattice = z; memcpy(Sq,Sq_backup,q*sizeof(int));
				memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
			}
		} else{ // flip of a random spin
			p = diceN(rng);	lattice.flip(p); newWeight = UpdateWeight();
			if(val(rng) < Metropolis(newWeight)) { z = lattice; currWeight = newWeight;}
				else { lattice.flip(p); currWeight = UpdateWeight();}
		}
	}
	
	
	
	#endif
	
	#ifdef MEASURE_CUSTOM_OBSERVABLES
	
	std::complex<double> calc_MD0(int n){ // calculate <z | MD_0 | z> for the current configuration of spins and observable n
		std::complex<double> sum = 0;
		for(int i=0;i<MD0_size[n];i++) sum -= double(2*(int((MD0_product[n][i] & (~lattice)).count())%2)-1) * MD0_coeff[n][i];
		return sum;
	}
	
	std::complex<double> calc_MD(int n, int k){ // calculate d_k = <z | MD_k | z> for the current configuration of spins and observable n
		std::complex<double> sum = 0;
		for(int i=0;i<MD_size[n][k];i++) sum -= double(2*(int((MD_product[n][k][i] & (~lattice)).count())%2)-1) * MD_coeff[n][k][i];
		return sum;
	}
	
	#endif
	
	double measure_H(){
		double R = d.z[q]/(-beta);
		if(q > 0) R += (d.divdiffs[q-1]/d.divdiffs[q]).get_double()*q/(-beta);
		return R;
	}
	
	double measure_H2(){
		double R = (d.z[q]/(-beta))*(d.z[q]/(-beta));
		if(q>0) R += (d.z[q]/(-beta) + d.z[q-1]/(-beta))*(d.divdiffs[q-1]/d.divdiffs[q]).get_double()*q/(-beta);
		if(q>1) R += (d.divdiffs[q-2]/d.divdiffs[q]).get_double()*(q*(q-1))/(-beta)/(-beta);
		return R;
	}
	
	double measure_Hdiag(){
		return currEnergy;
	}
	
	double measure_Hdiag2(){
		return currEnergy*currEnergy;
	}
	
	double measure_Hoffdiag(){
		double R = 0;
		if(q > 0) R += (d.divdiffs[q-1]/d.divdiffs[q]).get_double()*q/(-beta);
		return R;
	}
	
	double measure_Hoffdiag2(){
		double R = (d.z[q]/(-beta))*(d.z[q]/(-beta)) + currEnergy*(currEnergy - 2*measure_H());
		if(q>0) R += (d.z[q]/(-beta) + d.z[q-1]/(-beta))*(d.divdiffs[q-1]/d.divdiffs[q]).get_double()*q/(-beta);
		if(q>1) R += (d.divdiffs[q-2]/d.divdiffs[q]).get_double()*(q*(q-1))/(-beta)/(-beta);
		return R;
	}
	
	double measure_Z_magnetization(){
		return (2.0*lattice.count() - N)/N; // this value is between -1 and 1
	}
	
	double measure_Z_magnetization2(){
		double m = measure_Z_magnetization();
		return m*m;
	}
	
	double measure_ratio(){
		return (GetWeight2()/currWeight).get_double();
	}
	
	double measure_observable(int n){
		double R = 0;
		if(valid_observable[n]){
			if(n < Nobservables){
	#ifdef MEASURE_CUSTOM_OBSERVABLES
				int i,k,len,cont;
				std::complex<double> T = calc_MD0(n);
				for(k=0;k<MNop[n];k++){
					P = MP[n][k]; len = P.count(); if(len>q) continue;
					if(!NoRepetitionCheck(Sq+(q-len),len)) continue;
					cont = 0; for(i=0;i<len;i++) if(!P.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
					T +=	(d.divdiffs[q-len]/d.divdiffs[q]).get_double() *
							(beta_pow_factorial[q-len]/beta_pow_factorial[q]).get_double()/factorial[len] *
						(currD_partial[q-len]/currD) * calc_MD(n,k);
				}
	#ifdef ABS_WEIGHTS
				R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
				R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	#endif
			} else  switch(n-Nobservables){
					case 0:	R = measure_H(); break;
					case 1:	R = measure_H2(); break;
					case 2:	R = measure_Hdiag(); break;
					case 3:	R = measure_Hdiag2(); break;
					case 4:	R = measure_Hoffdiag(); break;
					case 5:	R = measure_Hoffdiag2(); break;
					case 6: R = measure_Z_magnetization(); break;
					case 7: R = measure_Z_magnetization2(); break;
					case 8: R = measure_ratio(); break; 
			}
		}
		return R;
	}
	
	
	void sync_beta2_factorial() {
		// Rebuild d2 to match the current configuration
		d2.CurrentLength = 0;
		GetEnergies();
		for(int i = 0; i <= q; i++) {
			d2.AddElement(-beta2 * Energies[i]);
		}
		currWeight2 = d2.divdiffs[q] * beta2_pow_factorial[q] * REALW(currD);
	}
	
	ExExFloat new_measure(){
	
		//sync_beta2_factorial();
	
		currWeight = GetWeight();
		currWeight2 = GetWeight2();
	
		if(beta2 == 0){
			return currWeight;
		}
	
		return currWeight2/currWeight;
	}
	
	void measure(){
		double R, sgn; int i;
		currWeight = GetWeight();
	#ifdef ABS_WEIGHTS
		sgn = currWeight.sgn() * cos(std::arg(currD)); // arg(W) = arg(currD) + arg(currWeight), where arg(currWeight) = either 0 or Pi
	#else
		sgn = currWeight.sgn();
	#endif
		meanq += q; if(maxq < q) maxq = q; in_bin_sum_sgn += sgn;
		if((m_step+1) % bin_length == 0){
			in_bin_sum_sgn /= bin_length; bin_mean_sgn[m_step/bin_length] = in_bin_sum_sgn; in_bin_sum_sgn = 0;
		}
		for(i=0;i<N_all_observables;i++){
			R = measure_observable(i); in_bin_sum[i] += R*sgn;
			if((m_step+1) % bin_length == 0){
				in_bin_sum[i] /= bin_length; bin_mean[i][m_step/bin_length] = in_bin_sum[i]; in_bin_sum[i] = 0;
			}
		}
	}
	
	//Overloaded function to avoid breaking other parts of the code 
	void set_beta(double beta_smaller, double beta_bigger){
		beta = beta_smaller;
		beta2 = beta_bigger;
		ExExFloat curr(1); beta_pow_factorial[0] = curr;
		ExExFloat curr2(1); beta2_pow_factorial[0] = curr2;
		for(int k=1;k<qmax;k++){ 
			curr*=(-beta)/k; 
			beta_pow_factorial[k] = curr;
			curr2*=(-beta2)/k; 
			beta2_pow_factorial[k] = curr2;	
		}
		currWeight = GetWeight();
		currWeight2 = GetWeight2();
	}
	
	void init(double beta_smaller, double beta_bigger){ // Don't run this from constructor. Assuming that divdiff_init() and init_rng() have been previously executed.
		lattice = 0; for(int i=N-1;i>=0;i--) if(dice2(rng)) lattice.set(i); z = lattice; q=0; // lattice is filled randomly and q = 0
		set_beta(beta_smaller, beta_bigger);
	}
	
	
	void set_beta(double beta_value){
		beta = beta_value;
		ExExFloat curr(1); beta_pow_factorial[0] = curr;
		for(int k=1;k<qmax;k++){ 
			curr*=(-beta)/k; 
			beta_pow_factorial[k] = curr;
		}
		currWeight = GetWeight();;
	}
	
	void init(double beta_value){ // Don't run this from constructor. Assuming that divdiff_init() and init_rng() have been previously executed.
		lattice = 0; for(int i=N-1;i>=0;i--) if(dice2(rng)) lattice.set(i); z = lattice; q=0; // lattice is filled randomly and q = 0
		set_beta(beta_value);
	}
	
	Replica(): d(4,500), d2(4,500){ // constructor
	#ifdef WORM_UPDATE
		for(int i=0;i<Ncycles;i++) cycles_used[i] = 1;
	#else
		for(int i=0;i<Ncycles;i++) cycles_used[i] = 0;
	#endif
	}
	
};

// ═══════════════════════════════════════════════════════════════════════════════
// RenyiReplica Class
// ═══════════════════════════════════════════════════════════════════════════════
//
// Paper: "A Universal QMC Framework for Rényi Entanglement Entropies"
//         Kalev, Barash, Hen (2025), Sections II.D–II.F
//
// WHAT THIS CLASS DOES (plain English):
//   This is the core quantum object that knows how to compute the WEIGHT
//   W^(α)(C; β) for a given configuration C at inverse temperature β.
//
//   The weight is defined in Paper Eqs. 19–29 and encodes the probability
//   amplitude for the quantum system to be in a particular "path" through
//   configuration space. When the Monte Carlo sampler collects many configurations
//   weighted by W^(α), the average gives us the reduced partition function Z_α(β).
//
// THE KEY IDEA — MULTIPLE REPLICAS OF SUBSYSTEM B:
//   To compute Z_α (the α-th power of the reduced density matrix, Paper Eq. 3),
//   we need to imagine α independent copies of subsystem B, all sharing the same
//   subsystem A.
//
//   Think of it this way: the reduced density matrix ρ_A is a "table" of quantum
//   amplitudes for subsystem A. Taking ρ_A^α means multiplying this table α times.
//   In the PMR language, this becomes: one quantum path (operator sequence) on
//   subsystem A, but that path is "cut" into α segments, each segment being
//   threaded through a different copy of subsystem B.
//
//   So for α=2: one zA, two independent zB configurations (zB^0, zB^1)
//   For α=3: one zA, three independent zB configurations (zB^0, zB^1, zB^2)
//
// THE WEIGHT FORMULA  (Paper Eqs. 23–29):
//   W^(α)(C; β) = Σ_{cuts j_1<j_2<...<j_{α-1}}
//                    [product of segment weights, one per replica]
//
//   Each "cut" at position j divides the operator sequence into segments.
//   Segment r (from cut j_{r-1} to j_r) must satisfy a "closure" condition:
//   the B-subsystem of replica r must return to its starting configuration.
//   Subsystem A must also close over the FULL sequence.
//
// OPTIMIZATIONS [not explicitly specified in paper]:
//   1. PREFIX B-PERMUTATION CACHE: precomputes the running B-state at each
//      position, so checking closure of a segment [start,end) is O(1) instead
//      of O(sequence length).
//   2. ITERATIVE CUT ENUMERATION: generates all valid cut positions without
//      recursion or large memory allocation.
//   3. ENERGY PRECOMPUTATION: calculates all energies and D-coefficients once
//      per configuration update, reusing them across all cut evaluations.
//   4. PER-INSTANCE RNG: each replica has its own random number generator,
//      making parallel execution (OpenMP) safe without locks.
// ═══════════════════════════════════════════════════════════════════════════════

// [N<=64] bucket key for RenyiReplica::positions_in_class:
// B-state (<= 64 bits) << 64 | D-zero count. Was uint64_t with a 32-bit B field (N <= 32).
typedef unsigned __int128 pic_key_t;
struct PicKeyHash {
    size_t operator()(pic_key_t k) const noexcept {
        uint64_t lo = (uint64_t)k, hi = (uint64_t)(k >> 64);
        return std::hash<uint64_t>()(hi * 0x9E3779B97F4A7C15ULL ^ lo);
    }
};

class RenyiReplica {
private:
    //=========================================================================
    // PREFIX B-PERMUTATION CACHE  [OPTIMIZATION — not in paper]
    //
    // For each replica r, B_prefix_cache[r][i] stores "what is the B-subsystem
    // state of replica r after applying the first i operators in the sequence?"
    //
    // WHY THIS MATTERS:
    //   To check if a segment [start, end) "closes" on subsystem B, we need to
    //   verify that applying operators Sq[start]...Sq[end-1] brings the B-state
    //   back to where it started. Without the cache, this requires walking through
    //   all (end-start) operators every time — slow when checking many cut positions.
    //
    //   With the cache: segment [start, end) closes iff
    //     B_prefix_cache[r][end] == B_prefix_cache[r][start]
    //   This is a single bitset comparison — O(1) instead of O(sequence length).
    //
    //   For α=3 with a sequence length of 100, there are ~5000 candidate cut
    //   pairs. Without the cache each check costs ~50 operations = 250,000 ops.
    //   With the cache: 5000 × O(1) = 5000 ops. About 50× faster.
    //=========================================================================
    std::vector<std::vector<std::bitset<N>>> B_prefix_cache;

    // [OPT-12] Shared prefix-XOR table: PXOR[i] = XOR of P_matrix[Sq[0..i-1]].
    // Since Sq[] (the operator sequence) is shared across all alpha replicas,
    // this is REPLICA-INDEPENDENT: state_i^(r) = lattice[r] ^ PXOR[i] for any
    // replica r. Built once per full cache rebuild (ComputePrefixBPermutations,
    // O(q)); reused for the O(1) intermediate-state lookup in
    // PartialUpdateAfterSwap (was an O(m) per-replica walk). Valid exactly
    // when prefix_cache_valid is true (rebuilt/invalidated in the same places,
    // so no separate validity flag is needed).
    std::vector<std::bitset<N>> PXOR;

    // [OPT 3 — superseded] valid_cuts_from_zero has been removed.
    // HasAnyValidCuts() now looks up b_class_positions[0][B_prefix_cache[0][0].to_ulong()]
    // which is maintained by BuildBClassPositions() / BClassUpdatePosition() and gives
    // the same B-only closure list without a separate vector.

    // [OPT 1] B-only class index — used by HasAnyValidCuts() which runs BEFORE
    // energies are computed and therefore cannot use the D-zero component of PIC.
    //
    // b_class_positions[r][B_ulong] = sorted positions i where
    //   B_prefix_cache[r][i].to_ulong() == B_ulong.
    //
    // Built in ComputePrefixBPermutations(); position m+1 updated incrementally
    // in PartialUpdateAfterSwap() (only B_prefix_cache[r][m+1] changes there).
    std::vector<std::unordered_map<unsigned long, std::vector<int>>> b_class_positions;

    // [OPT E / 4.1] Pre-allocated scratch vectors — sized once in the constructor,
    // reused on every call to EnumerateAndSumCuts() and HasAnyValidCuts().
    // Eliminates O(alpha) heap allocations per call, which at K=32 replicas ×
    // N_SAMPLES × STEPS_PER_SAMP iterations amounts to tens of millions of
    // unnecessary malloc/free round-trips through the MC loop.
    std::vector<int> scratch_seg_starts;  // [alpha]
    std::vector<int> scratch_seg_ends;    // [alpha]
    std::vector<int> scratch_cuts;        // [max(1, alpha-1)]

    // [OPT F / 4.2] Valid inner cut positions for the α=3 specialized path.
    // For a fixed outer cut c0, holds positions j2 where segment [c0, j2) closes
    // on replica 1: B_prefix_cache[1][j2] == B_prefix_cache[1][c0].
    // Analogous to valid_cuts_from_zero for the outer cut, but recomputed inside
    // the c0 loop because its contents depend on the current c0.
    // Capacity reserved once; size filled per c0 iteration.
    std::vector<int> scratch_valid_j2;

    // [OPT 1] Old pic-keys saved at the start of PartialUpdateAfterSwap() before
    // any cache entries are overwritten, enabling an incremental O(alpha) update
    // instead of a full O(alpha*q) rebuild.
    // Layout: scratch_old_pic_keys[r] = old key for position m+1 on replica r;
    //         scratch_old_pic_keys[alpha+r] = old key for position m+2.
    std::vector<pic_key_t> scratch_old_pic_keys;

    // Cache validity flags [OPTIMIZATION]
    // When the operator sequence changes (e.g. after an MC move), the cached
    // values are stale. Rather than recomputing immediately, we mark them invalid
    // and recompute lazily only when GetRenyiWeight() actually needs them.
    bool energies_valid;
    bool prefix_cache_valid;

    // [OPT 7] A-closure accumulator: XOR of all P_subsystem_A[Sq[i]] for i=0..q-1.
    // A-closure holds iff this bitset is all-zero (net A-permutation = identity).
    // Maintained incrementally: swap sub-moves don't change it (XOR commutes);
    // cycle completion updates it by XOR-ing out old ops and XOR-ing in new ops.
    // set_beta() does NOT invalidate it — only Sq changes do.
    std::bitset<N> A_net_permutation;
    bool a_closure_valid;   // true iff A_net_permutation reflects current Sq

    // [OPT A] Coarse-grained dirty flags — cleaner signal for callers.
    // config_dirty = true whenever lattice[] or Sq[] changed (any of the three
    //   fine-grained cache flags is false).  Set by InvalidateConfig().
    //   Cleared by EnsureCachesValid() once all caches are rebuilt.
    // beta_dirty   = true when beta changed but beta_pow_factorial not yet rebuilt.
    //   Set by InvalidateBeta().  Cleared by set_beta() after rebuilding.
    //   Used by Optimization B (PT swap fast reject) — not yet active.
    bool config_dirty;
    bool beta_dirty;

    // [OPT 11] Number of valid entries in beta_pow_factorial.
    // beta_pow_factorial[0..beta_pow_computed-1] are correct for the current beta.
    // Avoids recomputing all qmax entries on every set_beta() call during the
    // MBAR beta-sweep; entries are extended lazily in UpdateWeight().
    int beta_pow_computed;

    // [OPT 15] Zero-D coefficient prefix sums for O(1) segment zero-check.
    // D_zero_prefix[r][i] = number of k in [0,i) where D_replicas[r][k] == 0.
    // Invariant: valid whenever energies_valid == true.
    // Maintained incrementally in PartialUpdateAfterSwap() for the two changed
    // positions; rebuilt from scratch in PrecomputeEnergiesAndDCoefficients().
    std::vector<std::vector<int>> D_zero_prefix;

    // [OPT 1] Pruned cut enumeration index — §1.1 / §4 (Win 2).
    // positions_in_class[r][key] is a sorted list of positions i ∈ [0,q] where
    //   pic_key(r, i) == key,  i.e.  B_prefix_cache[r][i] AND D_zero_prefix[r][i]
    //   both match.  Two positions with the same key define a segment that BOTH
    //   closes on replica r (B-closure) AND has no zero D-coefficients (D-pruning).
    //
    // Used by EnumerateAndSumCuts() to replace:
    //   α=3: the O(q) scratch_valid_j2 scan + explicit D_zero_prefix checks
    //   α≥4: the full O(q^{α-2}) odometer walk
    // with O(1) lookups followed by iteration over only the valid positions.
    //
    // Invariant: valid whenever energies_valid == true (both components of the key
    //   are maintained in sync with energies).  Rebuilt by BuildPositionsInClass().
    std::vector<std::unordered_map<pic_key_t, std::vector<int>, PicKeyHash>> positions_in_class;

    // [OPT-3] alpha=3 specialized path scratch — sized qmax+1, allocated once,
    // reused on every EnumerateAndSumCuts() call. Replaces the old per-(c0,j2)
    // pair rebuild of segment 2 and per-segment D-products with a single O(q)
    // pass over each, done once per weight call:
    //   w2_all[j2]  — segment-2 weight for [j2,q), ALL j2 in one right-anchored
    //                 incremental divdiff pass (divided differences are exact
    //                 under any node insertion order — see build site).
    //   D2_suf[j2]  — segment-2 D-coefficient product for [j2,q), same pass.
    //   Pnz0[i]     — running product of NONZERO D_replicas[0][0..i), skipping
    //                 zero factors (tracked via D_zero_prefix). A segment
    //                 [a,b) drawn from positions_in_class is guaranteed
    //                 zero-free, so Pnz0[b]/Pnz0[a] recovers the exact
    //                 product over [a,b) with no spurious division by zero.
    //   Pnz1[i]     — same, for replica 1 (used for segment 1 = [c0,j2)).
    std::vector<ExExFloat> w2_all;
    std::vector<std::complex<double>> D2_suf;
    std::vector<std::complex<double>> Pnz0;
    std::vector<std::complex<double>> Pnz1;

    // [OPT-4] alpha=2 two-pass scratch: w0_cache[c0] = segment-0 weight for
    // [0,c0), filled once per candidate during the ascending Pass A, consumed
    // during the descending Pass B. Sized qmax+1, allocated once.
    std::vector<ExExFloat> w0_cache;

    // [OPT-1] Beta-independent enumeration cache. tuples_valid mirrors
    // energies_valid/prefix_cache_valid but tracks whether Pnz0/Pnz1/D2_suf
    // (all beta-independent -- see BuildTupleTables()) are current for THIS
    // configuration. tuples_nonzero caches HasAnyValidCuts()'s result (also
    // beta-independent) so a K-beta sweep can skip straight to "weight is 0
    // for every beta" without repeating the gate sequence K times.
    // prefix_unsafe_cached is the OPT-3/4 numerical guard flag, now member
    // state so EvaluateEnumeratedAt() (called once per beta) can read what
    // BuildTupleTables() (called once per sample) computed.
    bool tuples_valid;
    bool tuples_nonzero;
    bool prefix_unsafe_cached;

public:
    //=========================================================================
    // CORE QMC PARAMETERS  (Paper Section II.D–II.F)
    //=========================================================================

    // β = 1/T, the inverse temperature at which this replica is simulating.
    // Each replica on the β-ladder has a different value (Section II.F).
    double beta;

    // The current statistical weight W^(α)(C; β) of the current configuration.
    // This is what we importance-sample. Configurations with larger weight are
    // visited more often. See Paper Eqs. 19–29.
    // ExExFloat is an extended-precision float (mantissa × 2^exponent) used
    // because weights can span hundreds of orders of magnitude [OPTIMIZATION].
    ExExFloat currWeight;

    // q = number of operators in the current sequence S_{i_q} = P_{i_q}...P_{i_1}
    // This is the "path length" in the PMR series expansion (Paper Eq. 16).
    // q=0 means the identity path (no quantum fluctuations). q fluctuates during
    // the simulation as operators are inserted and deleted.
    int q;

    // Sq[0..q-1] = the operator indices of the current sequence.
    // Each Sq[i] is an integer 0..Nop-1 labeling which permutation operator
    // was applied at step i. (Paper Eq. 16: the i_q sequence)
    int Sq[qmax];

    // The diagonal energy E(z) = ⟨z|D_0|z⟩ of the current spin configuration.
    // This is the "classical" part of the energy — no quantum fluctuations.
    double currEnergy;

    // beta_pow_factorial[k] = (-β)^k / k!
    // These coefficients appear in the PMR series expansion weight (Paper Eq. 16).
    // Precomputing them once per β-change avoids repeated pow() and factorial
    // calls during weight evaluations [OPTIMIZATION].
    ExExFloat beta_pow_factorial[qmax];

    //=========================================================================
    // RÉNYI-SPECIFIC PARAMETERS  (Paper Section II.E)
    //=========================================================================

    // α = Rényi order = number of independent B-copies in this replica object.
    // For α=2: two B-configurations. For α=3: three B-configurations.
    // (Paper Eqs. 24–29: the number of "cuts" is α-1)
    int alpha;

    // [OVERLAP] Which Hamiltonian (0 = H1, 1 = H2 in the OV_* tables) copy r
    // uses. Copy 0 holds segment [0,c) of the sequence -- the operator that acts
    // FIRST -- and copy 1 holds [c,q). Set from g_ov_ham[] in the constructor.
    std::vector<int> ham_of_copy;

#ifdef OVERLAP_MODE
    // [OVERLAP fix] Trace mode: hopping-amplitude prefix/suffix products in extended-exponent
    // arithmetic. The inherited std::complex<double> products underflow to exactly 0 for long
    // walks (e.g. |d|^q < 1e-308), and an exact 0 is not caught by the prefix_unsafe guard.
    // Overlap tables have real amplitudes, so real ExExFloat products are exact here.
    std::vector<ExExFloat> ov_tp0, ov_ts1;
#endif

#ifdef OVERLAP_PLUS
    // ═════════════════════════════════════════════════════════════════════
    // [OVERLAP_PLUS] |+> trial state, open walks, fixed projection beta
    // (= the Beta macro), ladder over the interpolation path of
    // gsOverlap v18 Sec. III.F:  (0,0) -> (0,1) -> (1,1).
    //
    // Here "beta" (this->beta, the rung coordinate everywhere in the driver)
    // is s in [0,1]; rung s means the pair (lam_a, lam_b) of OvLambdas().
    // The weight is Eq. (W) with H(lam) = (1-lam) H1 + lam H2 in PMR form
    // (energies and hopping amplitudes interpolate linearly), segment
    // [0,c) at lam_b (acts first), segment [c,q) at lam_a, each with
    // projection tau = Beta/2, summed over the cut c, for an OPEN walk
    // starting at y0 = lattice[0] (sampled). The |+> endpoint factor
    // phi(y_q) phi(y0) = 1/dim is common to every rung and dropped.
    // ═════════════════════════════════════════════════════════════════════
    std::vector<double> ovE[2], ovD[2];   // energies at positions 0..q, d at ops 0..q-1, for H1/H2
    bool ov_valid = false;
    std::vector<ExExFloat> ov_bpf;          // (-tau)^k/k!
    int ov_bpf_len = 0;
    // Half-weights of the current walk: prefix pre[c] = segment [0,c) at lam_b,
    // suffix suf[c] = segment [c,q) at lam_a. Slot 0 caches lam = 0 (pure H1), slot 1
    // lam = 1 (pure H2), slot 2 is scratch for interior lam. Leg 1 of the ladder shares
    // the H1 suffix, leg 2 the H2 prefix, so evaluating all rungs costs about K+2 passes
    // instead of 2K. A pass stops at the first exactly-zero hop (only possible at a
    // removed bond of pure H2): every remaining half-weight is zero.
    // pre[c] is valid for c <= ov_pre_hi[slot] (zero beyond); suf[c] for c >= ov_suf_lo[slot].
    std::vector<ExExFloat> ov_pre[3], ov_suf[3];
    int ov_pre_hi[3] = {0, 0, 0}, ov_suf_lo[3] = {0, 0, 0};
    bool ov_pre_ok[2] = {false, false}, ov_suf_ok[2] = {false, false};

    static void OvLambdas(double s_val, double& lam_a, double& lam_b) {
        if (s_val <= 0.5) { lam_a = 0.0; lam_b = 2.0 * s_val; }
        else              { lam_a = 2.0 * s_val - 1.0; lam_b = 1.0; }
    }
    static double OvEnergyH(int h, const std::bitset<N>& cfg) {
        double e = 0.0;
        for (int k = 0; k < OV_D0_size[h]; k++)
            e -= (2 * (int)((OV_D0_product[h][k] & (~cfg)).count() % 2) - 1) * OV_D0_coeff[h][k];  // int: count() is unsigned
        return e;
    }
    static double OvDH(int h, int op, const std::bitset<N>& cfg) {
        double d = 0.0;
        for (int k = 0; k < OV_D_size[h][op]; k++)
            d -= (2 * (int)((OV_D_product[h][op][k] & (~cfg)).count() % 2) - 1) * OV_D_coeff[h][op][k];
        return d;
    }
    void OvPrecompute() {
        for (int h = 0; h < 2; h++) { ovE[h].resize(q + 1); ovD[h].resize(q > 0 ? q : 1); }
        std::bitset<N> cfg = lattice[0];
        for (int h = 0; h < 2; h++) ovE[h][0] = OvEnergyH(h, cfg);
        for (int i = 0; i < q; i++) {
            cfg ^= P_matrix[Sq[i]];   // d evaluated at the post-hop state (calc_d_for_config convention)
            for (int h = 0; h < 2; h++) {
                ovD[h][i] = OvDH(h, Sq[i], cfg);
                ovE[h][i + 1] = OvEnergyH(h, cfg);
            }
        }
        ov_valid = true;
        ov_pre_ok[0] = ov_pre_ok[1] = ov_suf_ok[0] = ov_suf_ok[1] = false;
    }
    static int OvSlot(double lam) { return lam == 0.0 ? 0 : (lam == 1.0 ? 1 : 2); }
    double OvEat(double lam, int i) const { return (1.0 - lam) * ovE[0][i] + lam * ovE[1][i]; }
    double OvDat(double lam, int i) const { return (1.0 - lam) * ovD[0][i] + lam * ovD[1][i]; }
    // Prefix pass: segment [0,c) at lam, including its hopping product.
    void OvPrefix(double lam, double tau, int slot) {
        std::vector<ExExFloat>& w = ov_pre[slot];
        if ((int)w.size() < qmax + 1) w.resize(qmax + 1);
        divdiff& d0 = divdiff_replicas[0];
        d0.CurrentLength = 0;
        ExExFloat P(1.0);
        int c = 0;
        for (; c <= q; c++) {
            if (c > 0) { const double dh = OvDat(lam, c - 1); if (dh == 0.0) break; P *= dh; }
            d0.AddElement(-tau * OvEat(lam, c));
            w[c] = d0.divdiffs[c] * ov_bpf[c] * P;
        }
        ov_pre_hi[slot] = c - 1;
    }
    // Suffix pass: segment [c,q) at lam, built right to left (divided differences are
    // symmetric in their nodes).
    void OvSuffix(double lam, double tau, int slot) {
        std::vector<ExExFloat>& w = ov_suf[slot];
        if ((int)w.size() < qmax + 1) w.resize(qmax + 1);
        divdiff& d1 = divdiff_replicas[1];
        d1.CurrentLength = 0;
        ExExFloat S(1.0);
        int c = q;
        for (; c >= 0; c--) {
            if (c < q) { const double dh = OvDat(lam, c); if (dh == 0.0) break; S *= dh; }
            d1.AddElement(-tau * OvEat(lam, c));
            w[c] = (d1.divdiffs[q - c] * ov_bpf[q - c]) * S;
        }
        ov_suf_lo[slot] = c + 1;
    }
    ExExFloat OvWeight(double s_val) {
        if (!ov_valid) OvPrecompute();
        double la, lb; OvLambdas(s_val, la, lb);
        const double tau = 0.5 * (double)(Beta);
        if ((int)ov_bpf.size() < qmax + 1) { ov_bpf.resize(qmax + 1); ov_bpf_len = 0; }
        if (ov_bpf_len == 0) { ov_bpf[0] = ExExFloat(1.0); ov_bpf_len = 1; }
        while (ov_bpf_len <= q) { ov_bpf[ov_bpf_len] = ov_bpf[ov_bpf_len - 1] * ((-tau) / ov_bpf_len); ov_bpf_len++; }
        const int sp = OvSlot(lb), ss = OvSlot(la);
        if (sp == 2 || !ov_pre_ok[sp]) { OvPrefix(lb, tau, sp); if (sp < 2) ov_pre_ok[sp] = true; }
        if (ss == 2 || !ov_suf_ok[ss]) { OvSuffix(la, tau, ss); if (ss < 2) ov_suf_ok[ss] = true; }
        std::vector<ExExFloat>& pre = ov_pre[sp];
        std::vector<ExExFloat>& suf = ov_suf[ss];
        ExExFloat total(0.0);
        for (int c = ov_pre_hi[sp]; c >= ov_suf_lo[ss]; c--) total += pre[c] * suf[c];
        return total;
    }
#endif

    // Bitmasks indicating which qubit indices belong to subsystem A or B.
    // A '1' bit at position i means qubit i is in that subsystem.
    // These come from SUBSYSTEM_A_MASK and SUBSYSTEM_B_MASK in parameters.hpp.
    std::bitset<N> subsystem_A_mask;
    std::bitset<N> subsystem_B_mask;

    // Precomputed projections of each permutation operator onto subsystems A and B.
    // P_subsystem_A[k] = P_matrix[k] & subsystem_A_mask
    // Instead of applying the full operator and then masking, we can directly
    // apply just the A-part or B-part. Used in closure checks [OPTIMIZATION].
    std::bitset<N> P_subsystem_A[Nop];
    std::bitset<N> P_subsystem_B[Nop];

    //=========================================================================
    // REPLICA CONFIGURATIONS  (Paper Eqs. 17–18, 24)
    //
    // This is the crucial data structure for the Rényi calculation.
    // There are α copies of the system configuration, distinguished by their
    // subsystem-B part. Subsystem A is SHARED across all copies.
    //
    // lattice[r] = current spin configuration of replica r.
    //   For all r: (lattice[r] & mask_A) must be identical — shared zA.
    //   For each r: (lattice[r] & mask_B) = zB^r, independent per replica.
    //
    // z[r] = "reference configuration" = the configuration at the start of
    //   the operator sequence (the |z_r⟩ in Paper Eq. 24). After each
    //   accepted MC move, z[r] is updated to match lattice[r].
    //
    // lattice_backup[r] = saved copy for MC rejection. If a proposed move
    //   is rejected, we restore lattice[r] from lattice_backup[r].
    //=========================================================================
    std::vector<std::bitset<N>> lattice;        // Current configs [replica r]
    std::vector<std::bitset<N>> z;              // Reference configs [replica r]
    std::vector<std::bitset<N>> lattice_backup; // Backup for MC rejection

    //=========================================================================
    // PRECOMPUTED QUANTITIES PER REPLICA  [OPTIMIZATION — not in paper]
    //
    // For each replica r, we precompute the energies and D-coefficients at
    // every position along the operator sequence. This is computed once per
    // configuration and reused when evaluating contributions from all
    // possible cut positions.
    //
    // Energies_replicas[r][i] = E(z_r^{(i)}) = diagonal energy of replica r
    //   after applying the first i operators. (Paper Eq. 16: E_{z_j})
    //
    // D_replicas[r][i] = d_{Sq[i]}(z_r^{(i)}) = the diagonal matrix element
    //   of the i-th operator for replica r. (Paper Eq. 16: the D factor)
    //
    // divdiff_replicas[r] = divided-difference object for replica r.
    //   This computes the divided-difference sum for the exp(-β·H) expansion
    //   on each segment. (See divdiff.hpp; implements Paper Eq. 16 weight.)
    //=========================================================================
    std::vector<std::vector<double>> Energies_replicas;
    std::vector<std::vector<std::complex<double>>> D_replicas;
    std::vector<divdiff> divdiff_replicas;

    //=========================================================================
    // CYCLE COMPLETION UPDATE VARIABLES  (Paper Ref. [22,23])
    //
    // These are scratch variables used by the cycle-completion Monte Carlo
    // move (see update() method). The cycle-completion move replaces a
    // subsequence of operators with a complementary subsequence from the
    // same algebraic cycle, preserving ergodicity of the Markov chain.
    // [OPTIMIZATION — this move is not described in the current paper but
    // is inherited from the base PMR-QMC framework, Ref. [22,23]]
    //=========================================================================
    int cycles_used[Ncycles];
    int Sq_backup[qmax];
    int Sq_subseq[qmax];
    int Sq_gaps[qmax];
    int min_index, max_index, found_cycles;
    std::bitset<Nop> P;

    //=========================================================================
    // PER-INSTANCE RANDOM NUMBER GENERATOR  [OPTIMIZATION — not in paper]
    //
    // Each RenyiReplica has its OWN independent random number generator,
    // seeded uniquely from its β value and memory address.
    //
    // WHY SEPARATE RNGs?
    //   When multiple replicas run simultaneously with OpenMP parallelism,
    //   they all call update() at the same time. If they shared a single
    //   global RNG, they would conflict (race condition = wrong random numbers
    //   + crashes). Each replica having its own RNG eliminates this problem
    //   without needing locks or synchronization.
    //=========================================================================
    std::mt19937 local_rng;
    std::uniform_real_distribution<double> local_val{0.0, 1.0};   // random float in [0,1)
    std::uniform_int_distribution<int>     local_diceN{0, N-1};   // random qubit index
    std::uniform_int_distribution<int>     local_diceNop{0, Nop-1}; // random operator index
    std::uniform_int_distribution<int>     local_dice2{0, 1};     // random bit flip
    std::geometric_distribution<int>       local_geometric{GAPS_GEOMETRIC_PARAMETER}; // random gap length

    //=========================================================================
    // STATISTICS  [OPTIMIZATION — diagnostic tracking]
    //=========================================================================
    int qmax_achieved;            // Flag: did the sequence ever hit the length cap?
    int TstepsFin;                // Flag: has thermalization completed?
    unsigned long long stp;       // Total steps taken (for progress tracking)
    unsigned long long m_step;    // Total measurements taken

    // [DIAG swap-fallback] How often PartialUpdateAfterSwap's cheap O(alpha)
    // incremental path is skipped in favour of the expensive O(alpha*q) rebuild
    // (window_zero_count_changed branch, added by the D_zero_prefix-drift
    // fix). Not reset across thermalization/sampling -- callers diff two
    // snapshots to get a per-block rate. See scan_swap_fallback_stats() in
    // mbar_common.hpp.
    unsigned long long swap_total_count    = 0;
    unsigned long long swap_fallback_count = 0;

    // [DIAG lattice] Move-acceptance counters (attempts that reached the
    // Metropolis test) for the three move classes of update(), and a
    // parallel-tempering round-trip label that travels WITH the configuration
    // (swapped in SwapConfigWith / exchanged in mpi_boundary_swap):
    //   -1 = this configuration has not yet touched either end of the ladder,
    //    0 = it last touched the bottom rung (beta_min),
    //    1 = it last touched the top rung (beta_target).
    // The drivers count a "round trip" every time a label-1 configuration
    // arrives at the bottom rung.  See fable_audit/BIAS_AUDIT_REPORT.md §5.
    unsigned long long flip_attempts = 0,      flip_accepts = 0;
    unsigned long long cutflip_attempts = 0,   cutflip_accepts = 0;   // [OV_CUT_FLIP]
    unsigned long long block_attempts = 0,     block_accepts = 0;
    unsigned long long composite_attempts = 0, composite_accepts = 0;
    int pt_label = -1;

    //=========================================================================
    // CONSTRUCTOR  (Paper Section II.E — sets up α-replica structure)
    //
    // Allocates storage for α replicas of the system, precomputes the
    // subsystem masks and operator projections, and initializes all scratch
    // arrays. No actual QMC initialization happens here — call init_renyi()
    // separately after setting up the global QMC infrastructure.
    //=========================================================================
    RenyiReplica(int renyi_index = 2)
        : alpha(renyi_index),
          energies_valid(false),
          prefix_cache_valid(false),
          a_closure_valid(false),         // [OPT 7]
          config_dirty(true),             // [OPT A] starts dirty — no caches built yet
          beta_dirty(false),              // [OPT A] beta_pow_factorial is not yet needed
          beta_pow_computed(0),           // [OPT 11]
          q(0),
          currEnergy(0.0),
          qmax_achieved(0),
          TstepsFin(0),
          stp(0),
          m_step(0),
          found_cycles(0),
          min_index(0),
          max_index(0) {
        
        ham_of_copy.assign(alpha, 0);
#ifdef OVERLAP_MODE
        for(int r = 0; r < alpha; r++) ham_of_copy[r] = g_ov_ham[r < 2 ? r : 1];
#endif
        subsystem_A_mask = std::bitset<N>(SUBSYSTEM_A_MASK);
        subsystem_B_mask = std::bitset<N>(SUBSYSTEM_B_MASK);
        
        if((subsystem_A_mask & subsystem_B_mask).any()) {
            std::cerr << "FATAL ERROR: Subsystem A and B masks overlap!" << std::endl;
            exit(1);
        }

        // ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────
        // Physics: ergodicity -- every valid configuration must be reachable
        // by the sampler, or the long-run average it computes isn't a true
        // thermal average, it's an average over whatever smaller subspace
        // the chain got trapped in.
        // Plain-English: each qubit needs to belong to subsystem A or B so
        // it actually gets flipped by SOME move. The code checked that A and
        // B don't overlap, but never checked that together they cover every
        // qubit -- a "gap" qubit belonging to neither would sit frozen
        // forever, silently making some configurations unreachable. Now a
        // hard startup error instead of a silent trap (harmless today only
        // because the current masks happen to cover all 9 qubits).
        // ─────────────────────────────────────────────────────────────────
        // [F19 fix] A qubit belonging to neither mask is never flipped by any
        // A-spin or B-spin move (replica.hpp's spin-flip branch only iterates
        // over masked qubits), silently making the chain non-ergodic on that
        // qubit. Only overlap was checked before; coverage was not.
        if((subsystem_A_mask | subsystem_B_mask).count() != (size_t)N) {
            std::cerr << "FATAL ERROR: Subsystem A and B masks do not cover all "
                       << N << " qubits (gap qubits would never be sampled)!" << std::endl;
            exit(1);
        }

        // Allocate storage for α replicas
        lattice.resize(alpha);
        z.resize(alpha);
        lattice_backup.resize(alpha);
        Energies_replicas.resize(alpha);
        D_replicas.resize(alpha);
        B_prefix_cache.resize(alpha);
        D_zero_prefix.resize(alpha);          // [OPT 15]
        positions_in_class.resize(alpha);     // [OPT 1] B+D combined
        b_class_positions.resize(alpha);      // [OPT 1] B-only (for HasAnyValidCuts)
        scratch_old_pic_keys.resize(2 * alpha, 0);  // [OPT 1] incremental update buffer

        for(int r = 0; r < alpha; r++) {
            divdiff_replicas.push_back(divdiff(qmax, 500));
            Energies_replicas[r].resize(qmax + 1);
            D_replicas[r].resize(qmax);
            B_prefix_cache[r].resize(qmax + 1);
            D_zero_prefix[r].assign(qmax + 1, 0);  // [OPT 15]
        }

        // [OPT-3] alpha=3 scratch — allocated once regardless of alpha (cheap:
        // O(qmax) ExExFloat/complex entries), used only when alpha==3.
        w2_all.resize(qmax + 1);
        D2_suf.resize(qmax + 1);
        Pnz0.resize(qmax + 1);
        Pnz1.resize(qmax + 1);
        w0_cache.resize(qmax + 1);  // [OPT-4]
        PXOR.resize(qmax + 1);      // [OPT-12]

        // [OPT E / 4.1] Reserve scratch vectors to their maximum needed sizes.
        // The vectors are now member-owned; no allocation happens inside the hot loops.
        scratch_seg_starts.resize(alpha);
        scratch_seg_ends.resize(alpha);
        scratch_cuts.resize(std::max(1, alpha - 1));
        scratch_valid_j2.reserve(qmax + 1);  // [OPT F / 4.2]

        // [OPT 7] Start with zero A-net-permutation (empty sequence)
        A_net_permutation.reset();
        
        for(int k = 0; k < Nop; k++) {
            P_subsystem_A[k] = P_matrix[k] & subsystem_A_mask;
            P_subsystem_B[k] = P_matrix[k] & subsystem_B_mask;
        }
        
        for(int i = 0; i < Ncycles; i++) {
            cycles_used[i] = 0;
        }
        
        q = 0;
        for(int i = 0; i < qmax; i++) {
            Sq[i] = 0;
            Sq_backup[i] = 0;
            Sq_subseq[i] = 0;
            Sq_gaps[i] = 0;
        }
    }
    
    //=========================================================================
    // INITIALIZATION  (Paper Section II.F, Eq. 10)
    //
    // Sets the β value and draws a random starting configuration.
    // Must be called after divdiff_init() and init_rng() have been run globally.
    //=========================================================================
    void init_renyi(double beta_value, int replica_index = 0) {
        beta = beta_value;

        // ── STUDY NOTE [F13 fix landed 2026-07-09 · reviewed 2026-09-10] ────
        // Physics: doesn't touch physics at all -- this is purely about
        // whether EXACTLY_REPRODUCIBLE actually reproduces anything.
        // Plain-English: the old code mixed the replica OBJECT'S MEMORY
        // ADDRESS into its random seed to make replicas at the same beta
        // distinct. But a memory address depends on things like ASLR and
        // heap layout, which are NOT the same from run to run even with the
        // same seed -- so "reproducible" runs weren't actually reproducible.
        // Now it uses the replica's fixed ladder slot number instead, which
        // never changes run to run, so the same seed really does replay the
        // exact same trajectory (useful for debugging, changes no physics).
        // ─────────────────────────────────────────────────────────────────

        // Seed this replica's private RNG from three sources XOR'd together:
        //   hash(beta)     — differs between rungs on the same ladder
        //   replica_index  — the caller's globally-unique slot index (ladder rung
        //                    number), so replicas at the same beta in the same run
        //                    still get distinct seeds. Deterministic, unlike the
        //                    object's memory address: under EXACTLY_REPRODUCIBLE
        //                    the address varies run-to-run (ASLR/heap layout) even
        //                    with an identical rng_seed, silently breaking the
        //                    reproducibility the flag promises. [F13 fix]
        //   rng_seed       — process-level entropy set once by init_rng() at startup;
        //                    differs between independent runs (when !EXACTLY_REPRODUCIBLE)
        //                    so that re-running the same binary produces genuinely
        //                    different trajectories rather than identical ones.
        local_rng.seed(static_cast<unsigned>(std::hash<double>{}(beta_value))
                       ^ static_cast<unsigned>(replica_index) * 2654435769u   // 2^32/golden ratio (standard hash multiplier)
                       ^ rng_seed);

        // ── CRITICAL: zA is SHARED across all α replicas  (Paper Eq. 24) ────
        //
        // Paper Eq. 24: Z_2(β) = Σ_{z_A, zB, zB'} Σ_{i_q} W^(2)(...) ⟨z_A|S^(A)_{i_q}|z_A⟩
        //
        // There is ONE z_A in the formula. The whole quantum path on subsystem A
        // must start and end at the same state z_A. This means ALL α copies of
        // the system share the same A-configuration at all times.
        //
        // IMPORTANT BUG FIX (documented here for clarity):
        //   An earlier version of this code randomly initialized each replica's
        //   A-bits independently. This meant replica 0 had z_A^0 and replica 1
        //   had z_A^1 ≠ z_A^0. Every MC move that flips A-spins applies the same
        //   flip to both replicas. So (z_A^0 XOR z_A^1) is permanently CONSERVED
        //   — the sampler is forever locked in the wrong subspace and can never
        //   reach configurations where z_A^0 = z_A^1 (the only valid ones).
        //   Fix: generate ONE shared z_A and assign it to all replicas.
        std::bitset<N> zA_shared;
        for(int i = N - 1; i >= 0; i--) {
            if(subsystem_A_mask.test(i) && local_dice2(local_rng)) zA_shared.set(i);
        }

        // ── Each replica gets its OWN independent zB^r  (Paper Eq. 24) ──────
        //
        // In Paper Eq. 24, the sum Σ_{z_A, zB, zB'} is over independent B-states.
        // zB^0, zB^1, ..., zB^{α-1} are each sampled from their own distribution.
        for(int r = 0; r < alpha; r++) {
            std::bitset<N> zB_r;
            for(int i = N - 1; i >= 0; i--) {
                if(subsystem_B_mask.test(i) && local_dice2(local_rng)) zB_r.set(i);
            }
            // Combine shared A-part with independent B-part for this replica
            lattice[r] = zA_shared | zB_r;
            z[r] = lattice[r];  // reference config = current config at start
        }

#if defined(OVERLAP_PLUS) && OV_INIT_CLASSICAL
        // [OV_INIT_CLASSICAL] Anneal y0 on H1's diagonal energy (Metropolis, T from 2|E|/N
        // down to ~0, zero-cost flips accepted so 1D walls diffuse and annihilate), then
        // copy to every replica (OVERLAP_PLUS: all qubits shared).
        {
            std::bitset<N> cfg = lattice[0];
            double e = OvEnergyH(0, cfg);
            const int sweeps = 200 * N;
            double T = 2.0 * std::max(1.0, std::fabs(e)) / N;
            const double cool = std::pow(1e-3, 1.0 / sweeps);
            for(int sw = 0; sw < sweeps; sw++, T *= cool) {
                for(int k = 0; k < N; k++) {
                    const int i = local_diceN(local_rng);
                    cfg.flip(i);
                    const double e2 = OvEnergyH(0, cfg);
                    if(e2 <= e || local_val(local_rng) < std::exp(-(e2 - e) / T)) e = e2;
                    else cfg.flip(i);
                }
            }
            for(int r = 0; r < alpha; r++) { lattice[r] = cfg; z[r] = cfg; }
        }
#endif
        // Start with no operators in the sequence (identity path)
        q = 0;

        // [OPT 11] Compute only beta_pow_factorial[0] at init (q=0 requires only index 0).
        // Additional entries are computed lazily in UpdateWeight() as q grows.
        // This avoids computing qmax entries upfront when only 1 is needed.
        ExExFloat curr(1.0);
        beta_pow_factorial[0] = curr;
        beta_pow_computed = 1;  // [OPT 11]

        // [OPT 7] Initialize A-net-permutation for empty sequence (identity)
        A_net_permutation.reset();
        a_closure_valid = false;  // will be set true by first CheckSubsystemAClosure()

        // Compute the initial weight W^(α)(C_0; β) for the starting configuration
        InvalidateCache();
        currWeight = GetRenyiWeight(beta, beta_pow_factorial);

        if(energies_valid) {
            currEnergy = Energies_replicas[0][0];
        }
    }
    
    //=========================================================================
    // SET BETA  (Paper Section II.C — used for PT swaps and MBAR reweighting)
    //
    // Changes the temperature of this replica to a new β value and immediately
    // recomputes the current weight at the new temperature.
    //
    // This function is called in TWO contexts:
    //
    // 1. PARALLEL TEMPERING SWAPS (Section II.C):
    //    When two neighboring replicas attempt to exchange configurations,
    //    each replica calls set_beta() with the OTHER's temperature to check
    //    whether the swap should be accepted (see attempt_pt_swap() in mbarpt.cpp).
    //    If the swap is rejected, set_beta() is called again to restore the
    //    original temperature.
    //
    // 2. MBAR REWEIGHTING (Section II.C, Eq. 11):
    //    After collecting a sample, we call set_beta(β_j) for every rung j
    //    to evaluate u[s][j] = -ln W^(α)(C; β_j). The configuration C stays
    //    fixed; only β changes. (See mbarpt.cpp sampling loop.)
    //=========================================================================
    void set_beta(double beta_value) {
        beta = beta_value;

        // [OPT 11] Recompute (-β)^k / k! only for k=0..q (not all qmax).
        // During MBAR reweighting this is called K=32 times per sample with the
        // same frozen configuration (q unchanged), so we only need entries up to q.
        // At scale: saves (qmax - q - 1) ExExFloat multiplications per call.
        // Example: qmax=100, q=30 → saves 69 ExExFloat mults × 100M calls = 6.9B ops.
        ExExFloat curr(1.0);
        beta_pow_factorial[0] = curr;
        {
            int cap = q + 1;
            if(cap > qmax) cap = qmax;
            for(int k = 1; k < cap; k++) {
                curr *= (-beta) / k;
                beta_pow_factorial[k] = curr;
            }
            beta_pow_computed = cap;  // [OPT 11]
        }
        beta_dirty = false;  // [OPT A] factorial rebuilt for current beta

        // Recompute the weight at the new β.
        //
        // [OPT 1] No InvalidateCache() here.
        //
        // The caches (Energies_replicas, D_replicas, B_prefix_cache) depend only
        // on lattice[] and Sq[] — NOT on β.  set_beta() is always called with a
        // frozen configuration:
        //
        //   • PT swap:        config stays put while β is probed at the neighbor's value
        //   • MBAR reweighting: inner loop sweeps all K β-values over the same snapshot
        //
        // In both cases the last update() (or init_renyi()) already rebuilt the
        // caches and left energies_valid = prefix_cache_valid = true.  Only
        // beta_pow_factorial (rebuilt above) differs between calls.
        //
        // GetRenyiWeight() therefore goes straight to EnumerateAndSumCuts(),
        // skipping PrecomputeEnergiesAndDCoefficients() and
        // ComputePrefixBPermutations() entirely.
        //
        // Savings: K × N_SAMPLES × K = 32 × 100,000 × 32 ≈ 100 M calls during
        // sampling, each avoiding O(α × q × D0_size) ≈ 1,200 ops → ~3.8 B ops
        // eliminated in the sampling phase alone.
        currWeight = GetRenyiWeight(beta, beta_pow_factorial);

        if(energies_valid) {
            currEnergy = Energies_replicas[0][0];
        }
    }

    //=========================================================================
    // CACHE MANAGEMENT  [OPTIMIZATION — not in paper]
    //
    // After any change to the operator sequence Sq[] or spin configurations
    // lattice[], the precomputed energies and B-prefix values are stale.
    // We call InvalidateCache() to mark them so they get recomputed on the
    // next call to GetRenyiWeight(). This "lazy evaluation" avoids wasting
    // time recomputing caches that might be thrown away if the MC move is
    // ultimately rejected.
    //
    // NOTE: set_beta() intentionally does NOT call InvalidateCache() because
    // it is only ever called with a frozen configuration — see [OPT 1] above.
    //=========================================================================
    void InvalidateCache() {
#ifdef OVERLAP_PLUS
        ov_valid = false;
#endif
        energies_valid = false;
        prefix_cache_valid = false;
        a_closure_valid = false;  // [OPT 7] Sq may have changed — rebuild on next check
        config_dirty = true;      // [OPT A] coarse flag for callers that only need to know
        tuples_valid = false;     // [OPT-1] Pnz0/Pnz1/D2_suf are stale until EnumerateTuples() reruns
    }

    // [OPT A] Explicit config invalidator — same as InvalidateCache(), exists so
    // callers in update() can express intent clearly.  set_beta() should NOT call
    // this because a beta change leaves lattice[] and Sq[] untouched.
    void InvalidateConfig() {
        InvalidateCache();  // sets all fine-grained flags + config_dirty
    }

    // [OPT A] Mark beta_pow_factorial as stale without touching config caches.
    // Called conceptually when beta changes but factorial hasn't been rebuilt yet.
    // set_beta() clears this after rebuilding.  Used by Optimization B (PT swap
    // fast reject) to avoid redundant work — not yet wired up actively.
    void InvalidateBeta() {
        beta_dirty = true;
    }

    //=========================================================================
    // SwapConfigWith()  [F17 fix — Paper Section II.C, parallel tempering]
    //
    // Swaps the PHYSICAL CONFIGURATION (lattice, reference configs, operator
    // sequence, energy) between this replica and other, WITHOUT touching beta,
    // beta_pow_factorial, or local_rng — those belong to the ladder SLOT, not
    // to the configuration that currently occupies it.
    //
    // WHY: the original PT swap exchanged beta between two replica OBJECTS,
    // leaving each object's configuration in place. After the first accepted
    // swap, "adjacent object slots" no longer means "adjacent temperatures" —
    // slot k's beta could be anything, so pairing do_pt_swap(k, k+1) by raw
    // array index silently degrades into near-random-pair tempering instead of
    // the intended even/odd adjacent-rung ladder (MBARPT_audit F17). Swapping
    // the CONFIGURATION instead of the temperature makes slot k ≡ rung k a
    // permanent invariant: no rung-tracking bookkeeping is needed anywhere,
    // and this generalizes cleanly to the MPI driver's cross-rank boundary
    // swaps (a bounded config message over MPI_Sendrecv, no global state).
    //
    // Caches are left invalid on both replicas (InvalidateCache()) and rebuild
    // lazily on the next update()/weight call — exactly like an ordinary
    // rejected MC move already does. currWeight is NOT touched here; the
    // caller (attempt_pt_swap) assigns it explicitly, since the two weight
    // values needed post-swap were already computed by the cross-probe
    // set_beta() calls before this is invoked.
    //=========================================================================
    // ── STUDY NOTE [F17 fix landed 2026-07-09 · reviewed 2026-09-10] ────────
    // Full physics/plain-English writeup is at attempt_pt_swap() in
    // mbar_common.hpp (the caller). Short version: this is the piece that
    // physically moves lattice/z/Sq/q/energy between the two replica
    // objects on an accepted swap, instead of the old behavior of leaving
    // configs in place and swapping which TEMPERATURE each object held --
    // that old behavior is what let "adjacent slot" stop meaning "adjacent
    // temperature" after the first accepted swap anywhere on the ladder.
    // ─────────────────────────────────────────────────────────────────────
    void SwapConfigWith(RenyiReplica& other) {
        lattice.swap(other.lattice);
        z.swap(other.z);
        std::swap(q, other.q);
        std::swap(Sq, other.Sq);
        std::swap(currEnergy, other.currEnergy);
        std::swap(pt_label, other.pt_label);   // [DIAG lattice] label follows the configuration
        InvalidateCache();
        other.InvalidateCache();
    }

    //=========================================================================
    // SaveState() / LoadState()  [F20 fix — checkpointing]
    //
    // Serializes exactly the state that fully determines this replica's future
    // trajectory: q/Sq/lattice/z/currEnergy (the physical configuration and
    // operator sequence) and local_rng (so a resumed trajectory is a genuine
    // continuation, not a re-seeded restart). Deliberately NOT saved:
    // currWeight, beta_pow_factorial, and all internal caches (Energies_
    // replicas, D_replicas, divdiff_replicas, prefix caches) -- these are pure
    // functions of the saved state plus beta, and the caller MUST call
    // set_beta(beta) immediately after LoadState() to rebuild them (same
    // lazy-rebuild pattern already used after SwapConfigWith or a rejected MC
    // move). beta/alpha are written too, purely as a corruption/mismatch
    // sanity check on load -- they come from the caller's init_renyi/
    // constructor and are not what LoadState restores control flow from.
    //
    // lattice[r]/z[r] are packed via to_ullong(), so this assumes N<=64 -- the
    // same assumption F17's MPI wire format already makes (mpi_boundary_swap).
    //=========================================================================
    // ── STUDY NOTE [F20 fix landed 2026-07-10 · reviewed 2026-09-10] ────────
    // Full writeup is at g_checkpoint_requested in mbar_common.hpp. Short
    // version: this is the "what to save" half of checkpointing -- exactly
    // the state that determines this replica's FUTURE trajectory (config +
    // RNG), deliberately NOT any of the derived caches, since those are pure
    // functions of this state plus beta and get rebuilt lazily on load.
    // ─────────────────────────────────────────────────────────────────────
    void SaveState(std::ofstream& out) const {
        out.write((const char*)&beta, sizeof(beta));
        out.write((const char*)&alpha, sizeof(alpha));
        out.write((const char*)&q, sizeof(q));
        out.write((const char*)Sq, sizeof(Sq));
        out.write((const char*)&currEnergy, sizeof(currEnergy));
        out.write((const char*)&qmax_achieved, sizeof(qmax_achieved));
        out.write((const char*)&local_rng, sizeof(local_rng));
        for (int r = 0; r < alpha; r++) {
            unsigned long long bits = lattice[r].to_ullong();
            out.write((const char*)&bits, sizeof(bits));
            bits = z[r].to_ullong();
            out.write((const char*)&bits, sizeof(bits));
        }
    }

    // Caller must have already constructed this replica with the correct
    // alpha (via the constructor) -- LoadState only restores mutable state, it
    // does not (re)size lattice[]/z[]. Caller MUST call set_beta(beta) right
    // after this returns, to rebuild beta_pow_factorial/currWeight/caches.
    void LoadState(std::ifstream& in) {
        double loaded_beta; int loaded_alpha;
        in.read((char*)&loaded_beta, sizeof(loaded_beta));
        in.read((char*)&loaded_alpha, sizeof(loaded_alpha));
        if (loaded_alpha != alpha) {
            std::cerr << "FATAL: checkpoint alpha mismatch (file has alpha=" << loaded_alpha
                      << ", this replica expects alpha=" << alpha << ") -- checkpoint is "
                         "corrupt or from a different run; delete it and restart.\n";
            exit(1);
        }
        beta = loaded_beta;
        in.read((char*)&q, sizeof(q));
        in.read((char*)Sq, sizeof(Sq));
        in.read((char*)&currEnergy, sizeof(currEnergy));
        in.read((char*)&qmax_achieved, sizeof(qmax_achieved));
        in.read((char*)&local_rng, sizeof(local_rng));
        for (int r = 0; r < alpha; r++) {
            unsigned long long bits;
            in.read((char*)&bits, sizeof(bits));
            lattice[r] = std::bitset<N>(bits);
            in.read((char*)&bits, sizeof(bits));
            z[r] = std::bitset<N>(bits);
        }
        InvalidateCache();
    }

    //=========================================================================
    // PartialUpdateAfterSwap()  [OPT 2 — not in paper]
    //
    // Called immediately after the adjacent-swap sub-move has written
    // Sq[m] ↔ Sq[m+1] into the sequence.  Updates only the three cache
    // entries that actually change, leaving all others intact.
    //
    // WHY ONLY THESE ENTRIES:
    //
    //   Before swap: ... state_m → [Sq[m]] → state_{m+1} → [Sq[m+1]] → state_{m+2} ...
    //   After  swap: ... state_m → [Sq[m+1]] → state_{m+1}' → [Sq[m]] → state_{m+2} ...
    //
    //   state_m      unchanged  (operators 0..m-1 are identical)
    //   state_{m+1}  CHANGES    (different operator applied at step m)
    //   state_{m+2}  unchanged  (XOR is commutative: A⊕B == B⊕A, so applying
    //                            both operators in either order gives the same state)
    //   state_{m+2..q} unchanged by propagation
    //
    //   Entries that change per replica r:
    //     D_replicas[r][m]          — same position, new Sq[m] + new state_{m+1}
    //     Energies_replicas[r][m+1] — energy of new state_{m+1}
    //     B_prefix_cache[r][m+1]    — B-part of new state_{m+1}
    //     D_replicas[r][m+1]        — new Sq[m+1], but state_{m+2} is unchanged
    //
    //   No "D_prefix" or "valid_cuts_from_zero" exist in this code — those are
    //   not cached here.  D_replicas[r][i] holds individual coefficients and
    //   ComputeCutContribution() multiplies them in a direct per-segment loop.
    //   Cut validity is checked on-the-fly by EnumerateAndSumCuts() via O(1)
    //   B_prefix_cache comparisons.  Nothing else needs rebuilding.
    //
    // Cost: O(α × (m + D0_size + D_maxsize))  ← walk + 2 energy/D updates
    //       vs O(α × q × D0_size) for a full rebuild.
    //       For a swap at position m=q/2 in a q=50 sequence: roughly 25× cheaper.
    //
    // Precondition: caches are valid on entry (always holds because update() only
    //   calls this from the adjacent-swap branch, after the previous sub-move left
    //   the caches valid — see [OPT 1] comment for the invariant argument).
    //=========================================================================
    void PartialUpdateAfterSwap(int m) {
#ifdef OVERLAP_PLUS
        ov_valid = false;
#endif
        // [OPT 4] Guard: partial update requires ALL cache entries to be valid,
        // since it only writes to positions m and m+1 and leaves everything else
        // untouched.  After lazy energy evaluation (GetRenyiWeight returned 0
        // early, skipping PrecomputeEnergiesAndDCoefficients), energies_valid
        // is false and the non-m entries contain stale values.  In that case,
        // fall back to full invalidation — the next GetRenyiWeight call will
        // rebuild from scratch (and may again skip energies via its own [OPT 4]
        // early exit if the swap also yields W=0).
        if(!energies_valid || !prefix_cache_valid) {
            InvalidateCache();
            return;
        }

        // [OPT 1] Save old pic-keys BEFORE any cache writes; enables O(alpha) incremental
        // PIC update below instead of an O(alpha*q) full rebuild.
        // pic_key upper-32 = B_prefix_cache, lower-32 = D_zero_prefix.
        for (int r = 0; r < alpha; r++) {
            scratch_old_pic_keys[r]         = pic_key(r, m + 1);
            scratch_old_pic_keys[alpha + r] = (m + 1 < q) ? pic_key(r, m + 2) : 0;
        }

        // [OPT-12] PXOR[0..m] are unaffected by the swap (only the entry at
        // m+1 depends on Sq[m], which just changed); PXOR[m+2..] are also
        // unaffected (XOR commutes -- the same two operators in swapped
        // order give the same running total, mirroring the state_{m+2}
        // invariance already noted below). Only PXOR[m+1] needs updating,
        // O(1), using the NEW (post-swap) Sq[m] -- shared across all
        // replicas, so this happens ONCE, not per replica.
        PXOR[m + 1] = PXOR[m] ^ P_matrix[Sq[m]];

        // ── STUDY NOTE [fix landed 2026-07-29 · reviewed 2026-09-10] ────────
        // Physics: this fix changed NOTHING about the physics formula itself
        // -- it's a bug in an internal performance shortcut (a cache) that
        // was silently feeding wrong inputs INTO an otherwise-correct
        // formula. The formula was never wrong; the numbers it was fed were.
        // Plain-English: the code keeps a running tally of "how many of
        // these factors are exactly zero" so it can quickly skip invalid
        // segments. After a swap move, it only patches the 2 positions
        // guaranteed to change, assuming the zero-COUNT in that small window
        // never shifts. That assumption isn't actually guaranteed -- and
        // when it silently failed, every zero-count downstream drifted out
        // of sync, one swap at a time, until ~97% of all alpha=3 samples
        // were being counted as exactly-zero-weight (impossible under
        // correct arithmetic, which is how this got caught). Fix below:
        // detect when the window's zero-count actually changes, and pay for
        // a cheap resync instead of trusting the shortcut blindly.
        // ─────────────────────────────────────────────────────────────────
        // [BUGFIX D_zero_prefix-drift] D_replicas[r][m]/[r][m+1] are matrix
        // elements evaluated at specific intermediate spin configurations
        // (calc_d_for_config(Sq[i], state_{i+1})). Swapping which operator
        // sits at m vs m+1 changes WHICH configuration each operator's d_k is
        // evaluated at (the operator now at m sees the state after only
        // itself; previously, if it was at m+1, it saw the state after BOTH
        // operators) -- there is no invariant guaranteeing the zero/nonzero
        // count within this 2-position window is preserved. When it isn't,
        // every D_zero_prefix[r][i] for i>m+2 -- and hence every pic_key
        // built from it -- silently goes stale, since only positions m+1/m+2
        // were ever patched below. This is undetectable locally (it's a
        // cumulative drift across many adjacent-swap sub-moves, worse for
        // longer q, i.e. worse at high beta) and was empirically confirmed
        // production-side: ~97-100% of samples' alpha=3 weight sums land on
        // an exact 0.0 mantissa, which is impossible under correct arithmetic
        // since every individual D0/D1/D2 factor entering EnumerateAndSumCuts
        // is guaranteed nonzero by the PIC selection itself (see fable audit,
        // job 474004 diagnostics). Fix: detect a window zero-count change and
        // pay for an O(q-m) D_zero_prefix suffix re-walk (cheap -- reuses the
        // already-correct D_replicas, no calc_d_for_config/energy recompute)
        // plus one BuildPositionsInClass() call, instead of silently drifting.
        // [DIAG swap-fallback] Count every call here; swap_fallback_count is
        // incremented below once we know whether this call took the
        // expensive branch.
        swap_total_count++;
        bool window_zero_count_changed = false;
        {
            const std::complex<double> zero_c(0.0, 0.0);
            for(int r = 0; r < alpha; r++) {
                bool old_zero_m  = (D_replicas[r][m]   == zero_c);
                bool old_zero_m1 = (D_replicas[r][m+1] == zero_c);

                // The swap Sq[m] ↔ Sq[m+1] has already been performed by the caller.
                // [OPT-12] state_m = lattice[r] ^ PXOR[m] -- O(1) via the shared
                // prefix table (was an O(m) per-replica walk applying operators
                // 0..m-1 one at a time).
                std::bitset<N> cfg = lattice[r] ^ PXOR[m];
                // cfg == state_m  (unchanged from before the swap)

                // Apply new Sq[m] (= old Sq[m+1]) → new state_{m+1}
                cfg ^= P_matrix[Sq[m]];

                // ── Update the three entries that depend on state_{m+1} ───────
                D_replicas[r][m]          = calc_d_for_config(r, Sq[m], cfg);
                Energies_replicas[r][m+1] = CalcCompositeEnergy(r, cfg);  // [OPT-14]
                B_prefix_cache[r][m+1]    = cfg & subsystem_B_mask;

                // Apply new Sq[m+1] (= old Sq[m]) → state_{m+2} (identical to before).
                // Sq[m+1] changed even though state_{m+2} didn't, so D[m+1] must update.
                cfg ^= P_matrix[Sq[m+1]];
                D_replicas[r][m+1] = calc_d_for_config(r, Sq[m+1], cfg);

                bool new_zero_m  = (D_replicas[r][m]   == zero_c);
                bool new_zero_m1 = (D_replicas[r][m+1] == zero_c);
                if ((old_zero_m ? 1 : 0) + (old_zero_m1 ? 1 : 0)
                    != (new_zero_m ? 1 : 0) + (new_zero_m1 ? 1 : 0))
                    window_zero_count_changed = true;
            }
        }
        // All other cache entries are unchanged and were already valid.
        energies_valid     = true;
        prefix_cache_valid = true;

        // [OPT-1] D_replicas/B_prefix_cache at m/m+1 just changed, so the
        // cached Pnz0/Pnz1/D2_suf (built from the OLD D_replicas) are stale.
        // Force a rebuild on the next EnumerateTuples() call -- this does NOT
        // regress the O(m)->O(1) win PartialUpdateAfterSwap exists for; it
        // only means the beta-independent tables are recomputed once at the
        // START of the next K-beta sweep, not for every individual MC step.
        tuples_valid = false;

        // [OPT 7] Adjacent swap does not change the net A-permutation (XOR commutes).
        // A_net_permutation = XOR of P_subsystem_A[Sq[i]] for all i; swapping two
        // elements leaves the total XOR unchanged.  So a_closure_valid stays true.
        a_closure_valid = true;

        // [OPT 15] Update D_zero_prefix for positions m+1 and m+2 incrementally.
        // D_replicas[r][m] and [r][m+1] just changed; all other positions are
        // unaffected (state_{m+2} is the same because XOR is commutative).
        // This preserves the invariant: D_zero_prefix valid ↔ energies_valid.
        for(int r = 0; r < alpha; r++) {
            const std::complex<double> zero_c(0.0, 0.0);
            bool dm_zero  = (D_replicas[r][m]   == zero_c);
            D_zero_prefix[r][m + 1] = D_zero_prefix[r][m] + (dm_zero ? 1 : 0);
            if(m + 1 < q) {
                bool dm1_zero = (D_replicas[r][m+1] == zero_c);
                D_zero_prefix[r][m + 2] = D_zero_prefix[r][m + 1] + (dm1_zero ? 1 : 0);

                // [BUGFIX D_zero_prefix-drift] The anchor D_zero_prefix[m+2] the
                // whole suffix [m+2,q] was built relative to may just have
                // shifted. Re-walk the suffix from the corrected anchor using
                // the already-correct (untouched) D_replicas[r][m+2..q-1] --
                // O(q-m), no calc_d_for_config/energy recompute.
                if (window_zero_count_changed) {
                    for(int i = m + 2; i < q; i++)
                        D_zero_prefix[r][i + 1] = D_zero_prefix[r][i]
                                                 + (D_replicas[r][i] == zero_c ? 1 : 0);
                }
            }
        }

        // [BUGFIX D_zero_prefix-drift] pic_key encodes D_zero_prefix, so a
        // suffix re-walk above invalidates positions_in_class[r] for every
        // replica beyond a simple two-position patch. Rebuild it fully here
        // (O(alpha*q), negligible at this scale) instead of the incremental
        // PicUpdatePosition path below, which assumes only m+1/m+2 moved.
        //
        // [BUGFIX b_class_positions-gap] This early return also skips the
        // incremental loop below -- which is where B_prefix_cache[r][m+1]'s
        // move to a new b_class_positions bucket normally gets applied via
        // BClassUpdatePosition. That move is UNCONDITIONAL (fires whenever
        // the two swapped operators' B-parts differ, independent of whether
        // the D-window flipped), so skipping it here would leave
        // b_class_positions stuck on the OLD bucket for m+1 -- corrupting
        // HasAnyValidCuts() (which reads b_class_positions, not
        // positions_in_class) via the exact same silent-zero-weight failure
        // mode this whole fix exists to close. Rebuild it too (same O(alpha*q)
        // order as BuildPositionsInClass(), no reason to special-case it).
        if (window_zero_count_changed) {
            swap_fallback_count++;   // [DIAG swap-fallback]
            BuildPositionsInClass();
            BuildBClassPositions();
            return;
        }

        // [OPT 1] Incremental PIC update: O(alpha) instead of O(alpha*q) full rebuild.
        // Position m+1: both B_prefix_cache and D_zero_prefix may have changed.
        // Position m+2: only D_zero_prefix may have changed (B is invariant — XOR commutes).
        // b_class_positions tracks B only — update only if B changed at m+1.
        for (int r = 0; r < alpha; r++) {
            pic_key_t new_key_m1 = pic_key(r, m + 1);
            if (new_key_m1 != scratch_old_pic_keys[r])
                PicUpdatePosition(r, m + 1, scratch_old_pic_keys[r], new_key_m1);
            unsigned long old_b_m1 = (unsigned long)(scratch_old_pic_keys[r] >> 64);
            unsigned long new_b_m1 = B_prefix_cache[r][m + 1].to_ulong();
            if (new_b_m1 != old_b_m1)
                BClassUpdatePosition(r, m + 1, old_b_m1, new_b_m1);
            if (m + 1 < q) {
                pic_key_t new_key_m2 = pic_key(r, m + 2);
                if (new_key_m2 != scratch_old_pic_keys[alpha + r])
                    PicUpdatePosition(r, m + 2, scratch_old_pic_keys[alpha + r], new_key_m2);
            }
        }

    }

private:
    //=========================================================================
    // ComputePrefixBPermutations()  [OPTIMIZATION — not in paper]
    //
    // Builds the B-prefix cache by walking through the operator sequence once.
    // For each replica r and each position i:
    //   B_prefix_cache[r][i] = B-state of replica r after operators Sq[0]...Sq[i-1]
    //
    // This is a simple forward pass: start from lattice[r]'s B-bits, and
    // apply each operator's B-part one by one. Each XOR applies the B-part of
    // permutation operator Sq[i] to the current B-state.
    //=========================================================================
    void ComputePrefixBPermutations() {
        // [OPT-12] Build the shared, replica-independent prefix-XOR table
        // ONCE: PXOR[i] = XOR of P_matrix[Sq[0..i-1]]. This single O(q) pass
        // replaces what used to be an implicit per-replica accumulation
        // pattern, and is what lets PartialUpdateAfterSwap look up any
        // replica's intermediate state in O(1) instead of walking Sq itself.
        PXOR[0].reset();
        for (int i = 0; i < q; i++) PXOR[i + 1] = PXOR[i] ^ P_matrix[Sq[i]];

        for(int r = 0; r < alpha; r++) {
            // B_prefix_cache[r][i] = (lattice[r] ^ PXOR[i]) & subsystem_B_mask.
            // Valid because AND distributes over XOR: (a^b)&m == (a&m)^(b&m),
            // so this is identical to accumulating P_subsystem_B[Sq[i]] one
            // step at a time -- just expressed via the shared PXOR table
            // instead of a separate per-replica running B-state.
            const std::bitset<N> lat_r = lattice[r];
            for(int i = 0; i <= q; i++)
                B_prefix_cache[r][i] = (lat_r ^ PXOR[i]) & subsystem_B_mask;
        }

        BuildBClassPositions();  // [OPT 1] — B-only index for HasAnyValidCuts
        prefix_cache_valid = true;
    }

    //=========================================================================
    // pic_key()  [OPT 1]
    //
    // Combined B+D key for position pos on replica r.
    // Encodes both the B-prefix state and the cumulative D-zero count, giving
    // a single 64-bit integer that uniquely identifies the equivalence class
    // of "valid segment endpoints" — positions that satisfy both B-closure
    // and D-nonzero for a segment starting at pos.
    //
    // Collision-free for N ≤ 32 (B-state fits in lower 32 bits) and
    // D-zero count ≤ 2^32 (impossible: count ≤ qmax = 100).
    //=========================================================================
    pic_key_t pic_key(int r, int pos) const {
        // 128-bit key: B-state in the upper 64 bits, D-zero count in the lower 32.
        // Collision-free for N <= 64 (to_ulong() is 64-bit; it throws for a set bit
        // beyond 64). BClassUpdatePosition recovers the B-state as key >> 64.
        static_assert(N <= 64, "pic_key packs the B-state into 64 bits; N > 64 needs a wider key");
        static_assert(sizeof(unsigned long) == 8, "to_ulong() must be 64-bit");
        return ((pic_key_t)B_prefix_cache[r][pos].to_ulong() << 64)
             | (pic_key_t)(uint32_t)D_zero_prefix[r][pos];
    }

    //=========================================================================
    // BuildPositionsInClass()  [OPT 1]
    //
    // Populates positions_in_class[r][key] with the sorted list of positions
    // i ∈ [0,q] that share pic_key(r,i) == key.
    //
    // Called after any update that modifies B_prefix_cache or D_zero_prefix:
    //   - PrecomputeEnergiesAndDCoefficients() (full rebuild)
    //   - PartialUpdateAfterSwap()            (incremental update, then this)
    //
    // Cost: O(α × q) — same order as the existing valid_cuts_from_zero rebuild.
    //=========================================================================
    void BuildPositionsInClass() {
        for (int r = 0; r < alpha; r++) {
            positions_in_class[r].clear();
            positions_in_class[r].reserve(q + 2);  // avoid rehashing during fill
            for (int i = 0; i <= q; i++)
                positions_in_class[r][pic_key(r, i)].push_back(i);
            // push_back in ascending i → vectors are already sorted.
        }
    }

    //=========================================================================
    // BuildBClassPositions()  [OPT 1]
    //
    // B-only class index for HasAnyValidCuts(), built/updated in
    // ComputePrefixBPermutations() and PartialUpdateAfterSwap().
    // Uses only B_prefix_cache (no D-zero), so it is valid as soon as
    // prefix_cache_valid is true — before energy precomputation.
    //=========================================================================
    void BuildBClassPositions() {
        for (int r = 0; r < alpha; r++) {
            b_class_positions[r].clear();
            b_class_positions[r].reserve(q + 2);
            for (int i = 0; i <= q; i++)
                b_class_positions[r][B_prefix_cache[r][i].to_ulong()].push_back(i);
        }
    }

    //=========================================================================
    // PicUpdatePosition()  [OPT 1]
    //
    // Moves position `pos` from old_key bucket to new_key bucket in
    // positions_in_class[r].  Used for incremental O(alpha) update in
    // PartialUpdateAfterSwap() instead of a full O(alpha*q) rebuild.
    // The bucket vectors are always sorted, so we use lower_bound.
    //=========================================================================
    void PicUpdatePosition(int r, int pos, pic_key_t old_key, pic_key_t new_key) {
        auto old_it = positions_in_class[r].find(old_key);
        if (old_it != positions_in_class[r].end()) {
            auto& v = old_it->second;
            v.erase(std::lower_bound(v.begin(), v.end(), pos));
            if (v.empty()) positions_in_class[r].erase(old_it);
        }
        auto& new_vec = positions_in_class[r][new_key];
        new_vec.insert(std::lower_bound(new_vec.begin(), new_vec.end(), pos), pos);
    }

    //=========================================================================
    // BClassUpdatePosition()  [OPT 1]
    //
    // Moves position `pos` from old_b bucket to new_b bucket in
    // b_class_positions[r].  Only B_prefix_cache[r][m+1] changes per
    // PartialUpdateAfterSwap(), so this is called once per replica with pos=m+1.
    //=========================================================================
    void BClassUpdatePosition(int r, int pos, unsigned long old_b, unsigned long new_b) {
        auto old_it = b_class_positions[r].find(old_b);
        if (old_it != b_class_positions[r].end()) {
            auto& v = old_it->second;
            v.erase(std::lower_bound(v.begin(), v.end(), pos));
            if (v.empty()) b_class_positions[r].erase(old_it);
        }
        auto& new_vec = b_class_positions[r][new_b];
        new_vec.insert(std::lower_bound(new_vec.begin(), new_vec.end(), pos), pos);
    }

    //=========================================================================
    // PicEnumerateInner()  [OPT 1]
    //
    // Recursive helper for the α≥4 multi-level pruned cut enumeration.
    // Called by EnumerateAndSumCuts() with r=1 for each fixed outer cut c0.
    //
    //   r        — current replica index (1 to alpha-1)
    //   prev_cut — left boundary of segment r (= cut at level r-1)
    //
    // At each level, looks up positions_in_class[r][pic_key(r, prev_cut)]
    // to get the pre-pruned list of valid cut positions, then recurses.
    // Only valid (B-closed, D-nonzero) cut combinations reach the leaf.
    //=========================================================================
    void PicEnumerateInner(int r, int prev_cut, ExExFloat& total_weight,
                           double target_beta, const ExExFloat* beta_factorials) {
        scratch_seg_starts[r] = prev_cut;

        if (r == alpha - 1) {
            // Last segment [prev_cut, q): q must be in the same class.
            auto it = positions_in_class[r].find(pic_key(r, prev_cut));
            if (it == positions_in_class[r].end()) return;
            if (!std::binary_search(it->second.begin(), it->second.end(), q)) return;
            total_weight += ComputeCutContribution(scratch_seg_starts, scratch_seg_ends,
                                                   target_beta, beta_factorials);
            return;
        }

        // ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────
        // Physics: Eq. 29 sums over EVERY cut tuple 0<=c1<=c2<=...<=q,
        // including cuts that coincide (an "empty segment"). The all-cuts-
        // equal-zero case is the classical (beta -> 0) limit of Z_alpha --
        // the single BIGGEST term in the whole sum at low beta.
        // Plain-English: picture the operator sequence as a rope of length q
        // that gets cut into alpha pieces. It's perfectly legal for a piece
        // to have zero length (cut right where the last cut was, or right at
        // the very start/end of the rope). The old search range here
        // excluded those zero-length pieces, so the code was quietly
        // leaving out entire terms from the sum -- including the most
        // important one. That's why every alpha>=3 run reported a NEGATIVE
        // Renyi entropy: it was summing a related-but-wrong quantity, not
        // just a noisier version of the right one. Independently verified
        // by tools/brute_force_weight.py -- a pure-combinatorics oracle with
        // no physics in it at all, which confirms the fixed enumerator
        // matches literal brute force and the old one strictly undercounts.
        // ─────────────────────────────────────────────────────────────────
        // Intermediate segment r: iterate valid cut positions in [prev_cut, q].
        // [F1 fix] Inclusive at both ends: *jt == prev_cut means segment r is
        // empty (the cut coincides with its own left boundary — a legitimate
        // Eq. 29 term, trivially in this position's own class); *jt == q means
        // every remaining segment collapses to empty at the right end. The old
        // exclusive (prev_cut, q) bounds silently dropped both cases.
        auto it = positions_in_class[r].find(pic_key(r, prev_cut));
        if (it == positions_in_class[r].end()) return;

        const auto& cands = it->second;
        auto begin  = std::lower_bound(cands.begin(), cands.end(), prev_cut);
        auto end_it = std::upper_bound(begin, cands.end(), q);

        for (auto jt = begin; jt != end_it; ++jt) {
            scratch_seg_ends[r] = *jt;
            PicEnumerateInner(r + 1, *jt, total_weight, target_beta, beta_factorials);
        }
    }

    //=========================================================================
    // CheckSegmentClosure()  (Paper Eqs. 25–29 closure constraints)
    //
    // Returns true if the B-subsystem of replica `replica` returns to its
    // starting state after applying operators Sq[start]...Sq[end-1].
    //
    // In the paper's notation:
    //   ⟨z_B^r | P^(B)_{i_q^{(j_{r-1}→j_r)}} | z_B^r⟩ ≠ 0
    // means the B-path from cut j_{r-1} to cut j_r must form a closed loop.
    //
    // Using the prefix cache: closure iff B_prefix[end] == B_prefix[start],
    // i.e. the accumulated B-permutation from position 0 to `end` equals the
    // accumulated B-permutation from 0 to `start` — meaning the segment
    // [start, end) contributed no net change to the B-state. [OPTIMIZATION]
    //=========================================================================
    bool CheckSegmentClosure(int replica, int start, int end) {
        if(!prefix_cache_valid) {
            ComputePrefixBPermutations();
        }
        return B_prefix_cache[replica][end] == B_prefix_cache[replica][start];
    }
    
    //=========================================================================
    // PrecomputeEnergiesAndDCoefficients()  [OPTIMIZATION — not in paper]
    //
    // Walks through the operator sequence ONCE for each replica r, computing:
    //   Energies_replicas[r][i] = E(state of replica r at position i)
    //   D_replicas[r][i]        = d_{Sq[i]}(state of replica r at position i)
    //
    // These values appear in the weight formula (Paper Eq. 16):
    //   W ~ D(z_A, z_B, i_q) · exp(-β[E_{z_0},...,E_{z_q}])
    //
    // WHY PRECOMPUTE?  Without precomputation, every call to GetRenyiWeight()
    // would need to walk through the full sequence for every cut position,
    // costing O(q × number_of_cuts) total. By precomputing once, we pay O(q)
    // once and reuse the results for all O(q^{α-1}) cut evaluations.
    //=========================================================================
    void PrecomputeEnergiesAndDCoefficients() {
        for(int r = 0; r < alpha; r++) {
            // Start from the current lattice state of replica r
            // (not z[r] — we walk forward from lattice[r], applying operators)
            std::bitset<N> current_config = lattice[r];

            // Energy at position 0 (before any operators)
            Energies_replicas[r][0] = CalcCompositeEnergy(r, current_config);  // [OPT-14]

            for(int i = 0; i < q; i++) {
                // Apply operator Sq[i]: this flips certain spins (XOR with P_matrix)
                current_config ^= P_matrix[Sq[i]];
                // D-coefficient = diagonal matrix element of Sq[i] at the POST-op state
                D_replicas[r][i] = calc_d_for_config(r, Sq[i], current_config);
                // Energy at position i+1 (after applying Sq[0]...Sq[i])
                Energies_replicas[r][i + 1] = CalcCompositeEnergy(r, current_config);  // [OPT-14]
            }
        }
        energies_valid = true;
        currEnergy = Energies_replicas[0][0];  // energy of replica 0 at position 0

        // [OPT 15] Build D_zero_prefix[r][0..q] for O(1) segment zero-D checks.
        // D_zero_prefix[r][i] = number of k in [0,i) with D_replicas[r][k] == 0.
        // ComputeCutContribution uses this to skip divdiff when D-product is 0.
        const std::complex<double> zero_c(0.0, 0.0);
        for(int r = 0; r < alpha; r++) {
            D_zero_prefix[r][0] = 0;
            for(int i = 0; i < q; i++) {
                D_zero_prefix[r][i + 1] = D_zero_prefix[r][i]
                                          + (D_replicas[r][i] == zero_c ? 1 : 0);
            }
        }

        BuildPositionsInClass();  // [OPT 1]
    }

    //=========================================================================
    // CheckSubsystemAClosure()  (Paper Eqs. 18, 20, 24)
    //
    // Returns true if the FULL operator sequence S_{i_q} = Sq[0]...Sq[q-1]
    // maps subsystem A back to itself.
    //
    // In the paper's notation (Eq. 20):
    //   ⟨z_A | S^(A)_{i_q} | z_A⟩ ≠ 0
    // means the operator sequence must be the identity on subsystem A.
    //
    // WHY THIS MUST HOLD:
    //   The reduced partition function Z_α is obtained by tracing over subsystem A.
    //   Only operator sequences that return to their starting A-state contribute
    //   a nonzero weight. Sequences that don't close on A give weight = 0.
    //
    // This check gates the entire weight calculation: if A doesn't close,
    // GetRenyiWeight() immediately returns 0.
    //=========================================================================
    // [OPT-12] Considered rederiving this from (PXOR[q] & subsystem_A_mask)
    // instead of the independent A_net_permutation accumulator below (they
    // are provably equal, since P_subsystem_A[k] == P_matrix[k] &
    // subsystem_A_mask -- verified in the constructor). NOT done: this check
    // deliberately runs BEFORE ComputePrefixBPermutations() in GetRenyiWeight
    // (see [OPT 4] Gate 2's comment) specifically to skip building PXOR/
    // B_prefix_cache entirely when A doesn't close. Routing this check
    // through PXOR would force PXOR to be built first, defeating that
    // lazy-evaluation ordering. A_net_permutation's own O(1)-amortized
    // incremental maintenance is already optimal for this call site, so it's
    // kept unchanged.
    bool CheckSubsystemAClosure() {
        // [OPT 7] Use the cached A-net-permutation instead of re-walking Sq each call.
        //
        // The classical derivation: A-closure holds iff the XOR (net permutation) of
        // all P_subsystem_A[Sq[i]] over the full sequence is the zero bitset —
        // equivalently, the composition of all A-operators is the identity.
        // Note: this condition is independent of lattice[] — it depends only on Sq[].
        //
        // a_closure_valid = true  ↔  A_net_permutation reflects the current Sq[].
        // It is set false only by InvalidateCache() (i.e. when Sq[] changes).
        // It survives set_beta() calls (β changes don't affect Sq[]).
        // PartialUpdateAfterSwap() keeps it true (XOR commutes: swap ≠ change XOR).
        //
        // MBAR reweighting: K=32 consecutive set_beta() calls after each MC sample.
        // The first call may rebuild the cache; subsequent 31 calls find
        // a_closure_valid=true and skip the O(q) walk entirely → O(1) each.
        if(!a_closure_valid) {
            A_net_permutation.reset();
            for(int i = 0; i < q; i++) {
                A_net_permutation ^= P_subsystem_A[Sq[i]];
            }
            a_closure_valid = true;
        }
        return A_net_permutation.none();  // closed iff net A-permutation is identity
    }
    
    //=========================================================================
    // ENERGY HELPERS  (Paper Section II.D, Eq. 15)
    //
    // These implement the diagonal parts of the PMR Hamiltonian (Eq. 15):
    //   H = D_0 + Σ_i D_i P_i
    // where D_0 is the "classical" (diagonal) part of H.
    //=========================================================================

    // CalcCompositeEnergy: returns E(z) = ⟨z|D_0|z⟩ for a given spin config z.
    // D_0 is the diagonal component of the Hamiltonian — this is the energy
    // of a purely classical spin state, with no quantum fluctuations.
    // For the Heisenberg model, this is the ZZ interaction energy (Paper Eq. 31).
    //
    // [OPT-14] Takes the full config directly. All three call sites already
    // hold the full config (cfg / current_config) and were splitting it into
    // (cfg & subsystem_A_mask, cfg & subsystem_B_mask) just so this function
    // could OR them straight back into full_config -- two throwaway bitset
    // AND ops plus one OR, every call, inside the O(alpha*q) energy-rebuild
    // loop, for nothing (A_mask/B_mask are complementary and exhaustive over
    // the N qubits, so (cfg&A)|(cfg&B) == cfg always).
    // [OVERLAP] Copy r's diagonal energy. In OVERLAP_MODE each copy r carries
    // its own Hamiltonian (ham_of_copy[r] indexes the OV_* tables generated by
    // tools/make_tables.py); otherwise the single compiled Hamiltonian is used.
    double CalcCompositeEnergy(int r, const std::bitset<N>& full_config) {
        double energy = 0.0;
#ifdef OVERLAP_MODE
        const int h = ham_of_copy[r];
        for(int k = 0; k < OV_D0_size[h]; k++) {
            int sign = 2 * ((OV_D0_product[h][k] & (~full_config)).count() % 2) - 1;
            energy -= sign * OV_D0_coeff[h][k];
        }
#else
        (void)r;
        // Sum over terms in D_0: each term is ±coeff depending on spin pattern
        for(int k = 0; k < D0_size; k++) {
            int sign = 2 * ((D0_product[k] & (~full_config)).count() % 2) - 1;
            energy -= sign * D0_coeff[k];
        }
#endif
        return energy;
    }

    // calc_d_for_config: returns d_k(z) = ⟨z|D_k|z⟩ for operator k at config z.
    // D_k is the diagonal matrix associated with the k-th permutation operator P_k.
    // In the PMR weight formula (Paper Eq. 16), each operator in the sequence
    // contributes one factor of d_k (the D in the D·P decomposition of Eq. 15).
    std::complex<double> calc_d_for_config(int r, int op_idx, const std::bitset<N>& config) {
        std::complex<double> d(0.0, 0.0);
#ifdef OVERLAP_MODE
        const int h = ham_of_copy[r];
        for(int k = 0; k < OV_D_size[h][op_idx]; k++) {
            int sign = 2 * ((OV_D_product[h][op_idx][k] & (~config)).count() % 2) - 1;
            d -= sign * OV_D_coeff[h][op_idx][k];
        }
#else
        (void)r;
        for(int k = 0; k < D_size[op_idx]; k++) {
            int sign = 2 * ((D_product[op_idx][k] & (~config)).count() % 2) - 1;
            d -= sign * D_coeff[op_idx][k];
        }
#endif
        return d;
    }
    
public:
    //=========================================================================
    // GetRenyiWeight()  —  THE CORE ALGORITHM  (Paper Eqs. 19–29)
    //
    // Computes the full weight W^(α)(C; β) for the current configuration C
    // at inverse temperature β. This is the function that defines WHAT the
    // Monte Carlo samples.
    //
    // For α=1 (Paper Eqs. 19–20):
    //   W^(1)(z_A, z_B, i_q; β) = D(z_A,z_B,i_q) · exp(-β[E_{z_0},...,E_{z_q}])
    //   Valid only when the FULL sequence closes on BOTH A and B.
    //   This is just the standard PMR partition function weight.
    //
    // For α=2 (Paper Eqs. 24–27):
    //   W^(2) = Σ_{cut j} [segment 0→j weight for B^0] × [segment j→q weight for B^1]
    //   Both segments must close on their respective B-replica.
    //   The FULL sequence must close on A.
    //
    // For α=3 (Paper Eqs. 28–29):
    //   W^(3) = Σ_{cuts j1<j2} [0→j1 for B^0] × [j1→j2 for B^1] × [j2→q for B^2]
    //   All three segments must close on their respective B-replicas.
    //
    // STEPS:
    //   1. Check A-subsystem closure (if not, return 0 immediately)
    //   2. Precompute energies and D-coefficients (if cache invalid)
    //   3. Precompute B-prefix states (if cache invalid)
    //   4. For α=1: single segment check and weight
    //   5. For α≥2: enumerate all valid cut positions and sum contributions
    //=========================================================================
    // [OPT-1] Reimplemented on top of EnumerateTuples()+EvaluateEnumeratedAt()
    // so there is ONE enumeration code path regardless of caller: the
    // single-beta MC path (UpdateWeight) hits both calls every time (no
    // regression -- same total work as before, just reorganized into two
    // named phases); the K-beta driver sweep calls EnumerateTuples() ONCE
    // per sample and EvaluateEnumeratedAt() K times directly (instead of calling this
    // function), which is where the win comes from -- see EnumerateTuples().
    ExExFloat GetRenyiWeight(double target_beta, const ExExFloat* beta_factorials) {
#ifdef OVERLAP_PLUS
        (void)beta_factorials;
        tuples_valid = true; tuples_nonzero = true;
        return OvWeight(target_beta);
#endif
        if (!tuples_valid) EnumerateTuples();
        return EvaluateEnumeratedAt(target_beta, beta_factorials);
    }

    //=========================================================================
    // EnumerateTuples()  [OPT-1]
    //
    // Runs every part of the weight computation that does NOT depend on
    // beta: the A-closure gate, B-prefix/energy cache rebuilds, the
    // HasAnyValidCuts() existence check, and (via BuildTupleTables()) the
    // D-product prefix/suffix tables the alpha=2/3 paths use. All of this
    // was previously repeated inside GetRenyiWeight() for EVERY one of the
    // K betas in the driver's u-fill sweep, even though none of it depends
    // on beta -- Sq[]/lattice[] are frozen for the whole sweep (the replica
    // doesn't move between weight evaluations at different betas).
    //
    // Call ONCE per sample (right after EnsureCachesValid()), then call
    // EvaluateEnumeratedAt() once per beta to get just the beta-dependent
    // part. Sets tuples_valid=true and tuples_nonzero (whether ANY term in
    // the weight sum can be nonzero) so EvaluateEnumeratedAt() -- and the
    // driver, via the return value here -- can skip ALL K per-beta
    // evaluations when the weight is identically zero for every beta at
    // once (a common case: ~100% of cuts are pruned at scale, see [OPT 15]).
    //
    // Returns 1 if the weight can be nonzero for some beta, 0 if it is
    // identically zero (A doesn't close, or no valid cut exists at all --
    // both beta-independent facts).
    //=========================================================================
    int EnumerateTuples() {
#ifdef OVERLAP_PLUS
        if (!ov_valid) OvPrecompute();
        tuples_valid = true; tuples_nonzero = true;
        return 1;
#endif
        if(!CheckSubsystemAClosure()) {
            tuples_valid = true;
            tuples_nonzero = false;
            return 0;
        }
        if(!prefix_cache_valid) {
            ComputePrefixBPermutations();
        }
        if(!HasAnyValidCuts()) {
            tuples_valid = true;
            tuples_nonzero = false;
            return 0;
        }
        if(!energies_valid) {
            PrecomputeEnergiesAndDCoefficients();
        }
        if(alpha >= 2) {
            BuildTupleTables();
        }
        tuples_valid = true;
        tuples_nonzero = true;
        return 1;
    }

    //=========================================================================
    // EvaluateEnumeratedAt()  [OPT-1]
    //
    // Computes W^(alpha)(C; target_beta) using the tables EnumerateTuples()
    // already built. Only the beta-DEPENDENT work happens here: divided
    // differences of -beta*Energies (via EnumerateAndSumCuts/
    // ComputeSingleSegmentWeight) and the beta_factorials combination.
    //
    // PRECONDITION: EnumerateTuples() must have been called since the last
    // configuration change (same contract as EnsureCachesValid() before
    // EvaluateWeightAt() -- violating this gives stale/wrong results).
    //=========================================================================
    ExExFloat EvaluateEnumeratedAt(double target_beta, const ExExFloat* beta_factorials) {
#ifdef OVERLAP_PLUS
        (void)beta_factorials;
        return OvWeight(target_beta);
#endif
        if(!tuples_nonzero) {
            return ExExFloat(0.0);
        }
        if(alpha == 1) {
            return ComputeSingleSegmentWeight(0, 0, q, target_beta, beta_factorials);
        }
        ExExFloat total_weight(0.0);
        EnumerateAndSumCuts(total_weight, target_beta, beta_factorials);
        return total_weight;
    }

private:
    //=========================================================================
    // EnumerateAndSumCuts()  (Paper Eqs. 25–29)
    //
    // Sums the weight contribution of every valid (α-1)-cut configuration.
    //
    // [OPT 3]  Outer loop restricted to valid_cuts_from_zero instead of 0..q.
    // [OPT E / 4.1]  Uses pre-allocated scratch_seg_starts/ends/cuts — no heap
    //                allocation inside the hot loop.
    // [OPT F / 4.2]  α=3 specialized path: precomputes scratch_valid_j2 for the
    //                inner cut, skipping all j2 values where segment 1 can't close.
    //                Matches the explicit-nested-loop structure described in §4.2
    //                of the professor's notes.
    // [OPT G / 4.4]  α=3 incremental divdiff: segment 0 is built ONCE per c0
    //                (constant across the j2 sweep); segment 1 is extended with a
    //                single AddElement per j2 step instead of a full O(L²) rebuild.
    // [OPT-3]        α=3 segment 2 (independent of c0) is now built ONCE for the
    //                whole call via a right-anchored suffix table (w2_all/D2_suf),
    //                instead of rebuilt from scratch per (c0,j2) pair; segment-0/1
    //                D-products use prefix-product tables (Pnz0/Pnz1) instead of a
    //                direct O(len) loop. Turns the α=3 inner cost from
    //                O(q^2*(q+L2^2)) into O(q^2) per weight call.
    // [OPT-4]        α=2 is its own two-pass path (ascending segment-0 cache,
    //                descending right-anchored segment-1), sharing Pnz0/Pnz1 with
    //                OPT-3. Turns α=2 from O(q^3) worst case into O(q^2).
    //=========================================================================
    //=========================================================================
    // BuildTupleTables()  [OPT-1, factored out of the old per-beta build in
    // EnumerateAndSumCuts -- see OPT-3 D / OPT-4 / OPT-3 C]
    //
    // Builds every BETA-INDEPENDENT table the alpha=2/alpha=3 weight paths
    // need: Pnz0/Pnz1 (D-product prefix products) and, for alpha=3, D2_suf
    // (D-product suffix). None of these involve target_beta -- they depend
    // only on D_replicas/D_zero_prefix, fixed for the whole K-beta sweep
    // within one sample. Called once by EnumerateTuples(), not once per beta
    // as the pre-OPT-1 code did.
    //
    // Precondition: energies_valid == true (D_replicas is current).
    //=========================================================================
    void BuildTupleTables() {
        // [OPT-3 D / OPT-4] See EnumerateAndSumCuts's original comment for
        // the full derivation: Pnz_r[i] = product of D_replicas[r][0..i),
        // skipping zero factors; any segment [a,b) drawn from
        // positions_in_class is guaranteed D-zero-free, so Pnz_r[b]/Pnz_r[a]
        // recovers the exact product over [a,b). Guard against magnitude
        // drift over up to qmax complex factors leaving [1e-100,1e100].
        prefix_unsafe_cached = false;
        if (alpha == 2 || alpha == 3) {
            const std::complex<double> zeroc(0.0, 0.0), onec(1.0, 0.0);
            Pnz0[0] = onec;
            for (int i = 0; i < q; i++) {
                Pnz0[i+1] = (D_replicas[0][i] == zeroc) ? Pnz0[i] : Pnz0[i] * D_replicas[0][i];
                double m = std::abs(Pnz0[i+1]);
                if (m > 0.0 && (m < 1e-100 || m > 1e100)) prefix_unsafe_cached = true;
            }
            Pnz1[0] = onec;
            for (int i = 0; i < q; i++) {
                Pnz1[i+1] = (D_replicas[1][i] == zeroc) ? Pnz1[i] : Pnz1[i] * D_replicas[1][i];
                double m = std::abs(Pnz1[i+1]);
                if (m > 0.0 && (m < 1e-100 || m > 1e100)) prefix_unsafe_cached = true;
            }
        }
        // [OPT-3 C] Segment-2 D-product suffix table (alpha=3 only). Pure
        // D-product, no divdiff/beta involved -- unlike w2_all (the weight
        // half of the same original loop), which stays beta-dependent and is
        // still rebuilt per-beta inside EnumerateAndSumCuts.
        if (alpha == 3) {
            D2_suf[q] = std::complex<double>(1.0, 0.0);
            for (int j2 = q - 1; j2 >= 0; j2--)
                D2_suf[j2] = D_replicas[2][j2] * D2_suf[j2 + 1];
        }
    }

    void EnumerateAndSumCuts(ExExFloat& total_weight,
                             double target_beta,
                             const ExExFloat* beta_factorials) {
        scratch_seg_starts[0] = 0;
        scratch_seg_ends[alpha - 1] = q;

        // [OPT 1] Outer loop over positions_in_class[0][key(0,0)]:
        // only c0 values where segment [0,c0) BOTH closes on replica 0 AND has no
        // D-zeros.  Replaces valid_cuts_from_zero + the explicit D_zero_prefix[0]
        // check that appeared inside the α=3 path.
        const pic_key_t key0 = pic_key(0, 0);
        auto it0 = positions_in_class[0].find(key0);
        if (it0 == positions_in_class[0].end()) return;

        // [OPT-1] Pnz0/Pnz1 (and, for alpha=3, D2_suf) are built once per
        // sample by EnumerateTuples()->BuildTupleTables(), not once per beta
        // here -- read the cached member tables/flag instead of rebuilding.
        // (GetRenyiWeight guarantees EnumerateTuples() has already run.)
        const bool prefix_unsafe = prefix_unsafe_cached;

        // ── α=2 two-pass path [OPT-4] ───────────────────────────────────────
        // Replaces the old "rebuild both segments from scratch per candidate
        // c0" (O(c0^2+(q-c0)^2) per cut -> O(q^3) worst case per weight call,
        // since AddElement's own internal cost is O(qmax) per call -- see
        // OPT-3 for why call COUNT, not just added-element count, dominates)
        // with two incremental passes -- O(q) AddElement calls each -- over
        // the SAME candidate list.
        if (alpha == 2) {
#ifdef OVERLAP_MODE
            // [OVERLAP fix] ov_tp0[c] = prod_{i<c} d0_i, ov_ts1[c] = prod_{i>=c} d1_i, extended range.
            ov_tp0.resize(q + 1); ov_ts1.resize(q + 1);
            ov_tp0[0] = ExExFloat(1.0);
            for (int i = 0; i < q; i++) ov_tp0[i + 1] = ov_tp0[i] * std::real(D_replicas[0][i]);
            ov_ts1[q] = ExExFloat(1.0);
            for (int i = q - 1; i >= 0; i--) ov_ts1[i] = ov_ts1[i + 1] * std::real(D_replicas[1][i]);
#endif
            // Pass A (ascending): build segment-0 divdiff incrementally over
            // replica 0, caching w0 at every candidate as we reach it.
            divdiff_replicas[0].CurrentLength = 0;
            divdiff_replicas[0].AddElement(-target_beta * Energies_replicas[0][0]);
            int built_len = 0;
            for (int c0 : it0->second) {
                while (built_len < c0) {
                    built_len++;
                    divdiff_replicas[0].AddElement(-target_beta * Energies_replicas[0][built_len]);
                }
                w0_cache[c0] = divdiff_replicas[0].divdiffs[c0] * beta_factorials[c0];
            }

            // Pass B (descending): build segment-1 divdiff right-anchored at
            // q over replica 1 (node-order-independence argument as in
            // OPT-3 B), combining with the Pass-A cache and the prefix
            // D-products as each candidate is reached.
            divdiff_replicas[1].CurrentLength = 0;
            divdiff_replicas[1].AddElement(-target_beta * Energies_replicas[1][q]);
            int built_down = q;
            for (auto rit = it0->second.rbegin(); rit != it0->second.rend(); ++rit) {
                int c0 = *rit;
                while (built_down > c0) {
                    built_down--;
                    divdiff_replicas[1].AddElement(-target_beta * Energies_replicas[1][built_down]);
                }
                // [OPT 1] Is q a valid endpoint for replica 1's segment [c0,q)?
                auto it1 = positions_in_class[1].find(pic_key(1, c0));
                if (it1 == positions_in_class[1].end()) continue;
                if (!std::binary_search(it1->second.begin(), it1->second.end(), q)) continue;

                ExExFloat w1 = divdiff_replicas[1].divdiffs[q - c0] * beta_factorials[q - c0];
#ifdef OVERLAP_MODE
                total_weight += w0_cache[c0] * w1 * ov_tp0[c0] * ov_ts1[c0];   // [OVERLAP fix] no underflow
#else
                std::complex<double> D0, D1;
                if (!prefix_unsafe) {
                    D0 = Pnz0[c0];
                    D1 = Pnz1[q] / Pnz1[c0];
                } else {
                    D0 = std::complex<double>(1.0, 0.0);
                    for (int i = 0; i < c0; i++) D0 *= D_replicas[0][i];
                    D1 = std::complex<double>(1.0, 0.0);
                    for (int i = c0; i < q; i++) D1 *= D_replicas[1][i];
                }
                total_weight += w0_cache[c0] * w1 * REALW(D0 * D1);
#endif
            }
            return;
        }

        // [OPT-3 B] alpha=3: precompute the segment-2 WEIGHT suffix table
        // ONCE per weight call, before the c0 loop (segment 2 = [j2,q) never
        // depends on c0). This part is beta-dependent (uses target_beta), so
        // unlike D2_suf (the D-product half, now built once per sample by
        // BuildTupleTables() -- see [OPT-1]) it must still be rebuilt here,
        // once per beta.
        if (alpha == 3) {
            // Divided differences are symmetric in node order (a general
            // property, independent of this implementation), so building the
            // table by adding nodes q, q-1, ..., 0 (right-to-left) gives the
            // exact same divdiffs[k] as an ascending [j2,j2+k] build would --
            // up to floating-point reassociation, same as any other OPT here.
            divdiff_replicas[2].CurrentLength = 0;
            divdiff_replicas[2].AddElement(-target_beta * Energies_replicas[2][q]);
            w2_all[q] = divdiff_replicas[2].divdiffs[0] * beta_factorials[0];
            for (int j2 = q - 1; j2 >= 0; j2--) {
                divdiff_replicas[2].AddElement(-target_beta * Energies_replicas[2][j2]);
                w2_all[j2] = divdiff_replicas[2].divdiffs[q - j2] * beta_factorials[q - j2];
            }
        }

        for (int c0 : it0->second) {
            scratch_seg_ends[0] = c0;

            // ── α=3 specialized path [OPT F/4.2] + [OPT G/4.4] + [OPT 1] ────
            if (alpha == 3) {
                // [OPT 1] j2 candidates: positions where segment [c0,j2) BOTH closes
                // on replica 1 AND has no D-zeros.  Replaces the O(q) scratch_valid_j2
                // scan (which was B-only) plus the separate D_zero_prefix[1] check.
                auto it1 = positions_in_class[1].find(pic_key(1, c0));
                if (it1 == positions_in_class[1].end()) continue;
                const auto& j2_cands = it1->second;
                // STUDY NOTE: this is the alpha=3 fast path's copy of the same
                // fix -- full physics/plain-English writeup is at
                // PicEnumerateInner()'s "[F1 fix]" comment above (fix landed
                // 2026-07-09, reviewed 2026-09-10).
                // [F1 fix] Inner cuts satisfy c0 <= j2 <= q (inclusive both ends):
                // j2 == c0 means segment 1 is empty; j2 == q means segment 2 is
                // empty. Both are legitimate Eq. 29 terms that the old exclusive
                // (c0, q) bounds silently dropped -- this is the bug that made
                // every alpha=3 run report a negative Renyi entropy.
                auto j2_begin = std::lower_bound(j2_cands.begin(), j2_cands.end(), c0);
                auto j2_end   = std::upper_bound(j2_begin,         j2_cands.end(), q);
                if (j2_begin == j2_end) continue;

                // [OPT-3 A] w0/D0 depend only on c0, not j2 -- hoisted out of
                // the j2 loop (used to be recomputed for every (c0,j2) pair).
                divdiff_replicas[0].CurrentLength = 0;
                for (int i = 0; i <= c0; i++)
                    divdiff_replicas[0].AddElement(-target_beta * Energies_replicas[0][i]);
                ExExFloat w0 = divdiff_replicas[0].divdiffs[c0] * beta_factorials[c0];
                std::complex<double> D0;
                if (!prefix_unsafe) {
                    D0 = Pnz0[c0];  // == product over [0,c0); zero-free by PIC guarantee
                } else {
                    D0 = std::complex<double>(1.0, 0.0);
                    for (int i = 0; i < c0; i++) D0 *= D_replicas[0][i];
                }

                // Segment 1 divdiff still depends on c0 -- built incrementally
                // across the j2 sweep exactly as before (this part unchanged).
                divdiff_replicas[1].CurrentLength = 0;
                divdiff_replicas[1].AddElement(-target_beta * Energies_replicas[1][c0]);
                int seg1_right = c0;

                for (auto jt = j2_begin; jt != j2_end; ++jt) {
                    int j2 = *jt;

                    // Extend segment 1 incrementally from seg1_right to j2.
                    while (seg1_right < j2) {
                        seg1_right++;
                        divdiff_replicas[1].AddElement(
                            -target_beta * Energies_replicas[1][seg1_right]);
                    }

                    // [OPT 1] Replace B_prefix_cache[2]+D_zero_prefix[2] with PIC lookup.
                    auto it2 = positions_in_class[2].find(pic_key(2, j2));
                    if (it2 == positions_in_class[2].end()) continue;
                    if (!std::binary_search(it2->second.begin(), it2->second.end(), q))
                        continue;

                    ExExFloat w1 = divdiff_replicas[1].divdiffs[j2 - c0] * beta_factorials[j2 - c0];
                    // [OPT-3 B] Segment-2 weight read from the precomputed suffix table.
                    ExExFloat w2 = w2_all[j2];

                    std::complex<double> D1;
                    if (!prefix_unsafe) {
                        D1 = Pnz1[j2] / Pnz1[c0];  // [OPT-3 D]
                    } else {
                        D1 = std::complex<double>(1.0, 0.0);
                        for (int i = c0; i < j2; i++) D1 *= D_replicas[1][i];
                    }
                    // [OPT-3 C] Segment-2 D-product read from the precomputed suffix table.
                    std::complex<double> D2 = D2_suf[j2];

                    total_weight += w0 * w1 * w2 * REALW(D0 * D1 * D2);
                }
                continue;
            }

            // ── α≥4: multi-level PIC enumeration  [OPT 1] ────────────────────
            // Replaces the O(q^{α-2}) odometer with a recursive scan that visits
            // only combinations satisfying B-closure AND D-nonzero at every level.
            scratch_seg_ends[alpha - 1] = q;  // invariant for the recursive helper
            PicEnumerateInner(1, c0, total_weight, target_beta, beta_factorials);
        }
    }
    
    //=========================================================================
    // HasAnyValidCuts()  [OPT 4 — lazy energy evaluation]
    //
    // Lightweight check: does ANY valid (α-1)-cut configuration exist?
    // Returns true on the first valid cut found, false if none exist.
    //
    // This is a pure B-closure check — it reads only B_prefix_cache, not
    // energies or D-coefficients.  Used by GetRenyiWeight() to skip the
    // expensive PrecomputeEnergiesAndDCoefficients() call when the weight
    // is guaranteed to be zero (no valid cuts ⟹ every term in the sum is 0).
    //
    // Mirrors the iteration structure of EnumerateAndSumCuts() exactly,
    // but returns bool instead of accumulating weight contributions.
    //
    // Precondition: prefix_cache_valid == true.
    //=========================================================================
    bool HasAnyValidCuts() {
        if(alpha == 1) {
            // α=1: one segment [0, q), check full-sequence B-closure on replica 0
            return B_prefix_cache[0][q] == B_prefix_cache[0][0];
        }

        // α≥2: outer loop over positions where segment [0,c0) closes on replica 0.
        // b_class_positions[0] is the B-only index (valid when prefix_cache_valid);
        // looking up B_prefix_cache[0][0] gives the same set that valid_cuts_from_zero
        // held, but without a separate vector to maintain.
        const auto it0 = b_class_positions[0].find(B_prefix_cache[0][0].to_ulong());
        if (it0 == b_class_positions[0].end()) return false;
        for (int c0 : it0->second) {

            // α=2: one cut, one additional check on replica 1
            if(alpha == 2) {
                if(B_prefix_cache[1][q] == B_prefix_cache[1][c0])
                    return true;
                continue;
            }

            // α=3 fast path: O(k_B) via b_class_positions instead of O(q) odometer.
            // Look up positions j2 where segment [c0,j2) closes on replica 1, then
            // check each for B-closure of segment [j2,q) on replica 2.
            if (alpha == 3) {
                auto it1 = b_class_positions[1].find(B_prefix_cache[1][c0].to_ulong());
                if (it1 == b_class_positions[1].end()) continue;
                const auto& j2_cands = it1->second;
                const std::bitset<N>& tgt2 = B_prefix_cache[2][q];
                // STUDY NOTE: same F1 fix as PicEnumerateInner() above (fix
                // landed 2026-07-09, reviewed 2026-09-10) -- this is just the
                // "does a valid cut exist at all" fast-path version, so it
                // must use the exact same inclusive bounds or it could say
                // "no valid cuts" while EnumerateAndSumCuts finds some (or
                // vice versa), silently desynchronizing the two functions.
                // [F1 fix] Inclusive [c0, q]: j2==c0 (segment 1 empty) and j2==q
                // (segment 2 empty) are valid, must mirror EnumerateAndSumCuts.
                auto jt     = std::lower_bound(j2_cands.begin(), j2_cands.end(), c0);
                auto jt_end = std::upper_bound(jt, j2_cands.end(), q);
                for (; jt != jt_end; ++jt)
                    if (B_prefix_cache[2][*jt] == tgt2) return true;
                continue;
            }

            // α≥4: inner odometer over cuts[1]..cuts[α-2]
            // [OPT E / 4.1] Use pre-allocated scratch_cuts instead of a local vector.
            for(int r = 0; r < alpha - 1; r++) scratch_cuts[r] = c0;

            while(true) {
                // Check segments 1..α-1 for B-closure
                bool all_close = true;
                for(int r = 1; r < alpha; r++) {
                    int seg_start = scratch_cuts[r - 1];
                    int seg_end   = (r == alpha - 1) ? q : scratch_cuts[r];
                    if(B_prefix_cache[r][seg_end] != B_prefix_cache[r][seg_start]) {
                        all_close = false;
                        break;
                    }
                }
                if(all_close) return true;

                // Advance inner odometer (cuts[1]..cuts[α-2]; cuts[0]=c0 fixed)
                int idx = alpha - 2;
                while(idx >= 1 && scratch_cuts[idx] == q) idx--;
                if(idx < 1) break;
                scratch_cuts[idx]++;
                for(int j = idx + 1; j < alpha - 1; j++) scratch_cuts[j] = scratch_cuts[idx];
            }
        }
        return false;
    }

    //=========================================================================
    // ComputeCutContribution()  (Paper Eqs. 25/29 — one term in the sum)
    //
    // Computes the weight contribution from ONE specific cut configuration.
    // This is the product over all α segments of their individual weights.
    //
    // Each segment's weight has two parts (Paper Eq. 16 applied per segment):
    //   1. Exponential (temperature) part: computed via divided differences
    //      of the energy sequence along that segment.
    //      = (-β)^{len} / len! × divdiff[len] of {-β·E_{z_0},...,-β·E_{z_len}}
    //   2. D-coefficient product: product of d_{Sq[i]} values along the segment.
    //
    // The final weight is the real part of the product of all segment weights
    // across all α replicas.
    //=========================================================================
    ExExFloat ComputeCutContribution(const std::vector<int>& segment_starts,
                                     const std::vector<int>& segment_ends,
                                     double target_beta,
                                     const ExExFloat* beta_factorials) {
        // [OPT 15] Early exit: if ANY segment on ANY replica has a zero D-coefficient,
        // the D-product for that segment is zero, making the entire contribution zero.
        // Skip the expensive O(L²) divided-difference computation entirely.
        //
        // Check uses the O(1) prefix-sum: segment [start, end) has a zero iff
        //   D_zero_prefix[r][end] != D_zero_prefix[r][start]
        // (valid because D_zero_prefix is maintained in sync with D_replicas).
        //
        // For the Heisenberg model, each D-coefficient is 0 with ~50% probability
        // (aligned bond pairs give d=0). For a segment of length L, the probability
        // of at least one zero is 1-(0.5)^L — essentially 1 for L≥10. This means
        // ~100% of cut contributions are skipped at scale (N=400, L≈100).
        for(int r = 0; r < alpha; r++) {
            if(D_zero_prefix[r][segment_ends[r]] != D_zero_prefix[r][segment_starts[r]])
                return ExExFloat(0.0);
        }

        ExExFloat total_contribution(1.0);
        std::complex<double> total_D_product(1.0);

        for(int r = 0; r < alpha; r++) {
            int start          = segment_starts[r];
            int end            = segment_ends[r];
            int segment_length = end - start;

            // Compute the divided-difference sum for this segment's energy sequence
            // This implements the exp(-β·H) approximation term for this segment
            divdiff_replicas[r].CurrentLength = 0;
            for(int i = 0; i <= segment_length; i++) {
                divdiff_replicas[r].AddElement(
                    -target_beta * Energies_replicas[r][start + i]
                );
            }

            // segment_exp_weight = (-β)^len / len! × divdiff[len]
            // (Paper Eq. 16: the e^{-β[...]} term factored over the segment)
            ExExFloat segment_exp_weight =
                divdiff_replicas[r].divdiffs[segment_length] *
                beta_factorials[segment_length];

            // D-coefficient product for this segment: Π_{i=start}^{end-1} d_{Sq[i]}
            // (Paper Eq. 16: the D(z_A, z_B, i_q) factor)
            std::complex<double> segment_D_product(1.0);
            for(int i = start; i < end; i++) {
                segment_D_product *= D_replicas[r][i];
            }

            total_contribution *= segment_exp_weight;
            total_D_product    *= segment_D_product;
        }

        // REALW takes the real part, matching the PMR weight convention
        // (Paper footnote 1: weight must be real and nonneg for sign-free systems)
        return total_contribution * REALW(total_D_product);
    }

    //=========================================================================
    // ComputeSingleSegmentWeight()  (Paper Eq. 19–20, α=1 case)
    //
    // Special-case weight computation for a single segment spanning [start, end).
    // Used for α=1 where there is only one replica and no cuts.
    // Exactly equivalent to ComputeCutContribution with α=1, but faster since
    // it avoids vector allocation and outer loop overhead [OPTIMIZATION].
    //=========================================================================
    ExExFloat ComputeSingleSegmentWeight(int replica, int start, int end,
                                         double target_beta,
                                         const ExExFloat* beta_factorials) {
        int length = end - start;

        // [OPT 15] Early exit: any zero D-coefficient in [start,end) → contribution=0.
        // Identical logic to ComputeCutContribution's early exit, specialized to α=1.
        if(D_zero_prefix[replica][end] != D_zero_prefix[replica][start])
            return ExExFloat(0.0);

        // Divided-difference sum for the full sequence energy path
        divdiff_replicas[replica].CurrentLength = 0;
        for(int i = 0; i <= length; i++) {
            divdiff_replicas[replica].AddElement(
                -target_beta * Energies_replicas[replica][start + i]
            );
        }

        ExExFloat exp_weight =
            divdiff_replicas[replica].divdiffs[length] * beta_factorials[length];

        // D-coefficient product for the full sequence
        std::complex<double> D_product(1.0);
        for(int i = start; i < end; i++) {
            D_product *= D_replicas[replica][i];
        }

        return exp_weight * REALW(D_product);
    }
    
public:
    //=========================================================================
    // UpdateWeight()  — recompute weight at current β for the current config
    //
    // Called after any accepted or rejected MC move to keep currWeight in sync.
    // MBAR always uses exact (full) weights — no approximations [OPTIMIZATION note:
    // the base Replica class has a fast UpdateWeight that incrementally updates;
    // here we always recompute from scratch via GetRenyiWeight for correctness].
    //=========================================================================
    ExExFloat UpdateWeight() {
        // [OPT B] If beta was restored directly after a PT swap rejection
        // (without calling set_beta()), beta_pow_factorial is stale for the old beta.
        // Rebuild from scratch before the lazy-extension step below uses it.
        if(beta_dirty) {
            ExExFloat curr(1.0);
            beta_pow_factorial[0] = curr;
            int cap = q + 1;
            if(cap > qmax) cap = qmax;
            for(int k = 1; k < cap; k++) {
                curr *= (-beta) / k;
                beta_pow_factorial[k] = curr;
            }
            beta_pow_computed = cap;
            beta_dirty = false;
        }
        // [OPT 11] Lazily extend beta_pow_factorial to cover all indices 0..q.
        // This is needed when q grew since the last set_beta() call (e.g. after
        // an insert-pair or cycle-completion sub-move in the composite update).
        // Each extension step is one ExExFloat multiply — O(1) amortized.
        while(beta_pow_computed <= q) {
            beta_pow_factorial[beta_pow_computed] =
                beta_pow_factorial[beta_pow_computed - 1] * ((-beta) / beta_pow_computed);
            beta_pow_computed++;
        }
        return GetRenyiWeight(beta, beta_pow_factorial);
    }

    //=========================================================================
    // EnsureCachesValid()  [Optimization A — Professor's Section 1.2]
    //
    // Guarantees that all three caches are current for the existing configuration
    // (lattice[] and Sq[]).  Call this ONCE per sample before the β-sweep.
    //
    // WHY THIS EXISTS:
    //   OPT 1 saves and restores currWeight on MC rejection, but it still calls
    //   InvalidateCache() which sets energies_valid = prefix_cache_valid =
    //   a_closure_valid = false.  After a rejection the restored configuration
    //   is correct but the caches are marked stale.  Previously set_beta() would
    //   re-build them on the first call of each K-rung sweep — paying the full
    //   O(α·q·N) cost once per sample whenever the last move was rejected.
    //
    //   EnsureCachesValid() makes that rebuild explicit, so EvaluateWeightAt()
    //   can be called K times in a row without any implicit rebuild.
    //
    // COST: O(α·q·D0_size) if stale, O(1) if already valid.
    //   For the 9-qubit system: ~1,200 ops when stale, 3 bool checks when valid.
    //=========================================================================
    void EnsureCachesValid() {
        // Rebuild A-net-permutation if Sq[] changed since the last check.
        // [OPT 7] a_closure_valid tracks whether A_net_permutation reflects Sq[].
        // InvalidateCache() clears it; we reconstruct here in O(q).
        if(!a_closure_valid) {
            A_net_permutation.reset();
            for(int i = 0; i < q; i++)
                A_net_permutation ^= P_subsystem_A[Sq[i]];
            a_closure_valid = true;
        }

        // Rebuild B-prefix cache if stale (pure function of lattice[] and Sq[]).
        // Needed by CheckSegmentClosure() inside GetRenyiWeight().
        if(!prefix_cache_valid) ComputePrefixBPermutations();

        // Rebuild energy and D-coefficient cache if stale (pure function of
        // lattice[] and Sq[]).  Also rebuilds D_zero_prefix [OPT 15].
        if(!energies_valid) PrecomputeEnergiesAndDCoefficients();

        config_dirty = false;  // [OPT A] all fine-grained caches are now valid
    }

    //=========================================================================
    // EvaluateWeightAt()  [Optimization A — Professor's Section 1.3]
    //
    // Computes W^(α)(C; target_beta) WITHOUT modifying any replica state.
    //
    // WHY THIS IS FASTER THAN set_beta():
    //   set_beta() modifies this->beta, this->beta_pow_factorial, and
    //   this->currWeight — all replica state that must later be restored.
    //   In the MBAR sampling loop we call it K=32 times and then call it once
    //   more just to undo the changes (32+1 full set_beta calls per sample).
    //
    //   EvaluateWeightAt() instead builds beta_pow_factorial on the STACK
    //   (ExExFloat local_bpf[qmax] ≈ 1.2 KB — no heap, no allocation) and
    //   passes it directly to GetRenyiWeight().  The replica's own beta,
    //   beta_pow_factorial, and currWeight are never touched.
    //
    //   K calls to EvaluateWeightAt() replace K+1 calls to set_beta(), and
    //   since the caches were prepared by EnsureCachesValid(), every call goes
    //   straight to EnumerateAndSumCuts() — the O(q^{α-1}) inner sum — with
    //   no cache rebuilds at all.
    //
    // PRECONDITION: EnsureCachesValid() must have been called since the last
    //   configuration change (MC move). Violating this gives stale results.
    //
    // THREAD-SAFETY: reads replica state (caches) but never writes it.
    //   Multiple threads may call EvaluateWeightAt() on DIFFERENT replicas
    //   simultaneously without synchronization.  Do NOT call on the SAME
    //   replica from multiple threads — the underlying GetRenyiWeight reads
    //   shared member arrays.
    //=========================================================================
    ExExFloat EvaluateWeightAt(double target_beta) {
        // Build (-target_beta)^k / k! on the stack for k = 0..q.
        // Only entries 0..q are used; qmax entries would waste time since q < qmax
        // for typical configurations.
        ExExFloat local_bpf[qmax];
        ExExFloat curr(1.0);
        local_bpf[0] = curr;
        int cap = (q + 1 < qmax) ? q + 1 : qmax;
        for(int k = 1; k < cap; k++) {
            curr *= (-target_beta) / k;
            local_bpf[k] = curr;
        }

        // Evaluate the full Rényi weight using the local table and existing caches.
        // GetRenyiWeight() reads lattice[], Sq[], B_prefix_cache, Energies_replicas,
        // D_replicas, and D_zero_prefix — all valid after EnsureCachesValid().
        // It writes nothing to member state.
        return GetRenyiWeight(target_beta, local_bpf);
    }

    //=========================================================================
    // EvaluateWeightAt()  [OPT-2 — shared per-rung beta-factorial table]
    //
    // Same contract as the single-argument overload above, but takes a
    // PRE-BUILT (-target_beta)^k/k! table instead of rebuilding local_bpf on
    // the stack every call. The K-rung beta ladder never changes during a
    // pipeline run, so the driver builds one bpf[K][qmax] table ONCE up
    // front (see run_mbar_pt_pipeline) and passes bpf[j].data() here for
    // rung j. This removes O(q) redundant ExExFloat multiplies per call —
    // at K rungs x N_SAMPLES x replicas, that's O(K^2*q) wasted work per
    // sample eliminated (was rebuilt identically by every replica, every
    // sample, for the same K betas).
    //
    // PRECONDITION: bpf_row[0..q] must equal (-target_beta)^k/k! for
    // k=0..q. Caller-owned; not touched here.
    //=========================================================================
    ExExFloat EvaluateWeightAt(double target_beta, const ExExFloat* bpf_row) {
        return GetRenyiWeight(target_beta, bpf_row);
    }

    //=========================================================================
    // Metropolis()  — standard Metropolis acceptance probability
    //
    // Returns min(1, |W_new / W_old|) — the probability of accepting a proposed
    // MC move. If the new weight is larger, always accept (return 1.0).
    // If smaller, accept with probability W_new/W_old.
    //
    // This satisfies detailed balance: the probability of being in state A and
    // moving to B equals the probability of being in B and moving to A,
    // weighted by the ratio of their weights. This guarantees the Markov chain
    // converges to the correct thermal distribution (Paper Section II.F).
    //=========================================================================
    double Metropolis(ExExFloat newWeight) {
        return std::min(1.0, std::abs((newWeight / currWeight).get_double()));
    }

    //=========================================================================
    // CONFIGURATION MANAGEMENT  [helper utilities]
    //
    // UpdateReferenceConfigs: after an accepted move, copy lattice[] → z[] so
    //   the reference configuration stays current with the sampled configuration.
    //
    // BackupLattices / RestoreLattices: save and restore lattice[] when
    //   attempting an MC move that may be rejected. This is cheaper than
    //   recomputing the configuration from scratch.
    //=========================================================================
    void UpdateReferenceConfigs() {
        for(int r = 0; r < alpha; r++) {
            z[r] = lattice[r];
        }
    }

    void BackupLattices() {
        for(int r = 0; r < alpha; r++) {
            lattice_backup[r] = lattice[r];
        }
    }

    void RestoreLattices() {
        for(int r = 0; r < alpha; r++) {
            lattice[r] = lattice_backup[r];
        }
    }

    std::string CompositeStateKey(const std::vector<std::bitset<N>>& state) const {
        std::string key;
        key.reserve((size_t)alpha * 12);
        for(int r = 0; r < alpha; r++) {
            key += std::to_string(state[r].to_ulong());
            key += ':';
        }
        return key;
    }

    void BuildCompositeStateOccurrences(std::unordered_map<std::string, std::vector<int>>& occ) const {
        occ.clear();
        occ.reserve(q + 2);
        std::vector<std::bitset<N>> state = lattice;
        occ[CompositeStateKey(state)].push_back(0);
        for(int i = 0; i < q; i++) {
            for(int r = 0; r < alpha; r++) state[r] ^= P_matrix[Sq[i]];
            occ[CompositeStateKey(state)].push_back(i + 1);
        }
    }

    bool PickTwoPositions(const std::vector<int>& positions, int& a, int& b, int min_len) {
        if(positions.size() < 2) return false;
        for(int attempt = 0; attempt < 32; attempt++) {
            int i = int(local_val(local_rng) * positions.size());
            int j = int(local_val(local_rng) * (positions.size() - 1));
            if(j >= i) j++;
            a = positions[std::min(i, j)];
            b = positions[std::max(i, j)];
            if(b - a >= min_len) return true;
        }
        return false;
    }

    bool ApplyInnerLoopReversal() {
        if(q < 2) return false;
        std::unordered_map<std::string, std::vector<int>> occ;
        BuildCompositeStateOccurrences(occ);
        std::vector<const std::vector<int>*> groups;
        for(const auto& kv : occ) if(kv.second.size() >= 2) groups.push_back(&kv.second);
        if(groups.empty()) return false;
        for(int attempt = 0; attempt < 32; attempt++) {
            const std::vector<int>& positions = *groups[int(local_val(local_rng) * groups.size())];
            int a, b;
            if(!PickTwoPositions(positions, a, b, 2)) continue;
            int left = a, right = b - 1;
            while(left < right) {
                int tmp = Sq[left];
                Sq[left] = Sq[right];
                Sq[right] = tmp;
                left++;
                right--;
            }
            InvalidateCache();
            return true;
        }
        return false;
    }

    bool ApplyLoopExchange() {
        if(q < 2) return false;
        std::unordered_map<std::string, std::vector<int>> occ;
        BuildCompositeStateOccurrences(occ);
        std::vector<const std::vector<int>*> groups;
        for(const auto& kv : occ) if(kv.second.size() >= 3) groups.push_back(&kv.second);
        if(groups.empty()) return false;
        for(int attempt = 0; attempt < 32; attempt++) {
            const std::vector<int>& positions = *groups[int(local_val(local_rng) * groups.size())];
            std::vector<int> picks;
            picks.reserve(3);
            while(picks.size() < 3) {
                int p = positions[int(local_val(local_rng) * positions.size())];
                if(std::find(picks.begin(), picks.end(), p) == picks.end()) picks.push_back(p);
            }
            std::sort(picks.begin(), picks.end());
            int a = picks[0], b = picks[1], c = picks[2];
            if(b <= a || c <= b) continue;
            int tmp[qmax], pos = 0;
            for(int i = 0; i < a; i++) tmp[pos++] = Sq[i];
            for(int i = b; i < c; i++) tmp[pos++] = Sq[i];
            for(int i = a; i < b; i++) tmp[pos++] = Sq[i];
            for(int i = c; i < q; i++) tmp[pos++] = Sq[i];
            memcpy(Sq, tmp, q * sizeof(int));
            InvalidateCache();
            return true;
        }
        return false;
    }

    bool ApplySegmentExchange() {
        if(q < 3) return false;
        std::unordered_map<std::string, std::vector<int>> occ;
        BuildCompositeStateOccurrences(occ);
        std::vector<const std::vector<int>*> groups;
        for(const auto& kv : occ) if(kv.second.size() >= 2) groups.push_back(&kv.second);
        if(groups.size() < 2) return false;
        // [DIAG segex] SEGMENT_EXCHANGE_MAX_ATTEMPTS=1 makes this a single draw
        // (a non-interleaved draw is a no-op), which is the detailed-balanced form.
        for(int attempt = 0; attempt < SEGMENT_EXCHANGE_MAX_ATTEMPTS; attempt++) {
            int gi = int(local_val(local_rng) * groups.size());
            int gj = int(local_val(local_rng) * (groups.size() - 1));
            if(gj >= gi) gj++;
            int x0, x1, y0, y1;
            if(!PickTwoPositions(*groups[gi], x0, x1, 1)) continue;
            if(!PickTwoPositions(*groups[gj], y0, y1, 1)) continue;
            int a, b, c, d;
            if(x0 < y0 && y0 < x1 && x1 < y1) {
                a = x0; b = y0; c = x1; d = y1;
            } else if(y0 < x0 && x0 < y1 && y1 < x1) {
                a = y0; b = x0; c = y1; d = x1;
            } else {
                continue;
            }
            int tmp[qmax], pos = 0;
            for(int i = 0; i < a; i++) tmp[pos++] = Sq[i];
            for(int i = c; i < d; i++) tmp[pos++] = Sq[i];
            for(int i = b; i < c; i++) tmp[pos++] = Sq[i];
            for(int i = a; i < b; i++) tmp[pos++] = Sq[i];
            for(int i = d; i < q; i++) tmp[pos++] = Sq[i];
            memcpy(Sq, tmp, q * sizeof(int));
            InvalidateCache();
            return true;
        }
        return false;
    }

    bool TryTrueLoopMove(bool (RenyiReplica::*move)()) {
        int oldSq[qmax];
        memcpy(oldSq, Sq, q * sizeof(int));
        ExExFloat oldWeight = currWeight;
        if(!(this->*move)()) return false;
        ExExFloat newWeight = UpdateWeight();
        if(newWeight.get_mantissa() != 0.0 && local_val(local_rng) < Metropolis(newWeight)) {
            currWeight = newWeight;
            return true;
        }
        memcpy(Sq, oldSq, q * sizeof(int));
        InvalidateCache();
        currWeight = oldWeight;
        return false;
    }
    
    //=========================================================================
    // MC UPDATE HELPERS
    //=========================================================================
    int NoRepetitionCheck(int* sequence, int r) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < i; j++) {
                if(sequence[j] == sequence[i]) return 0;
            }
        }
        return 1;
    }
    
    void PickSubsequence(int r) {
        int m = int(local_val(local_rng) * (q - r + 1));
        for(int i = 0; i < r; i++) {
            Sq_subseq[i] = Sq[i + m];
        }
        min_index = m;
        max_index = m + r - 1;
    }
    
    int FindCycles(int r) {
        std::bitset<Ncycles> curr;
        curr.set();
        
        for(int i = 0; i < r; i++) {
            curr &= P_in_cycles[Sq_subseq[i]];
        }
        
        #ifndef EXHAUSTIVE_CYCLE_SEARCH
        for(int i = 0; i < Ncycles; i++) {
            if(curr[i]) {
                if(cycle_len[i] < 2 * r - 1 || cycle_len[i] > 2 * r + 1) {
                    curr.reset(i);
                }
            }
        }
        #endif
        
        found_cycles = curr.count();
        if(found_cycles > 0) {
            int k = int(local_val(local_rng) * found_cycles);
            int i = 0, sum = 0;
            while(sum <= k) {
                sum += curr[i++];
            }
            return i - 1;
        }
        return -1;
    }
    
    //=========================================================================
    // update()  —  ONE MONTE CARLO STEP  (Paper Section II.F, Refs [22,23])
    //
    // Proposes a random change to the current configuration C = (lattice, Sq)
    // and accepts or rejects it via the Metropolis criterion so that the
    // long-run frequency of visiting each configuration is proportional to
    // its weight W^(α)(C; β). This is the "detailed balance" condition.
    //
    // THREE TYPES OF MOVES are attempted with fixed probabilities:
    //
    //   80% — COMPOSITE UPDATE: changes to the OPERATOR SEQUENCE Sq[]
    //          (adds, removes, swaps, or restructures operators)
    //   10% — BLOCK SWAP: circular rotation of Sq[] with matching lattice update
    //   10% — SPIN FLIP: changes a single spin in the lattice configuration
    //
    // The mix ensures ERGODICITY (every valid configuration can be reached)
    // while keeping individual moves small enough that acceptance rates
    // remain reasonably high.
    //
    // IMPORTANT for the Rényi case:
    //   Operator sequence moves (composite + block swap) act symmetrically on
    //   all α replicas' lattice states — this is correct because they probe
    //   the joint configuration space.
    //   Spin flips have different rules for A-spins vs B-spins (see below).
    //=========================================================================
    void update() {
        int i, m, p, r, u, oldq, cont;
        double v = (Nop > 0) ? local_val(local_rng) : 1.0;
        ExExFloat newWeight;
        double Rfactor;

        // ════════════════════════════════════════════════════════════════════
        // COMPOSITE UPDATE (80% of moves)  (Refs [22,23])
        //
        // This is the primary sampling move for the OPERATOR SEQUENCE.
        // It chains together multiple sub-moves into a single macro-step,
        // then makes ONE accept/reject decision for the whole chain.
        //
        // WHY CHAIN MOVES? [OPTIMIZATION]
        //   A single swap or insert/delete rarely changes the weight much.
        //   Chaining many moves together creates "macro-moves" that can
        //   traverse the configuration space more efficiently.
        //   The chain stops randomly with probability COMPOSITE_UPDATE_BREAK_PROBABILITY
        //   at each step (geometrically distributed length).
        //
        // Rfactor accumulates the ratio of proposal probabilities (necessary
        // for detailed balance when insert/delete moves change q).
        // ════════════════════════════════════════════════════════════════════
        if(v < 0.8) {
            Rfactor = 1.0;
            oldq = q;
            memcpy(Sq_backup, Sq, q * sizeof(int));  // save sequence for possible rollback
            BackupLattices();                          // save lattice for possible rollback
            // [OPT 1] Save the pre-move weight so we can restore it on rejection
            // instead of calling UpdateWeight() (which recomputes GetRenyiWeight from scratch).
            // Correctness: on rejection we restore Sq and lattice to the backed-up state,
            // which is exactly the configuration that produced savedWeight.  The cache is
            // marked invalid by InvalidateCache() so it will be rebuilt on the next call.
            const ExExFloat savedWeight = currWeight;
            newWeight = currWeight;

            do {
                cont = 0;  // cont=1 means "keep chaining" (new weight is zero, try more)
                v = local_val(local_rng);
                // [F3 fix] active_moves/thresholds now read the two split flags
                // independently: ENABLE_INNER_LOOP_REVERSAL (disabled, no Hastings
                // correction) and ENABLE_LOOP_EXCHANGE_UPDATES (the two symmetric
                // block-permutation moves, unaffected).
                const int active_moves = 4 + (ENABLE_INNER_LOOP_REVERSAL ? 1 : 0)
                                            + (ENABLE_LOOP_EXCHANGE_UPDATES ? 1 : 0)
                                            + (ENABLE_SEGMENT_EXCHANGE ? 1 : 0)    // [DIAG segex]
                                            + (OV_SINGLE_MOVES ? 4 : 0);           // [OVERLAP_PLUS] single insert/remove + start-shift pair
                const double inv_moves = 1.0 / double(active_moves);
                const double t_swap = inv_moves;
                const double t_del = t_swap + inv_moves;
                const double t_ins = t_del + inv_moves;
                const double t_cycle = t_ins + inv_moves;
                const double t_inner = t_cycle + (ENABLE_INNER_LOOP_REVERSAL ? inv_moves : 0.0);
                const double t_exchange = t_inner + (ENABLE_LOOP_EXCHANGE_UPDATES ? inv_moves : 0.0);
                const double t_segment = t_exchange + (ENABLE_SEGMENT_EXCHANGE ? inv_moves : 0.0);   // [DIAG segex]
                const double t_sins = t_segment + (OV_SINGLE_MOVES ? inv_moves : 0.0);
                const double t_srem = t_sins + (OV_SINGLE_MOVES ? inv_moves : 0.0);
                const double t_fins = t_srem + (OV_SINGLE_MOVES ? inv_moves : 0.0);
                const double t_frem = t_fins + (OV_SINGLE_MOVES ? inv_moves : 0.0);

                // ── Sub-move 1: ADJACENT SWAP (25%) ─────────────────────────
                // Swaps the order of two neighboring operators Sq[m] ↔ Sq[m+1].
                // This changes the "path" without changing its length.
                // Paper: operator ordering affects the energy path and thus the weight.
                if(v < t_swap) {
                    if(q >= 2 && !DISABLE_MOVE_SWAP) {   // [DIAG moves]
                        m = int(local_val(local_rng) * (q - 1));
                        if(Sq[m] != Sq[m + 1]) {
                            p = Sq[m]; Sq[m] = Sq[m + 1]; Sq[m + 1] = p;
                            PartialUpdateAfterSwap(m);  // [OPT 2]: 4 writes/replica instead of full rebuild
                            newWeight = UpdateWeight();
                            // ── STUDY NOTE [F2 fix landed 2026-07-09 · reviewed
                            //    2026-09-10] ──────────────────────────────────
                            // Physics: detailed balance -- the microscopic
                            // reversibility condition that guarantees this
                            // Markov chain converges to the correct thermal
                            // distribution instead of a biased one.
                            // Plain-English: when a proposed move lands on a
                            // "dead end" (zero weight), every OTHER sub-move
                            // in this chain says "that's a dead end, try
                            // another sub-move instead of stopping the chain
                            // here" (cont=1 below). The swap sub-move alone
                            // forgot to say that, so it could sometimes end
                            // the whole chain right on a dead end when every
                            // sibling move wouldn't have -- a small asymmetry
                            // between how likely the forward chain was to
                            // stop here vs. how likely its reverse would be.
                            // ────────────────────────────────────────────────
                            // [F2 fix] Every other sub-move in this chain sets cont=1 when it
                            // lands on a zero-weight state, so the composite update keeps
                            // chaining instead of accepting/rejecting a doomed proposal. This
                            // one didn't, unlike the legacy Replica::update()'s equivalent
                            // swap branch -- a detailed-balance asymmetry (see MBARPT_audit F2).
                            if(newWeight.get_mantissa() == 0.0) cont = 1;
                        }
                    }
                }
                // ── Sub-move 2: DELETE PAIR (25%) ────────────────────────────
                // Removes two identical adjacent operators Sq[m]=Sq[m+1] from
                // the sequence. Because P^2 = identity for permutation operators,
                // applying the same operator twice returns to the same state —
                // so removing the pair is a valid move that preserves closure.
                // Rfactor /= Nop accounts for the change in proposal probability
                // (we're removing a pair, making it harder to reach this shorter
                // sequence from the shorter one by insertion).
                else if(v < t_del) {
                    if(q >= 2) {
                        m = int(local_val(local_rng) * (q - 1));
                        if(Sq[m] == Sq[m + 1]) {
                            for(i = m; i < q - 2; i++) Sq[i] = Sq[i + 2];
                            q -= 2; Rfactor /= Nop;
                            InvalidateCache();
                            newWeight = UpdateWeight();
                            if(newWeight.get_mantissa() == 0.0) cont = 1;
                        }
                    }
                }
                // ── Sub-move 3: INSERT PAIR (25%) ────────────────────────────
                // Inserts two identical operators Sq[m]=Sq[m+1]=p at a random
                // position. The reverse of delete pair. Rfactor *= Nop.
                // Inserting a pair p,p doesn't change the final state (P²=I),
                // but it expands the path length and changes the weight through
                // the new energy points along the longer path.
                else if(v < t_ins) {
                    if(q + 2 < qmax) {
                        m = int(local_val(local_rng) * (q + 1));
                        p = local_diceNop(local_rng);
                        for(i = q - 1; i >= m; i--) Sq[i + 2] = Sq[i];
                        q += 2; Sq[m] = Sq[m + 1] = p;
                        Rfactor *= Nop;
                        InvalidateCache();
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    } else qmax_achieved = 1;  // sequence at max length cap
                }
                // ── Sub-move 4: CYCLE COMPLETION (25%) ──────────────────────
                // Replaces a subsequence of operators with the "complementary"
                // subsequence from the same algebraic cycle. This is a more
                // sophisticated move that preserves the operator group structure.
                //
                // INTUITION: Operators form mathematical cycles (like clock
                // positions). If operators {A, B} are part of the same cycle,
                // replacing them with the remaining operators {C, D} in that
                // cycle is a valid symmetry-preserving transformation.
                //
                // This move is CRITICAL for ergodicity — without it, the sampler
                // can get stuck in subspaces of the operator algebra. (Refs [22,23])
                else if(v < t_cycle) {
                    int j2 = 0; double wfactor;
                    u = local_geometric(local_rng);  // random gap length
                    if(q >= u + rmin && !DISABLE_MOVE_CYCLE) {   // [DIAG moves]
                        int inv_pr = std::min(rmax, q - u) - (rmin) + 1;
                        r = int(local_val(local_rng) * inv_pr) + (rmin);
                        PickSubsequence(r + u);
                        std::shuffle(Sq_subseq, Sq_subseq + r + u, local_rng);
                        if(NoRepetitionCheck(Sq_subseq, r)) {
                            for(i = 0; i < u; i++) Sq_gaps[i] = Sq_subseq[i + r];
                            m = FindCycles(r);
                            if(found_cycles > 0) {
                                P = cycles[m];
                                for(i = 0; i < r; i++) P.reset(Sq_subseq[i]);
                                p = (int)P.count();
                                if(q + p - r < qmax) {
                                    if(r < p)      for(i = q-1; i > max_index; i--) Sq[i+p-r] = Sq[i];
                                    else if(r > p) for(i = max_index+1; i < q; i++) Sq[i+p-r] = Sq[i];
                                    for(i = 0; i < p; i++) {
                                        while(!P.test(j2)) j2++;
                                        Sq_subseq[i] = Sq[min_index + i] = j2++;
                                    }
                                    for(i = 0; i < u; i++) Sq[min_index + p + i] = Sq_gaps[i];
                                    std::shuffle(Sq + min_index, Sq + min_index + p + u, local_rng);
                                    q += p - r;
                                    InvalidateCache();
                                    newWeight = UpdateWeight();
                                    // wfactor = ratio of proposal probabilities for detailed balance
                                    wfactor = found_cycles;
                                    FindCycles(p);
                                    wfactor /= found_cycles;
                                    wfactor *= factorial[p] / factorial[r];
                                    wfactor *= inv_pr;
                                    inv_pr = std::min(rmax, q - u) - (rmin) + 1;
                                    wfactor /= inv_pr;
                                    Rfactor *= wfactor;
                                    cycles_used[m] = 1;
                                    if(newWeight.get_mantissa() == 0.0) cont = 1;
                                } else qmax_achieved = 1;
                            }
                        }
                    }
                }
                else if(v < t_inner) {
#if ENABLE_INNER_LOOP_REVERSAL
                    // [F3] Disabled by default -- ApplyInnerLoopReversal's proposal is not
                    // symmetric (reversing a segment changes interior composite states, so
                    // the occurrence-map group sizes differ forward vs. reverse) and this
                    // branch applies plain Metropolis with no Hastings correction. Left in
                    // place, dead behind the flag, for a future corrected re-enable.
                    if(ApplyInnerLoopReversal()) {
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    }
#endif
                }
                else if(v < t_exchange) {
#if ENABLE_LOOP_EXCHANGE_UPDATES
                    if(!DISABLE_MOVE_LOOPEX && ApplyLoopExchange()) {   // [DIAG moves]
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    }
#endif
                }
                else if(v < t_segment) {
#if ENABLE_SEGMENT_EXCHANGE
                    if(ApplySegmentExchange()) {
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    }
#endif
                }
                // [OVERLAP_PLUS] Single-operator insertion/removal. Open walks need
                // them: pair moves and cycle substitutions never change P_Gamma, so
                // without these the endpoint y_q could never move away from y0.
                // Insertion: generator uniform over Nop, gap uniform over q+1;
                // removal: position uniform over q. Equal selection probabilities give
                // T(rem)/T(ins) = Nop for insertion (Rfactor *= Nop), 1/Nop for removal.
                else if(v < t_sins) {
                    if(q + 1 < qmax) {
                        m = int(local_val(local_rng) * (q + 1));
                        p = local_diceNop(local_rng);
                        for(i = q - 1; i >= m; i--) Sq[i + 1] = Sq[i];
                        Sq[m] = p; q += 1;
                        Rfactor *= Nop;
                        InvalidateCache();
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    } else qmax_achieved = 1;
                }
                else if(v < t_srem) {
                    if(q >= 1) {
                        m = int(local_val(local_rng) * q);
                        for(i = m; i < q - 1; i++) Sq[i] = Sq[i + 1];
                        q -= 1;
                        Rfactor /= Nop;
                        InvalidateCache();
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    }
                }
                // [OVERLAP_PLUS] Start-shift pair: the only moves besides single spin
                // flips that change the walk's start y0. Without them y0 is frozen
                // (flipping a spin of y0 flips it along the whole walk and is almost
                // never accepted at large beta), and each replica stays conditioned on
                // its initial y0 -- the cause of the bimodal 4x3 results.
                // Insertion: prepend P_p and set y0' = P_p y0, so the walk y0' -> y0 ->
                // (old walk) keeps every old intermediate state; p uniform over Nop.
                // Removal: drop Sq[0] and set y0' = P_{Sq[0]} y0 (deterministic).
                // T(rem)/T(ins) = Nop for insertion, 1/Nop for removal.
                else if(v < t_fins) {
                    if(q + 1 < qmax) {
                        p = local_diceNop(local_rng);
                        for(i = q - 1; i >= 0; i--) Sq[i + 1] = Sq[i];
                        Sq[0] = p; q += 1;
                        for(int rep = 0; rep < alpha; rep++) lattice[rep] ^= P_matrix[p];
                        Rfactor *= Nop;
                        InvalidateCache();
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    } else qmax_achieved = 1;
                }
                else if(v < t_frem) {
                    if(q >= 1) {
                        p = Sq[0];
                        for(i = 0; i < q - 1; i++) Sq[i] = Sq[i + 1];
                        q -= 1;
                        for(int rep = 0; rep < alpha; rep++) lattice[rep] ^= P_matrix[p];
                        Rfactor /= Nop;
                        InvalidateCache();
                        newWeight = UpdateWeight();
                        if(newWeight.get_mantissa() == 0.0) cont = 1;
                    }
                }
            // Keep chaining sub-moves with probability COMPOSITE_UPDATE_BREAK_PROBABILITY
            } while(cont && local_val(local_rng) > COMPOSITE_UPDATE_BREAK_PROBABILITY);

            // ── Final accept/reject for the entire composite chain ───────────
            // Metropolis ratio includes both weight ratio AND Rfactor (proposal ratio).
            // This single decision covers all the sub-moves performed in the chain.
            composite_attempts++;                 // [DIAG lattice]
            if(newWeight.get_mantissa() != 0.0 &&
               local_val(local_rng) < Metropolis(newWeight * Rfactor)) {
                composite_accepts++;              // [DIAG lattice]
                currWeight = newWeight;          // ACCEPT: keep new sequence
                UpdateReferenceConfigs();         // update z[] to match new lattice[]
            } else {
                q = oldq;                         // REJECT: restore old sequence length
                memcpy(Sq, Sq_backup, q * sizeof(int));  // restore operator sequence
                RestoreLattices();                // restore lattice configs
                InvalidateCache();
                // [OPT 1] Restore pre-move weight directly — no recomputation needed.
                // The restored (Sq, lattice) is identical to the pre-move state, so
                // savedWeight is exactly correct.  The cache will be rebuilt lazily
                // on the next GetRenyiWeight call.
                currWeight = savedWeight;
            }
        }
        // ════════════════════════════════════════════════════════════════════
        // BLOCK SWAP (10% of moves)  (Refs [22,23])  [OPTIMIZATION]
        //
        // Performs a CIRCULAR ROTATION of the operator sequence.
        // Instead of [op_0, op_1, ..., op_m, op_{m+1}, ..., op_{q-1}], we get
        //            [op_{m+1}, ..., op_{q-1}, op_0, ..., op_m].
        //
        // To keep the physics consistent, we also advance the lattice state by
        // applying operators op_0...op_m before the rotation. This effectively
        // "relabels" where the path starts without changing the path itself.
        //
        // WHY IS THIS USEFUL? It changes which part of the cycle appears first
        // in the sequence, enabling exploration of configurations that pure
        // insert/delete/swap moves might take a very long time to reach.
        // ════════════════════════════════════════════════════════════════════
        // [OVERLAP_PLUS] The cyclic rotation below is only valid for CLOSED walks
        // (rotating an open walk and shifting y0 is not an involution), so it is
        // switched off for the open |+> ensemble; that 10% goes to spin flips.
        else if(v < 0.9 && q >= 2 && !DISABLE_MOVE_BLOCK && !OV_SINGLE_MOVES) {   // [DIAG moves]
            m = (q == 2) ? 0 : int(local_val(local_rng) * (q - 1));

            // [OPT 1] Save pre-move weight for potential rejection restoration.
            const ExExFloat savedWeight = currWeight;

            // Save lattice BEFORE modifying, so RestoreLattices() can undo on reject
            BackupLattices();

            // Advance all replicas' lattice by applying operators Sq[0]...Sq[m]
            // (this shifts the "starting point" of the path forward by m+1 operators)
            for(int rep = 0; rep < alpha; rep++) {
                for(i = 0; i <= m; i++) {
                    lattice[rep] ^= P_matrix[Sq[i]];
                }
            }

            // Rotate the operator sequence: [0..m, m+1..q-1] → [m+1..q-1, 0..m]
            for(i = 0; i <= m; i++)       Sq_backup[q - 1 - m + i] = Sq[i];
            for(i = m + 1; i < q; i++)    Sq_backup[i - m - 1] = Sq[i];
            for(i = 0; i < q; i++) { p = Sq[i]; Sq[i] = Sq_backup[i]; Sq_backup[i] = p; }

            InvalidateCache();
            newWeight = UpdateWeight();

            block_attempts++;                     // [DIAG lattice]
            if(local_val(local_rng) < Metropolis(newWeight)) {
                block_accepts++;                  // [DIAG lattice]
                currWeight = newWeight;           // ACCEPT
                UpdateReferenceConfigs();
            } else {
                memcpy(Sq, Sq_backup, q * sizeof(int));  // REJECT: restore sequence
                RestoreLattices();                         // restore lattice
                InvalidateCache();
                currWeight = savedWeight;  // [OPT 1] restore; skip recomputation
            }
        }
        // ════════════════════════════════════════════════════════════════════
        // SPIN FLIP (10% of moves)  (Paper Section II.F)
        //
        // Flips a single spin (qubit) in the lattice configuration, which
        // changes the classical energy contribution E(z_A, z_B).
        //
        // The Rényi replica structure requires different handling depending
        // on whether the flipped spin belongs to subsystem A or B:
        //
        //   A-SPIN: zA is SHARED across all α replicas.
        //     We must flip the same bit in ALL replica lattices simultaneously.
        //     This is required because the weight formula (Paper Eq. 24) has
        //     a SINGLE z_A — all replicas must agree on the A-configuration.
        //
        //   B-SPIN: each replica has its OWN independent zB^r.
        //     We flip the bit in ONE randomly chosen replica only.
        //     This is required because the B-configurations are summed
        //     independently in Paper Eq. 24 (Σ_{zB, zB'}).
        //     IMPORTANT: if we flipped B-spins in ALL replicas simultaneously,
        //     the difference (zB^0 XOR zB^1) would be conserved and the sampler
        //     would be trapped — it could never explore all B-configurations.
        //
        //   α=1: only one replica, so flip any spin freely (standard QMC).
        // ════════════════════════════════════════════════════════════════════
        else {
#if defined(OVERLAP_PLUS) && OV_CUT_FLIP
            // [OV_CUT_FLIP] Block flip of one side of the cut, in y0 and so at every
            // imaginary time. Self-inverse, proposal symmetric -> plain Metropolis.
            if(OV_CUT_MASK.any() && local_val(local_rng) < OV_CUT_FLIP_PROB) {
                const ExExFloat savedWeight = currWeight;
                for(int rep = 0; rep < alpha; rep++) lattice[rep] ^= OV_CUT_MASK;
                InvalidateCache();
                newWeight = UpdateWeight();
                cutflip_attempts++;
                if(newWeight.get_mantissa() != 0.0 && local_val(local_rng) < Metropolis(newWeight)) {
                    cutflip_accepts++;
                    currWeight = newWeight;
                    UpdateReferenceConfigs();
                } else {
                    for(int rep = 0; rep < alpha; rep++) lattice[rep] ^= OV_CUT_MASK;
                    InvalidateCache();
                    currWeight = savedWeight;
                }
                return;
            }
#endif
            p = local_diceN(local_rng);  // pick a random qubit index

            if(alpha == 1) {
                // Standard single-replica spin flip
                const ExExFloat savedWeight = currWeight;  // [OPT 1]
                lattice[0].flip(p);
                InvalidateCache();
                newWeight = UpdateWeight();
                flip_attempts++;                               // [DIAG lattice]
                if(local_val(local_rng) < Metropolis(newWeight)) {
                    flip_accepts++;                            // [DIAG lattice]
                    currWeight = newWeight;
                    UpdateReferenceConfigs();
                } else {
                    lattice[0].flip(p);   // undo the flip
                    InvalidateCache();
                    currWeight = savedWeight;  // [OPT 1] restore; skip recomputation
                }
            } else if(subsystem_A_mask.test(p)) {
                // A-spin: flip in ALL replicas simultaneously (shared zA)
                const ExExFloat savedWeight = currWeight;  // [OPT 1]
                for(int rep = 0; rep < alpha; rep++) lattice[rep].flip(p);
                InvalidateCache();
                newWeight = UpdateWeight();
                flip_attempts++;                               // [DIAG lattice]
                if(local_val(local_rng) < Metropolis(newWeight)) {
                    flip_accepts++;                            // [DIAG lattice]
                    currWeight = newWeight;
                    UpdateReferenceConfigs();
                } else {
                    // Undo: flip back in all replicas
                    for(int rep = 0; rep < alpha; rep++) lattice[rep].flip(p);
                    InvalidateCache();
                    currWeight = savedWeight;  // [OPT 1] restore; skip recomputation
                }
            } else if(subsystem_B_mask.test(p)) {
                // B-spin: flip in ONE randomly chosen replica only (independent zB^r)
                int rep = int(local_val(local_rng) * alpha);
                const ExExFloat savedWeight = currWeight;  // [OPT 1]
                lattice[rep].flip(p);
                InvalidateCache();
                newWeight = UpdateWeight();
                flip_attempts++;                               // [DIAG lattice]
                if(local_val(local_rng) < Metropolis(newWeight)) {
                    flip_accepts++;                            // [DIAG lattice]
                    currWeight = newWeight;
                    UpdateReferenceConfigs();
                } else {
                    lattice[rep].flip(p);  // undo: flip back the same replica
                    InvalidateCache();
                    currWeight = savedWeight;  // [OPT 1] restore; skip recomputation
                }
            }
            // Note: if p belongs to neither A nor B (gap qubit), nothing happens.
        }
    }
    
    //=========================================================================
    // ACCESSORS  [simple getters for external use]
    //=========================================================================
    ExExFloat GetWeight() const { return currWeight; }  // current W^(α)(C; β)
    int GetQ() const { return q; }                      // current sequence length
    double GetEnergy() const { return currEnergy; }     // current classical energy
    int GetAlpha() const { return alpha; }              // Rényi order

    //=========================================================================
    // DIAGNOSTICS  [OPTIMIZATION — debugging/validation utility, not in paper]
    //
    // Prints the current state of the replica including whether the closure
    // conditions are satisfied. Useful for verifying correctness during
    // development, but not called during production runs.
    //=========================================================================
    void PrintStatus() {
        std::cout << "\nRenyiReplica (α=" << alpha << "):" << std::endl;
        std::cout << "  q = " << q << std::endl;
        std::cout << "  Weight = " << currWeight.get_double() << std::endl;
        std::cout << "  Energy = " << currEnergy << std::endl;
        std::cout << "  A-closure: " << (CheckSubsystemAClosure() ? "YES" : "NO") << std::endl;
        for(int r = 0; r < alpha; r++) {
            std::cout << "  B-closure[" << r << "]: "
                      << (CheckSegmentClosure(r, 0, q) ? "YES" : "NO") << std::endl;
        }
    }
};
