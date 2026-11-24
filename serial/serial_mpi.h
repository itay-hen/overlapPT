// serial_mpi.h -- minimal single-process stand-in for MPI, for building and testing on a laptop
// without an MPI installation: compile with -DSERIAL_MPI -I serial (build_overlap.sh does this).
// With one rank, every collective is a copy and the point-to-point calls are
// never reached (they only occur between neighbouring ranks).  Do NOT use this
// for production runs; use mpicxx/mpirun on the cluster.
#pragma once
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define MPI_VERSION 3
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef struct { int MPI_SOURCE, MPI_TAG, MPI_ERROR; } MPI_Status;

#define MPI_COMM_WORLD 0
#define MPI_STATUS_IGNORE ((MPI_Status*)0)
enum { MPI_INT = 1, MPI_DOUBLE, MPI_LONG_LONG, MPI_UNSIGNED, MPI_UNSIGNED_LONG_LONG };
enum { MPI_SUM = 1, MPI_MAX, MPI_LOR };

inline size_t serial_mpi_size(MPI_Datatype t) {
    switch (t) {
        case MPI_INT: return sizeof(int);
        case MPI_DOUBLE: return sizeof(double);
        case MPI_LONG_LONG: return sizeof(long long);
        case MPI_UNSIGNED: return sizeof(unsigned);
        case MPI_UNSIGNED_LONG_LONG: return sizeof(unsigned long long);
    }
    std::fprintf(stderr, "serial_mpi: unknown datatype\n"); std::abort();
}
inline void serial_mpi_p2p() {
    std::fprintf(stderr, "serial_mpi: point-to-point call with a single rank\n"); std::abort();
}

inline int MPI_Init(int*, char***) { return 0; }
inline int MPI_Finalize() { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int* r) { *r = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int* s) { *s = 1; return 0; }
inline int MPI_Abort(MPI_Comm, int code) { std::exit(code); }
inline double MPI_Wtime() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
inline int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm) { return 0; }
inline int MPI_Allreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op, MPI_Comm) {
    if (s != r) std::memmove(r, s, n * serial_mpi_size(t)); return 0;
}
inline int MPI_Reduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op, int, MPI_Comm) {
    if (s != r) std::memmove(r, s, n * serial_mpi_size(t)); return 0;
}
inline int MPI_Allgatherv(const void* s, int n, MPI_Datatype t, void* r, const int*, const int* displs,
                          MPI_Datatype rt, MPI_Comm) {
    std::memmove((char*)r + displs[0] * serial_mpi_size(rt), s, n * serial_mpi_size(t)); return 0;
}
inline int MPI_Send(const void*, int, MPI_Datatype, int, int, MPI_Comm) { serial_mpi_p2p(); return 1; }
inline int MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*) { serial_mpi_p2p(); return 1; }
inline int MPI_Sendrecv(const void*, int, MPI_Datatype, int, int, void*, int, MPI_Datatype, int, int,
                        MPI_Comm, MPI_Status*) { serial_mpi_p2p(); return 1; }
