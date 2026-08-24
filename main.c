/*
 * StarPU-MPI counterpart of ../parsec-repro/halo_ring: same task shape
 * (a ring of NT tiles, block-distributed over MPI ranks, exchanging a small
 * typed buffer between neighboring tiles for NSTEPS independent rounds),
 * used to check whether StarPU's MPI layer is subject to the same
 * cross-rank data-exchange race documented in ../PARSEC_ISSUE.md.
 *
 * Task graph (same shape as halo_ring.jdf):
 *
 *   Send(k, j) --typed buffer--> Recv(k, (j+1) % NT)
 *
 * Unlike the PaRSEC PTG version, StarPU has no separate DSL/code-generation
 * step: the same "submit tasks, let the runtime infer the DAG from data
 * dependencies" driver code runs on every rank (SPMD-style), and
 * starpu_mpi_task_insert() decides, per task, whether this rank executes it
 * and whether a send/recv is needed to move data between ranks.
 *
 * Every tile has a zero-payload "anchor" handle (starpu_void_data) whose
 * only purpose is to pin a task's execution to the rank owning that tile,
 * exactly like halo_ring's TILES data collection. The actual payload is one
 * vector handle per (k, j) pair -- the buffer produced by Send(k,j) and
 * consumed by Recv(k, (j+1) % NT) -- registered with home rank = owner of
 * tile j (the producer), matching the JDF's WRITE/READ pairing.
 *
 * Unlike the PaRSEC reproducer (empty task bodies, since only the
 * hang/crash mattered there), Send fills its buffer with a per-round marker
 * and Recv verifies it, so this also catches silent data corruption, not
 * just hangs/crashes.
 */

#include <starpu.h>
#include <starpu_mpi.h>

#include <mpi.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HALO_ELEMS 64 /* doubles per exchanged buffer, matches halo_ring */

static atomic_int g_mismatches = 0;

static void send_cpu_func(void *buffers[], void *cl_arg)
{
    /* buffers[0] is the zero-payload anchor: nothing to read/write. */
    double *halo = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    long    marker;
    starpu_codelet_unpack_args(cl_arg, &marker);
    for (int e = 0; e < HALO_ELEMS; e++) {
        halo[e] = (double)marker;
    }
}

static void recv_cpu_func(void *buffers[], void *cl_arg)
{
    double *halo = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    long    expected;
    starpu_codelet_unpack_args(cl_arg, &expected);
    for (int e = 0; e < HALO_ELEMS; e++) {
        if (halo[e] != (double)expected) {
            atomic_fetch_add(&g_mismatches, 1);
            break;
        }
    }
}

static struct starpu_codelet send_cl = {
    .cpu_funcs      = {send_cpu_func},
    .cpu_funcs_name = {"send_cpu_func"},
    .nbuffers       = 2,
    .modes          = {STARPU_RW, STARPU_W},
};

static struct starpu_codelet recv_cl = {
    .cpu_funcs      = {recv_cpu_func},
    .cpu_funcs_name = {"recv_cpu_func"},
    .nbuffers       = 2,
    .modes          = {STARPU_RW, STARPU_R},
};

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--nt N] [--nsteps N]\n"
            "  --nt      number of ring tiles, must be a multiple of the MPI world size (default 8)\n"
            "  --nsteps  number of independent Send/Recv rounds (default 200)\n",
            prog);
}

int main(int argc, char *argv[])
{
    int nt     = 8;
    int nsteps = 200;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--nt") == 0 && a + 1 < argc) {
            nt = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--nsteps") == 0 && a + 1 < argc) {
            nsteps = atoi(argv[++a]);
        } else if (strcmp(argv[a], "-h") == 0 || strcmp(argv[a], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    int ret = starpu_mpi_init_conf(&argc, &argv, 1 /* initialize_mpi */, MPI_COMM_WORLD, NULL);
    if (ret != 0) {
        fprintf(stderr, "starpu_mpi_init_conf failed: %d\n", ret);
        return 1;
    }

    int rank, world;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &world);

    if (nt % world != 0) {
        if (rank == 0) {
            fprintf(stderr, "--nt (%d) must be a multiple of the MPI world size (%d)\n", nt, world);
        }
        starpu_mpi_shutdown();
        return 1;
    }

    if (rank == 0) {
        fprintf(stdout, "starpu_halo_ring: world=%d nt=%d nsteps=%d\n", world, nt, nsteps);
        fflush(stdout);
    }

    int block = nt / world;
    #define OWNER(t) ((t) / block)

    /* One zero-payload anchor per tile, identical on every rank -- pins
     * Send(k,j)/Recv(k,i) execution to the rank owning tile j/i. */
    starpu_data_handle_t *anchor = calloc((size_t)nt, sizeof(*anchor));
    starpu_mpi_tag_t      tag    = 0;
    for (int t = 0; t < nt; t++) {
        starpu_void_data_register(&anchor[t]);
        starpu_mpi_data_register(anchor[t], tag++, OWNER(t));
    }

    /* One payload vector per (k, j): produced by Send(k,j) on OWNER(j),
     * consumed by Recv(k, (j+1) % nt) on OWNER((j+1) % nt). Only the owner
     * rank allocates real backing storage; other ranks register a handle
     * with no local copy (home_node = -1), matching StarPU-MPI's standard
     * "everyone builds the same distribution, only the owner has data"
     * pattern (see e.g. StarPU's stencil5 example). */
    starpu_data_handle_t **halo = calloc((size_t)nsteps, sizeof(*halo));
    double               ***halo_buf = calloc((size_t)nsteps, sizeof(*halo_buf));
    for (int k = 0; k < nsteps; k++) {
        halo[k]     = calloc((size_t)nt, sizeof(**halo));
        halo_buf[k] = calloc((size_t)nt, sizeof(**halo_buf));
        for (int j = 0; j < nt; j++) {
            int owner = OWNER(j);
            if (owner == rank) {
                halo_buf[k][j] = malloc(HALO_ELEMS * sizeof(double));
                starpu_vector_data_register(&halo[k][j], STARPU_MAIN_RAM, (uintptr_t)halo_buf[k][j],
                                             HALO_ELEMS, sizeof(double));
            } else {
                starpu_vector_data_register(&halo[k][j], -1, 0, HALO_ELEMS, sizeof(double));
            }
            starpu_mpi_data_register(halo[k][j], tag++, owner);
        }
    }

    for (int k = 0; k < nsteps; k++) {
        for (int j = 0; j < nt; j++) {
            long marker = (long)k * nt + j;
            starpu_mpi_task_insert(MPI_COMM_WORLD, &send_cl, STARPU_RW, anchor[j], STARPU_W, halo[k][j],
                                    STARPU_VALUE, &marker, sizeof(marker), 0);
        }
        for (int i = 0; i < nt; i++) {
            int  jprev    = (i - 1 + nt) % nt;
            long expected = (long)k * nt + jprev;
            starpu_mpi_task_insert(MPI_COMM_WORLD, &recv_cl, STARPU_RW, anchor[i], STARPU_R, halo[k][jprev],
                                    STARPU_VALUE, &expected, sizeof(expected), 0);
        }
    }

    /* Printed per-rank, unbuffered: a rank that hangs waiting on a message
     * that never arrives will print "entering" but never "returned from",
     * distinguishing a genuine hang from a crash. */
    fprintf(stderr, "[%d] entering starpu_task_wait_for_all\n", rank);
    fflush(stderr);
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
    fprintf(stderr, "[%d] returned from starpu_task_wait_for_all\n", rank);
    fflush(stderr);

    int total_mismatches = 0;
    int local_mismatches = atomic_load(&g_mismatches);
    MPI_Reduce(&local_mismatches, &total_mismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int k = 0; k < nsteps; k++) {
        for (int j = 0; j < nt; j++) {
            starpu_data_unregister(halo[k][j]);
            free(halo_buf[k][j]);
        }
        free(halo[k]);
        free(halo_buf[k]);
    }
    free(halo);
    free(halo_buf);
    for (int t = 0; t < nt; t++) {
        starpu_data_unregister(anchor[t]);
    }
    free(anchor);

    if (rank == 0) {
        if (total_mismatches == 0) {
            fprintf(stdout, "starpu_halo_ring: completed %d steps without error\n", nsteps);
        } else {
            fprintf(stdout, "starpu_halo_ring: completed %d steps with %d DATA MISMATCHES\n", nsteps,
                    total_mismatches);
        }
        fflush(stdout);
    }

    starpu_mpi_shutdown();

    return total_mismatches != 0;
}
