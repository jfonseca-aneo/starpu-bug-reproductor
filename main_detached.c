/*
 * Lower-level counterpart of main.c: same ring-of-tiles workload, same
 * cross-rank stress pattern, but data movement uses StarPU-MPI's explicit
 * point-to-point API (starpu_mpi_isend_detached / starpu_mpi_irecv_detached)
 * instead of starpu_mpi_task_insert's automatic ownership-driven transfer
 * inference. This isolates whether the deadlock seen in main.c lives in the
 * high-level "insert a task, let the runtime figure out the DAG and the
 * transfers" layer, or in the lower-level communication engine itself that
 * layer is built on top of.
 *
 * Unlike main.c, there is no shared/symmetric "everyone registers a handle
 * for every tile" data distribution: each rank only registers handles for
 * the tiles it owns, and explicitly names the destination/source rank and a
 * matching tag for every cross-rank transfer -- the same shape a hand-rolled
 * MPI halo exchange would have, just going through StarPU's data handles
 * for buffer/interface bookkeeping instead of raw MPI_Isend/Irecv.
 *
 * Same-rank neighbor hops skip MPI entirely (a local memcpy), matching how
 * SeWaS's own SEWASSequential engine handles same-process tile-to-tile
 * transfers.
 */

#include <starpu.h>
#include <starpu_mpi.h>

#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HALO_ELEMS 64 /* doubles per exchanged buffer, matches halo_ring */

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
        fprintf(stdout, "starpu_halo_ring_detached: world=%d nt=%d nsteps=%d\n", world, nt, nsteps);
        fflush(stdout);
    }

    int block = nt / world;
    #define OWNER(t) ((t) / block)

    /* Only local tiles [rank*block, (rank+1)*block) are registered at all --
     * unlike main.c's symmetric distribution, no rank needs to know about
     * data it neither produces nor consumes. */
    double               ***send_buf = calloc((size_t)nsteps, sizeof(*send_buf));
    double               ***recv_buf = calloc((size_t)nsteps, sizeof(*recv_buf));
    starpu_data_handle_t **send_h    = calloc((size_t)nsteps, sizeof(*send_h));
    starpu_data_handle_t **recv_h    = calloc((size_t)nsteps, sizeof(*recv_h));

    for (int k = 0; k < nsteps; k++) {
        send_buf[k] = calloc((size_t)block, sizeof(**send_buf));
        recv_buf[k] = calloc((size_t)block, sizeof(**recv_buf));
        send_h[k]   = calloc((size_t)block, sizeof(**send_h));
        recv_h[k]   = calloc((size_t)block, sizeof(**recv_h));
        for (int lj = 0; lj < block; lj++) {
            int  j      = rank * block + lj;
            long marker = (long)k * nt + j;

            send_buf[k][lj] = malloc(HALO_ELEMS * sizeof(double));
            for (int e = 0; e < HALO_ELEMS; e++) {
                send_buf[k][lj][e] = (double)marker;
            }
            starpu_vector_data_register(&send_h[k][lj], STARPU_MAIN_RAM, (uintptr_t)send_buf[k][lj],
                                         HALO_ELEMS, sizeof(double));

            recv_buf[k][lj] = calloc(HALO_ELEMS, sizeof(double));
            starpu_vector_data_register(&recv_h[k][lj], STARPU_MAIN_RAM, (uintptr_t)recv_buf[k][lj],
                                         HALO_ELEMS, sizeof(double));
        }
    }

    for (int k = 0; k < nsteps; k++) {
        for (int lj = 0; lj < block; lj++) {
            int j     = rank * block + lj;
            int jnext = (j + 1) % nt;
            int rnext = OWNER(jnext);
            if (rnext == rank) {
                int li_next = jnext - rank * block;
                memcpy(recv_buf[k][li_next], send_buf[k][lj], HALO_ELEMS * sizeof(double));
            } else {
                starpu_mpi_tag_t tag = (starpu_mpi_tag_t)((long)k * nt + j);
                starpu_mpi_isend_detached(send_h[k][lj], rnext, tag, MPI_COMM_WORLD, NULL, NULL);
            }
        }
        for (int li = 0; li < block; li++) {
            int i     = rank * block + li;
            int jprev = (i - 1 + nt) % nt;
            int rprev = OWNER(jprev);
            if (rprev != rank) {
                starpu_mpi_tag_t tag = (starpu_mpi_tag_t)((long)k * nt + jprev);
                starpu_mpi_irecv_detached(recv_h[k][li], rprev, tag, MPI_COMM_WORLD, NULL, NULL);
            }
            /* rprev == rank: already filled by the memcpy above in this same round. */
        }
    }

    /* Printed per-rank, unbuffered: a rank that hangs waiting on a message
     * that never arrives will print "entering" but never "returned from",
     * distinguishing a genuine hang from a crash. */
    fprintf(stderr, "[%d] entering starpu_mpi_wait_for_all\n", rank);
    fflush(stderr);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    fprintf(stderr, "[%d] returned from starpu_mpi_wait_for_all\n", rank);
    fflush(stderr);

    int local_mismatches = 0;
    for (int k = 0; k < nsteps; k++) {
        for (int li = 0; li < block; li++) {
            int  i        = rank * block + li;
            int  jprev    = (i - 1 + nt) % nt;
            long expected = (long)k * nt + jprev;
            for (int e = 0; e < HALO_ELEMS; e++) {
                if (recv_buf[k][li][e] != (double)expected) {
                    local_mismatches++;
                    break;
                }
            }
        }
    }

    int total_mismatches = 0;
    MPI_Reduce(&local_mismatches, &total_mismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int k = 0; k < nsteps; k++) {
        for (int lj = 0; lj < block; lj++) {
            starpu_data_unregister(send_h[k][lj]);
            starpu_data_unregister(recv_h[k][lj]);
            free(send_buf[k][lj]);
            free(recv_buf[k][lj]);
        }
        free(send_buf[k]);
        free(recv_buf[k]);
        free(send_h[k]);
        free(recv_h[k]);
    }
    free(send_buf);
    free(recv_buf);
    free(send_h);
    free(recv_h);

    if (rank == 0) {
        if (total_mismatches == 0) {
            fprintf(stdout, "starpu_halo_ring_detached: completed %d steps without error\n", nsteps);
        } else {
            fprintf(stdout, "starpu_halo_ring_detached: completed %d steps with %d DATA MISMATCHES\n",
                    nsteps, total_mismatches);
        }
        fflush(stdout);
    }

    starpu_mpi_shutdown();

    return total_mismatches != 0;
}
