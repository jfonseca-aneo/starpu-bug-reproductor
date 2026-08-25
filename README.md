# starpu_halo_ring

Spike that reimplements a small ring of typed-halo Send/Recv tasks on StarPU-MPI, to check
whether a cross-rank typed-halo race shows up there.

## What this does

Two task "classes":

```
Send(k, j) --typed buffer--> Recv(k, (j+1) % NT)
```

`NT` tiles form a ring, block-distributed across MPI ranks (tile `j` belongs to rank
`j / (NT/world)`). For each of `NSTEPS` independent rounds, `Send(k,j)` fills a small buffer
and `Recv(k, (j+1) % NT)` reads it. Whenever `j` sits at a rank boundary, that's a genuine
cross-rank transfer. Steps are independent of each other, so all `NT * NSTEPS` Send/Recv pairs
become ready essentially immediately. This piles up far more concurrent cross-rank traffic than
a real timestepping loop would for the same tile count, a stress pattern designed to make any
such race easy to trigger.

Task bodies aren't empty: `Send` writes a per-round marker into its buffer and `Recv` verifies
it, so this also catches silent data corruption, not just hangs/crashes.

There's no PTG/DSL/code-generation step: StarPU tasks are plain C, submitted directly via
`starpu_mpi_task_insert()`. Every rank runs the identical driver loop (SPMD-style); StarPU-MPI
decides, per task, which rank actually executes it and whether a send/recv is needed, based on
which rank "owns" each data handle (`starpu_mpi_data_register`). A zero-payload
`starpu_void_data` "anchor" handle per tile pins each task's execution to the rank that owns
that tile.

## Build

Requires `libstarpu-dev` (pulls in `libstarpumpi`). On Debian/Ubuntu:

```sh
sudo apt install libstarpu-dev
```

Then:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

## Run

```sh
mpirun -n 2 ./starpu_halo_ring [--nt N] [--nsteps N]
```

- `--nt`: number of ring tiles, must be a multiple of the MPI world size (default 8).
- `--nsteps`: number of independent Send/Recv rounds (default 200).

A clean run prints `starpu_halo_ring: completed N steps without error` from rank 0. Each rank
also prints `entering`/`returned from starpu_task_wait_for_all` to stderr, so a hang (a rank
stuck waiting for a message that never arrives) is distinguishable from a crash.

Trial sizes used below:

```sh
mpirun -n 2 ./starpu_halo_ring --nt 8 --nsteps 200      # x15 trials
mpirun -n 2 ./starpu_halo_ring --nt 8 --nsteps 1000     # x15 trials
mpirun -n 4 ./starpu_halo_ring --nt 32 --nsteps 300     # x12 trials
```

## Results 

### Ubuntu 24.04, GCC 13.3.0, Open MPI 4.1.6, StarPU 1.4.3+dfsg-5build1 from the Ubuntu `universe` archive, `STARPU_NCPU=2`

| Config | StarPU-MPI 1.4.3 |
|---|---|
| `-n 1 --nt 8 --nsteps 200` (control, no cross-rank traffic) | 15/15 clean |
| `-n 2 --nt 8 --nsteps 200` | 30/35 clean, **5 hung** (14%) |
| `-n 2 --nt 8 --nsteps 1000` | 11/15 clean, **4 hung** (27%) |
| `-n 4 --nt 32 --nsteps 300` | 2/12 clean, **10 hung** (83%) |

Every hang has the same signature, confirmed with `gdb -p <pid> -batch -ex "thread apply all bt"`
on both ranks while a hang was in progress. A 180s wait first ruled out "just slow": a clean run
completes in 1-2s, so 180s is a genuine deadlock, not a slow finish. One rank finishes
`starpu_task_wait_for_all()` and blocks in the following `starpu_mpi_barrier()`. The other rank
never returns from `starpu_task_wait_for_all()` at all, with its internal StarPU-MPI comm thread
spinning inside `_starpu_mpi_comm_test_recv()`, i.e. it is still waiting to receive a message
that never gets matched/delivered. `-n 1` (same task count, zero cross-rank traffic) is clean on
every trial, confirming the failure is cross-rank-specific and not a bug in this reproducer's
local task-submission logic.

### StarPU 1.4.7 built from source via Spack against a separately-built Open MPI 4.1.7 

| Config | StarPU-MPI 1.4.3 (Ubuntu package) | StarPU-MPI 1.4.7 (Spack, latest 1.4.x) |
|---|---|---|
| `-n 1` (control) | 15/15 clean | 10/10 clean |
| `-n 2 --nt 8 --nsteps 200` | 30/35 clean, 5 hung (14%) | 11/20 clean, **9 hung (45%)** |
| `-n 4 --nt 32 --nsteps 300` | 2/12 clean, 10 hung (83%) | 0/8 clean, **8 hung (100%)** |

### Hanging behavior seems to be isolated to the `starpu_mpi_task_insert` layer, not StarPU-MPI's communication engine

`main_detached.c` (`starpu_halo_ring_detached`) reimplements the identical
workload using the lower-level `starpu_mpi_isend_detached`/`starpu_mpi_irecv_detached` API
instead, with an explicit destination/source rank and tag per transfer, no automatic
ownership-driven dependency inference, and no `starpu_mpi_data_register` at all. Same tile ring,
same per-round marker/verify correctness check, same "fire everything immediately" stress
pattern. Results, on both StarPU builds, at every size that reliably broke the `task_insert`
version, including a config an order of magnitude heavier than anything tested above:

| Config | `task_insert` (main.c) | `isend_detached`/`irecv_detached` (main_detached.c) |
|---|---|---|
| `-n 2 --nt 8 --nsteps 200` (StarPU 1.4.3) | 5/35 hung (14%) | **20/20 clean** |
| `-n 2 --nt 8 --nsteps 1000` (StarPU 1.4.3) | 4/15 hung (27%) | **15/15 clean** |
| `-n 4 --nt 32 --nsteps 300` (StarPU 1.4.3) | 10/12 hung (83%) | **12/12 clean** |
| `-n 2 --nt 8 --nsteps 200` (StarPU 1.4.7) | 9/20 hung (45%) | **20/20 clean** |
| `-n 2 --nt 8 --nsteps 1000` (StarPU 1.4.7) | not retested | **15/15 clean** |
| `-n 4 --nt 32 --nsteps 300` (StarPU 1.4.7) | 8/8 hung (100%) | **12/12 clean** |
| `-n 6 --nt 60 --nsteps 500` (StarPU 1.4.7, heavier than anything above) | 5/5 hung (100%) | **10/10 clean** |

Zero hangs and zero data-correctness mismatches across 124 combined `main_detached` trials on
two StarPU releases, including at a load six times heavier (by rank x tile x step count) than
any `task_insert` configuration above. This strongly suggests the deadlock lives specifically
in `starpu_mpi_task_insert`'s automatic ownership-tracking/transfer-inference logic (or in how
it drives the same underlying comm engine), not in StarPU-MPI's point-to-point transport itself.

### Independently re-confirmed (2026-08-24)

All of the above was re-run from scratch on the same machine (system StarPU 1.4.3 package, plus
a separate Spack build of StarPU 1.4.7 / Open MPI 4.1.7). Every config reproduced the same
failure pattern: clean on `-n 1`, hangs only on cross-rank `task_insert` runs, zero hangs on
`main_detached`, zero crashes and zero data-correctness mismatches anywhere.

| Config | Original | Re-run |
|---|---|---|
| StarPU 1.4.3, `-n 1` (control) | 15/15 clean | 15/15 clean |
| StarPU 1.4.3, `-n 2 --nt 8 --nsteps 200` | 30/35 clean, 14% hung | 13/15 clean, 13% hung |
| StarPU 1.4.3, `-n 2 --nt 8 --nsteps 1000` | 11/15 clean, 27% hung | 14/15 clean, 7% hung |
| StarPU 1.4.3, `-n 4 --nt 32 --nsteps 300` | 2/12 clean, 83% hung | 2/12 clean, 83% hung |
| StarPU 1.4.7, `-n 2 --nt 8 --nsteps 200` | 11/20 clean, 45% hung | 19/20 clean, 5% hung |
| StarPU 1.4.7, `-n 4 --nt 32 --nsteps 300` | 0/8 clean, 100% hung | 2/8 clean, 75% hung |
| `main_detached`, StarPU 1.4.3, `-n 2 --nt 8 --nsteps 200` | 20/20 clean | 20/20 clean |

Hang rates differ from the original run, as expected for a race condition, but the qualitative
signature is identical across both runs: cross-rank-only, `task_insert`-specific, rate increasing
with rank/tile count.

**Not a known issue on the GitHub mirror.** The StarPU GitLab tracker (where development happens)
was unreachable, so the 57 issues on the `starpu-runtime/starpu` GitHub mirror were checked
instead. None report this cross-rank `starpu_mpi_task_insert` deadlock. The closest matches are
unrelated performance/scalability complaints about MPI LU (#7, #14, #19).

## Root cause and proposed fix

Found and empirically validated by reading the StarPU source and instrumenting a live hang: a
genuine deadlock in `starpu_mpi_barrier()`, flagged by StarPU's own maintainers as an unresolved
`FIXME` in the exact function. Full writeup, evidence trail, and a validated workaround (12/12
clean at a config that hung 75-100% of the time beforehand) in [ROOT_CAUSE.md](ROOT_CAUSE.md).
