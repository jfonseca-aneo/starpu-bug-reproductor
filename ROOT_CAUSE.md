# Root cause and proposed fix

Confirmed against StarPU 1.4.7 (tag `starpu-1.4.7`, vendored locally for this investigation)
by combining a live captured trace with direct source reading. This is a genuine deadlock in
`starpu_mpi_barrier()`, not a data race, and not specific to any one StarPU release: the code
path involved is unchanged across the 1.4.x line, and the maintainers have an unresolved `FIXME`
sitting in the exact function.

## Mechanism

1. `starpu_mpi_task_insert`'s automatic implicit-transfer machinery dispatches sends
   asynchronously: once a producing task's data is ready, the send request is queued onto
   `ready_send_requests`, to be dispatched later by StarPU-MPI's single background progress
   thread. Under `coop_sends` batching this queueing can be deferred quite late — we captured a
   case where a send bound for a peer rank wasn't pushed onto the queue until essentially the
   very end of the run.
2. `starpu_mpi_barrier()` (`mpi/src/mpi/starpu_mpi_mpi.c:802`) submits its own internal request
   (`BARRIER_REQ`). `_starpu_mpi_submit_ready_request`'s routing only special-cases `SEND_REQ`
   (line 301: pushed to `ready_send_requests`); every other request type, `BARRIER_REQ` included,
   falls to the `else` branch and lands on `ready_recv_requests` (line 303).
3. The progress thread's main loop (`mpi/src/mpi/starpu_mpi_mpi.c:1363-1400`) fully drains
   `ready_recv_requests` (up to `STARPU_MPI_NREADY_PROCESS` items) *before* it ever looks at
   `ready_send_requests`, within the same loop iteration.
4. `BARRIER_REQ`'s handler, `_starpu_mpi_barrier_func` (line 785), calls a blocking, collective
   `MPI_Barrier()` (line 795) directly on the progress thread. The surrounding comment is the
   maintainers' own acknowledgement of this hazard:

   ```c
   /* FIXME: rather use MPI_Ibarrier and make it a detached request.
    * We'd then be able to introduce starpu_mpi_ibarrier, and make
    * starpu_mpi_barrier just call starpu_mpi_ibarrier(); starpu_mpi_wait();
    * That'll solve locking issue when intermixing starpu_mpi_barrier with
    * other communications.
    */
   ```
5. If, at the moment one rank's application thread calls `starpu_mpi_barrier()`, that rank still
   has an outbound send sitting un-dispatched in `ready_send_requests`, the progress thread
   commits to the blocking `MPI_Barrier()` before it ever reaches the send-draining loop that
   follows it in the same function. That send is now permanently stranded.
6. The destination rank can't complete its own `starpu_task_wait_for_all()` — it's still waiting
   for exactly that stranded send — so it never reaches its own `starpu_mpi_barrier()` call, and
   the barrier never completes on the sending rank either. Deadlock.

This explains every symptom in [README.md](README.md):
- Silent hang, zero crashes, zero data corruption: nothing is corrupted, a legitimate request is
  simply stranded.
- Cross-rank only: `-n 1` has no sends to strand.
- Gets worse with heavier configs: more concurrent traffic per destination raises the odds a send
  (or a `coop_sends` batch) is still queued at the exact moment the barrier is entered.
- Absent from `main_detached.c`: it calls `starpu_mpi_wait_for_all()`, not
  `starpu_mpi_barrier()` — a purely local wait on a request counter, with no blocking collective
  call inside the progress thread to create this hazard.

## Evidence trail

Captured with `STARPU_MPI_COMM=1 STARPU_MPI_DEBUG_LEVEL_MAX=5` (StarPU built with
`--enable-mpi-verbose`) on a hung `-n 4 --nt 32 --nsteps 300` trial: rank 3's orphaned request
`0x1dd0d480` (`RECV_REQ`, tag 9623, source rank 2) was found by diffing "posted" against
"completed" request addresses in its log — the one request posted but never completed. Rank 2's
log shows the matching `SEND_REQ` (`0x1b455fa0`, tag 9623, dest rank 3) created under
`coop_sends` early in the run, but only actually pushed onto `ready_send_requests` at line 8696
of an ~8700-line log — immediately followed by rank 2 submitting, then *handling*, its own
`BARRIER_REQ`, with no further activity from rank 2 afterward. Rank 2's progress thread reached
the barrier request first (queued on `ready_recv_requests`, drained before `ready_send_requests`)
and blocked inside `MPI_Barrier()`, stranding tag 9623 for good.

**Ruled out:** a real, upstream-acknowledged, unrelated race in `_starpu_mpi_isend_size_func`
(commit `9e4a9fc3d`, "Make sure to send envelope and data together" — present on `master` but
never backported to any 1.4.x release, confirmed via `git merge-base --is-ancestor` against every
1.4.x tag through 1.4.12) was the leading hypothesis at first. We built a patched StarPU 1.4.7
with that fix backported and reran the heaviest hang-prone config: 11/12 trials still hung. That
commit is a real, separate defect worth keeping fixed upstream, but it is not the cause of this
hang.

## Proposed fix

Two independent options:

1. **Upstream, general fix** (matches the maintainers' own `FIXME`): change
   `_starpu_mpi_barrier_func` to issue a non-blocking `MPI_Ibarrier()` and complete the request
   asynchronously (polled from the progress loop like other requests), instead of a blocking
   `MPI_Barrier()` executed directly on the progress thread. This lets the progress thread keep
   draining `ready_send_requests` while the barrier is in flight, closing this whole class of
   deadlock with no behavior change for the non-racing case.
2. **Immediate, validated workaround for any `task_insert`-based program**: don't pair
   `starpu_task_wait_for_all()` with `starpu_mpi_barrier()`. Use `starpu_mpi_wait_for_all(comm)`
   instead — it only waits locally for `posted_requests == 0`, with no blocking collective call
   on the progress thread. We patched a copy of this reproducer's `main.c` this way and reran the
   heaviest hang-prone config (`-n 4 --nt 32 --nsteps 300`, 75-100% hung on this same StarPU 1.4.7
   build): **12/12 clean, zero hangs, zero mismatches.**

`main.c` itself is left unmodified here since its purpose is to reproduce the bug, not to work
around it.
