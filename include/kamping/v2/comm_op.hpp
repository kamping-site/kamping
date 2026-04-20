#pragma once

namespace kamping::comm_op {

struct recv {};
struct bcast {};
struct allgather {};
struct allgatherv {};
struct alltoall {};
struct alltoallv {};
struct sendrecv {};
struct reduce {};
struct allreduce {};
struct gather {};
struct gatherv {};
struct scatter {};
struct scatterv {};
struct scan {};
struct exscan {};
} // namespace kamping::comm_op
