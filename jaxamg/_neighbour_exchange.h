// Neighbour-only exchange of packed buffers.
//
// A plan names this rank's send and receive peers with per-peer element counts,
// in ascending peer order, over a communicator duplicated for the transport (so
// its messages never match AmgX's or the caller's). One call posts every
// receive and send (MPI_Irecv / MPI_Isend) and waits once: per exchange, a rank
// talks only to its peers, in one latency. The reverse direction swaps the
// roles, which is the exchange's transpose. Plans are registered once on the
// host and named by an id; the JAX side passes the id as a static attribute
// and threads mpi4jax's ordering token through the call.
//
// Buffers may be longer than the plan's totals (the sharded layout pads them to
// a global maximum so every rank's program has one shape). Only the plan's
// prefix is sent; the received buffer's tail beyond the plan is zeroed.
#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace jaxamg_exchange
{
  namespace ffi = xla::ffi;

  struct Side
  {
    std::vector<int> peers;
    std::vector<int64_t> counts;
    std::vector<int64_t> displs;
    int64_t total = 0;
  };

  struct Plan
  {
    MPI_Comm comm;
    Side send;
    Side recv;
  };

  inline std::mutex &PlanMutex()
  {
    static std::mutex m;
    return m;
  }

  inline std::unordered_map<int64_t, Plan> &Plans()
  {
    static std::unordered_map<int64_t, Plan> plans;
    return plans;
  }

  inline Side MakeSide(const std::vector<int> &peers, const std::vector<int64_t> &counts)
  {
    Side side;
    side.peers = peers;
    side.counts = counts;
    side.displs.resize(peers.size());
    for (size_t i = 0; i < peers.size(); ++i)
    {
      side.displs[i] = side.total;
      side.total += counts[i];
    }
    return side;
  }

  // Register a plan under a caller-chosen id (the Python layer allocates ids in
  // the same order on every rank). The communicator handle is MPI's own
  // (mpi4py's ``MPI._handleof``); a C-style cast serves pointer and integer
  // MPI_Comm types alike.
  inline void RegisterPlan(int64_t plan_id, uint64_t comm_handle,
                           const std::vector<int> &send_peers,
                           const std::vector<int64_t> &send_counts,
                           const std::vector<int> &recv_peers,
                           const std::vector<int64_t> &recv_counts)
  {
    if (send_peers.size() != send_counts.size() || recv_peers.size() != recv_counts.size())
      throw std::invalid_argument("exchange plan: peers and counts differ in length");
    Plan plan;
    plan.comm = (MPI_Comm)(uintptr_t)comm_handle;
    plan.send = MakeSide(send_peers, send_counts);
    plan.recv = MakeSide(recv_peers, recv_counts);
    std::lock_guard<std::mutex> lock(PlanMutex());
    Plans()[plan_id] = std::move(plan);
  }

  inline void ReleasePlan(int64_t plan_id)
  {
    std::lock_guard<std::mutex> lock(PlanMutex());
    Plans().erase(plan_id);
  }

  inline size_t PlanCount()
  {
    std::lock_guard<std::mutex> lock(PlanMutex());
    return Plans().size();
  }

  constexpr int kTag = 7201;

  inline ffi::Error Exchange(cudaStream_t stream, ffi::AnyBuffer input,
                             ffi::Result<ffi::AnyBuffer> output, int64_t plan_id,
                             int32_t reverse, int32_t device_buffers)
  {
    Plan plan;
    {
      std::lock_guard<std::mutex> lock(PlanMutex());
      auto it = Plans().find(plan_id);
      if (it == Plans().end())
        return ffi::Error::InvalidArgument("neighbour exchange: unknown plan id " +
                                           std::to_string(plan_id));
      plan = it->second;
    }
    const Side &out_side = reverse ? plan.recv : plan.send;  // what this rank sends
    const Side &in_side = reverse ? plan.send : plan.recv;   // what it receives
    if (input.element_type() != output->element_type())
      return ffi::Error::InvalidArgument("neighbour exchange: dtype mismatch");
    const size_t width = ffi::ByteWidth(input.element_type());
    const int64_t n_in = static_cast<int64_t>(input.element_count());
    const int64_t n_out = static_cast<int64_t>(output->element_count());
    if (n_in < out_side.total || n_out < in_side.total)
      return ffi::Error::InvalidArgument(
          "neighbour exchange: buffers shorter than the plan (send " +
          std::to_string(n_in) + " < " + std::to_string(out_side.total) + " or receive " +
          std::to_string(n_out) + " < " + std::to_string(in_side.total) + ")");

    char *send_dev = static_cast<char *>(input.untyped_data());
    char *recv_dev = static_cast<char *>(output->untyped_data());

    // The send buffer is produced on XLA's stream: finish it before MPI reads.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return ffi::Error::Internal("neighbour exchange: stream synchronization failed");

    std::vector<char> send_host, recv_host;
    char *send_buf = send_dev;
    char *recv_buf = recv_dev;
    if (!device_buffers)
    {
      send_host.resize(out_side.total * width);
      recv_host.resize(in_side.total * width);
      if (out_side.total &&
          cudaMemcpy(send_host.data(), send_dev, out_side.total * width,
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        return ffi::Error::Internal("neighbour exchange: device-to-host copy failed");
      send_buf = send_host.data();
      recv_buf = recv_host.data();
    }

    // Every argument is validated before anything is posted (the Python layer
    // refuses oversized messages on every rank first); once requests are
    // posted their buffers must outlive them, so an MPI failure is fatal.
    for (const Side *side : {&in_side, &out_side})
      for (int64_t count : side->counts)
        if (count * static_cast<int64_t>(width) > INT32_MAX)
          return ffi::Error::InvalidArgument("neighbour exchange: message over 2 GiB");
    auto fatal = [&](const char *what, int code) {
      char text[MPI_MAX_ERROR_STRING];
      int len = 0;
      MPI_Error_string(code, text, &len);
      std::fprintf(stderr, "jaxamg neighbour exchange: %s failed: %.*s\n", what, len,
                   text);
      MPI_Abort(plan.comm, code);
    };

    std::vector<MPI_Request> requests;
    requests.reserve(in_side.peers.size() + out_side.peers.size());
    for (size_t i = 0; i < in_side.peers.size(); ++i)
    {
      MPI_Request r;
      int code = MPI_Irecv(recv_buf + in_side.displs[i] * width,
                           static_cast<int>(in_side.counts[i] * width), MPI_BYTE,
                           in_side.peers[i], kTag, plan.comm, &r);
      if (code != MPI_SUCCESS)
        fatal("MPI_Irecv", code);
      requests.push_back(r);
    }
    for (size_t i = 0; i < out_side.peers.size(); ++i)
    {
      MPI_Request r;
      int code = MPI_Isend(send_buf + out_side.displs[i] * width,
                           static_cast<int>(out_side.counts[i] * width), MPI_BYTE,
                           out_side.peers[i], kTag, plan.comm, &r);
      if (code != MPI_SUCCESS)
        fatal("MPI_Isend", code);
      requests.push_back(r);
    }
    if (!requests.empty())
    {
      int code = MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                             MPI_STATUSES_IGNORE);
      if (code != MPI_SUCCESS)
        fatal("MPI_Waitall", code);
    }

    // On XLA's stream (a synchronous copy from pageable memory may return
    // before the data lands), then wait: recv_host is freed on return.
    if (!device_buffers && in_side.total &&
        (cudaMemcpyAsync(recv_dev, recv_host.data(), in_side.total * width,
                         cudaMemcpyHostToDevice, stream) != cudaSuccess ||
         cudaStreamSynchronize(stream) != cudaSuccess))
      return ffi::Error::Internal("neighbour exchange: host-to-device copy failed");
    if (n_out > in_side.total &&
        cudaMemsetAsync(recv_dev + in_side.total * width, 0,
                        (n_out - in_side.total) * width, stream) != cudaSuccess)
      return ffi::Error::Internal("neighbour exchange: zeroing the padding failed");
    return ffi::Error::Success();
  }

  inline ffi::Error ExchangeImpl(cudaStream_t stream, ffi::AnyBuffer input, ffi::Token,
                                 ffi::Result<ffi::AnyBuffer> output, ffi::Result<ffi::Token>,
                                 int64_t plan_id, int32_t reverse, int32_t device_buffers)
  {
    return Exchange(stream, input, output, plan_id, reverse, device_buffers);
  }

  XLA_FFI_DEFINE_HANDLER(
      NeighbourExchange,
      ExchangeImpl,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Arg<ffi::AnyBuffer>()   // packed send buffer
          .Arg<ffi::Token>()       // ordering token (mpi4jax's chain, or a fresh one)
          .Ret<ffi::AnyBuffer>()   // packed receive buffer
          .Ret<ffi::Token>()       // ordering token out
          .Attr<int64_t>("plan_id")
          .Attr<int32_t>("reverse")
          .Attr<int32_t>("device_buffers"));
} // namespace jaxamg_exchange
