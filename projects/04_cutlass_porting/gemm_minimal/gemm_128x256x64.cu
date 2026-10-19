// A deliberately simple CuTe Tensor Core GEMM, fixed-size and unpipelined.
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
constexpr int M = 128, N = 256, K = 64;
constexpr int BM = 64, BN = 128, BK = 32, Threads = 128;
using Half = cute::half_t;
using Mma = decltype(make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},
    Layout<Shape<_2, _2, _1>>{}));

void check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}
void require(bool ok, char const* msg) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

// These are logical coordinates, independent of element type and memory space.
void verify_mapping() {
  auto la = make_layout(make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
  auto lb = make_layout(make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
  auto lc = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, _1{}));
  require(la(67, 37) == 4325, "A address");
  require(lb(141, 37) == 9061, "B address");
  require(lc(67, 141) == 17293, "C address");
  std::printf("A layout: "); print(la); std::printf("\nB layout: "); print(lb);
  std::printf("\nC layout: "); print(lc); std::printf("\n");

  Mma mma;
  CUTE_STATIC_ASSERT_V(size(mma) == Int<Threads>{});
  auto idA = make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{}));
  auto idB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
  auto idC = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
  std::vector<int> seenA(BM * BK), seenB(BN * BK), seenC(BM * BN);
  for (int t = 0; t < Threads; ++t) {
    auto thr = mma.get_slice(t);
    auto a = thr.partition_A(idA);
    auto b = thr.partition_B(idB);
    auto c = thr.partition_C(idC);
    CUTE_STATIC_ASSERT_V(size<0>(a) == _8{});
    CUTE_STATIC_ASSERT_V(size<1>(a) == _2{});
    CUTE_STATIC_ASSERT_V(size<2>(a) == _2{});
    CUTE_STATIC_ASSERT_V(size<0>(b) == _4{});
    CUTE_STATIC_ASSERT_V(size<1>(b) == _8{});
    CUTE_STATIC_ASSERT_V(size<2>(b) == _2{});
    CUTE_STATIC_ASSERT_V(size<0>(c) == _4{});
    CUTE_STATIC_ASSERT_V(size<1>(c) == _2{});
    CUTE_STATIC_ASSERT_V(size<2>(c) == _8{});
    if (t == 0) {
      std::printf("partition_A: "); print(a.layout());
      std::printf("\npartition_B: "); print(b.layout());
      std::printf("\npartition_C: "); print(c.layout()); std::printf("\n");
    }
    int lane = t % 32, wm = (t / 32) % 2, wn = t / 64;
    int x = lane % 4, y = lane / 4;
    for (int mi = 0; mi < 2; ++mi)
      for (int ki = 0; ki < 2; ++ki)
        for (int v = 0; v < 8; ++v) {
          auto p = a(v, mi, ki);
          int m = y + 8 * ((v / 2) % 2) + 16 * wm + 32 * mi;
          int k = 2 * x + v % 2 + 8 * (v / 4) + 16 * ki;
          require(get<0>(p) == m && get<1>(p) == k, "A partition formula");
          ++seenA[m * BK + k];
        }
    for (int ni = 0; ni < 8; ++ni)
      for (int ki = 0; ki < 2; ++ki)
        for (int v = 0; v < 4; ++v) {
          auto p = b(v, ni, ki);
          int n = y + 8 * wn + 16 * ni;
          int k = 2 * x + v % 2 + 8 * (v / 2) + 16 * ki;
          require(get<0>(p) == n && get<1>(p) == k, "B partition formula");
          ++seenB[n * BK + k];
        }
    for (int mi = 0; mi < 2; ++mi)
      for (int ni = 0; ni < 8; ++ni)
        for (int v = 0; v < 4; ++v) {
          auto p = c(v, mi, ni);
          int m = y + 8 * (v / 2) + 16 * wm + 32 * mi;
          int n = 2 * x + v % 2 + 8 * wn + 16 * ni;
          require(get<0>(p) == m && get<1>(p) == n, "C partition formula");
          ++seenC[m * BN + n];
        }
  }
  for (int count : seenA) require(count == 2, "A reused by two N warps");
  for (int count : seenB) require(count == 2, "B reused by two M warps");
  for (int count : seenC) require(count == 1, "C covered exactly once");
  std::printf("PASS: all A/B/C thread-coordinate formulas and coverage\n");
}

__global__ void gemm_kernel(Half const* A, Half const* B, float* C) {
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(Int<M>{}, Int<K>{}),
                        make_stride(Int<K>{}, _1{}));
  // B is mathematically (K,N) column-major, viewed by CuTe as (N,K).
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(Int<N>{}, Int<K>{}),
                        make_stride(Int<K>{}, _1{}));
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(Int<M>{}, Int<N>{}),
                        make_stride(Int<N>{}, _1{}));
  auto tile = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  auto gA = local_tile(mA, tile, coord, Step<_1, X, _1>{});
  auto gB = local_tile(mB, tile, coord, Step<X, _1, _1>{});
  auto gC = local_tile(mC, tile, coord, Step<_1, _1, X>{});

  __shared__ Half bufA[BM * BK];
  __shared__ Half bufB[BN * BK];
  auto sA = make_tensor(make_smem_ptr(bufA),
      make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, _1{})));
  auto sB = make_tensor(make_smem_ptr(bufB),
      make_layout(make_shape(Int<BN>{}, Int<BK>{}), make_stride(Int<BK>{}, _1{})));

  Mma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto tAsA = thr.partition_A(sA);
  auto tBsB = thr.partition_B(sB);
  auto tCgC = thr.partition_C(gC);
  auto rA = thr.make_fragment_A(tAsA);
  auto rB = thr.make_fragment_B(tBsB);
  auto acc = thr.make_fragment_C(tCgC);
  clear(acc);
  for (int kt = 0; kt < K / BK; ++kt) {
    // Cooperative scalar copy: physical offsets q=t+128*j, K contiguous.
    for (int q = threadIdx.x; q < BM * BK; q += Threads)
      sA(q / BK, q % BK) = gA(q / BK, q % BK, kt);
    for (int q = threadIdx.x; q < BN * BK; q += Threads)
      sB(q / BK, q % BK) = gB(q / BK, q % BK, kt);
    __syncthreads();
    copy(tAsA, rA);
    copy(tBsB, rB);
    gemm(mma, rA, rB, acc);
    __syncthreads();  // all readers finish before the next overwrite
  }
  copy(acc, tCgC);
}

int main(int argc, char** argv) {
  verify_mapping();
  if (argc > 1) return 0;  // --layout-only: no CUDA device required
  std::vector<Half> a(M * K), b(N * K);
  std::vector<float> c(M * N);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) a[m * K + k] = Half(float((m * 3 + k * 5) % 17 - 8) / 8);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) b[n * K + k] = Half(float((n * 7 + k * 3) % 19 - 9) / 8);
  Half *da, *db;
  float* dc;
  check(cudaMalloc(&da, a.size() * sizeof(Half)));
  check(cudaMalloc(&db, b.size() * sizeof(Half)));
  check(cudaMalloc(&dc, c.size() * sizeof(float)));
  check(cudaMemcpy(da, a.data(), a.size() * sizeof(Half), cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), b.size() * sizeof(Half), cudaMemcpyHostToDevice));
  check(cudaMemset(dc, 0xff, c.size() * sizeof(float))); // unwritten elements remain NaN
  gemm_kernel<<<dim3(M / BM, N / BN), Threads>>>(da, db, dc);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  check(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double max_error = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += double(float(a[m * K + k])) * double(float(b[n * K + k]));
      double error = std::abs(double(c[m * N + n]) - ref);
      require(std::isfinite(c[m * N + n]) && error <= 1e-5, "GPU vs CPU reference");
      if (error > max_error) max_error = error;
    }
  check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
  std::printf("PASS: all %d outputs; max_abs_error=%.9g\n", M * N, max_error);
}
