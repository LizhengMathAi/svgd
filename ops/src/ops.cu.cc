// #ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ops.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


/* share memory */
template<class T>
struct SharedMemory {
    __device__ inline operator T *() {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

/* outputTensor = inputTensor */
template <typename T>
void ComputeMethods<T>::identityLauncher(
  const Eigen::ThreadPoolDevice& device, const T* inputTensor, const int n, T* outputTensor) {
  for (int i = 0; i < n; i++) {
    outputTensor[i] = inputTensor[i];
  }
}

namespace identity{
  template <typename T>
  __global__ void kernel(
    const T* inputTensor, const int n, T* outputTensor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      outputTensor[i] = inputTensor[i];
    }
  }
}

template <typename T>
void ComputeMethods<T>::identityLauncher(
  const Eigen::GpuDevice& device, const T* inputTensor, const int n, T* outputTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  identity::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, n, outputTensor);
}

/* k[0] = argmax(vec) */
template <typename T>
void ComputeMethods<T>::reduceArgmaxLauncher(
  const Eigen::ThreadPoolDevice& device, const int sSize, const T* vec, const int n, int* k) {
  int ind = 0;
  for (int i = 1; i < n; i++) {
    if (vec[i] > vec[ind]) {
      ind = i;
    }
  }
  k[0] = ind;
}

namespace argmax{
  template <typename T>
  __device__ T execute(
    const int sSize, const int sCount, int *sVec, int tid, const T* vec, const int n) {
    sVec[tid] = tid;
    __syncthreads();
    for (int j = 1; j < sCount; j++) {
      if (j * sSize + tid < n) {
        sVec[tid] = (vec[j * sSize + tid] > vec[sVec[tid]]) ? j * sSize + tid : sVec[tid];
      }
      __syncthreads();
    }

    int t = sSize;
    
    while (t > 1) {
      int tail = t % 2;
      t /= 2;
      if (tid < t) {
        sVec[tid] = (vec[sVec[tid + t]] > vec[sVec[tid]]) ? sVec[tid + t] : sVec[tid];
      }
      if (tail == 1) {
        sVec[0] = (vec[sVec[2 * t]] > vec[sVec[0]]) ? sVec[2 * t] : sVec[0];
      }
      __syncthreads();
    }

    return (tid == 0) ? sVec[0] : 0;
  }

  template <typename T>
  __global__ void kernel(
    const int sSize, const T* vec, const int n, int* k) {
    int *sVec = SharedMemory<int>();
    int sCount = n / sSize + 1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sSize; i += blockDim.x * gridDim.x) {
      int r = execute(sSize, sCount, sVec, i, vec, n);
      if (i == 0) k[0] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::reduceArgmaxLauncher(
  const Eigen::GpuDevice& device, const int sSize, const T* vec, const int n, int* k) {
  CudaLaunchConfig config = GetCudaLaunchConfig(sSize, device);
  int sMemSize = sSize * sizeof(int);
  argmax::kernel<T><<<config.block_count, config.thread_per_block, sMemSize, device.stream()>>>(
    sSize, vec, n, k);
}

/* recv[0] = sum(vec) */
template <typename T>
void ComputeMethods<T>::reduceSumLauncher(
  const Eigen::ThreadPoolDevice& device, const int sSize, const T* vec, const int n, T* recv) {
  recv[0] = 0;
  for (int i = 0; i < n; i++) {
    recv[0] += vec[i];
  }
}

namespace sum{
  template <typename T>
  __device__ T execute(
    const int sSize, const int sCount, T *sVec, int tid, const T* vec, const int n) {
    T temp = 0;
    if (tid < sSize) {
      for (int j = 0; j < sCount; j++) {
        temp += (j * sSize + tid < n) ? vec[j * sSize + tid] : 0;
      }
      sVec[tid] = temp;
    }
    __syncthreads();

    int t = sSize;
    
    while (t > 1) {
      int tail = t % 2;
      t /= 2;
      if (tid < t) {
        sVec[tid] += sVec[tid + t];
      }
      if (tail == 1) {
        sVec[0] += sVec[2 * t];
      }
      __syncthreads();
    }

    return (tid == 0) ? sVec[0] : 0;
  }

  template <typename T>
  __global__ void kernel(
    const int sSize, const T* vec, const int n, T* recv) {
    T *sVec = SharedMemory<T>();
    int sCount = n / sSize + 1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      T r = execute(sSize, sCount, sVec, i, vec, n);
      if (i == 0) recv[0] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::reduceSumLauncher(
  const Eigen::GpuDevice& device, const int sSize, const T* vec, const int n, T* recv) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  int sMemSize = sSize * sizeof(T);
  sum::kernel<T><<<config.block_count, config.thread_per_block, sMemSize, device.stream()>>>(
    sSize, vec, n, recv);
}

/* recv[0] = sum(vec1 * vec2) */
template <typename T>
void ComputeMethods<T>::reduceInnerProductLauncher(
  const Eigen::ThreadPoolDevice& device, const int sSize, const T* vec1, const T* vec2, const int n, T* recv) {
  recv[0] = 0;
  for (int i = 0; i < n; i++) {
    recv[0] += vec1[i] * vec2[i];
  }
}

namespace innerProduct{
  template <typename T>
  __device__ T execute(
    const int sSize, const int sCount, T *sVec, int tid, const T* vec1, const T* vec2, const int n) {
    T temp = 0;
    if (tid < sSize) {
      for (int j = 0; j < sCount; j++) {
        temp += (j * sSize + tid < n) ? vec1[j * sSize + tid] * vec2[j * sSize + tid] : 0;
      }
      sVec[tid] = temp;
    }
    __syncthreads();

    int t = sSize;
    
    while (t > 1) {
      int tail = t % 2;
      t /= 2;
      if (tid < t) {
        sVec[tid] += sVec[tid + t];
      }
      if (tail == 1) {
        sVec[0] += sVec[2 * t];
      }
      __syncthreads();
    }

    return (tid == 0) ? sVec[0] : 0;
  }

  template <typename T>
  __global__ void kernel(
    const int sSize, const T* vec1, const T* vec2, const int n, T* recv) {
    T *sVec = SharedMemory<T>();
    int sCount = n / sSize + 1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      T r = execute(sSize, sCount, sVec, i, vec1, vec2, n);
      if (i == 0) recv[0] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::reduceInnerProductLauncher(
  const Eigen::GpuDevice& device, const int sSize, const T* vec1, const T* vec2, const int n, T* recv) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  int sMemSize = sSize * sizeof(T);
  innerProduct::kernel<T><<<config.block_count, config.thread_per_block, sMemSize, device.stream()>>>(
    sSize, vec1, vec2, n, recv);
}

/* recv[0] = sum(vec1 * vec3) / sum(vec2 * vec3) */
template <typename T>
void ComputeMethods<T>::reduceDoubleInnerProductLauncher(
  const Eigen::ThreadPoolDevice& device, const int sSize, const T* vec1, const T* vec2, const T* vec3, const int n, T* recv) {
  T temp1 = 0;
  T temp2 = 0;
  for (int i = 0; i < n; i++) {
    temp1 += vec1[i] * vec3[i];
    temp2 += vec2[i] * vec3[i];
  }
  recv[0] = temp1 / temp2;
}

namespace doubleInnerProduct{
  template <typename T>
  __device__ T execute(
    const int sSize, const int sCount, T *sVec, int tid, const T* vec1, const T* vec2, const T* vec3, const int n) {
    T temp1 = 0;
    T temp2 = 0;
    if (tid < sSize) {
      for (int j = 0; j < sCount; j++) {
        temp1 += (j * sSize + tid < n) ? vec1[j * sSize + tid] * vec3[j * sSize + tid] : 0;
        temp2 += (j * sSize + tid < n) ? vec2[j * sSize + tid] * vec3[j * sSize + tid] : 0;
      }
      sVec[2 * tid] = temp1;
      sVec[2 * tid + 1] = temp2;
    }
    __syncthreads();

    int t = sSize;
    
    while (t > 1) {
      int tail = t % 2;
      t /= 2;
      if (tid < t) {
        sVec[2 * tid] += sVec[2 * tid + 2 * t];
        sVec[2 * tid + 1] += sVec[2 * tid + 1 + 2 * t];
      }
      if (tail == 1) {
        sVec[0] += sVec[4 * t];
        sVec[1] += sVec[1 + 4 * t];
      }
      __syncthreads();
    }

    return (tid == 0) ? sVec[0] / sVec[1] : 0;
  }

  template <typename T>
  __global__ void kernel(
    const int sSize, const T* vec1, const T* vec2, const T* vec3, const int n, T* recv) {
    T *sVec = SharedMemory<T>();
    int sCount = n / sSize + 1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      T r = execute(sSize, sCount, sVec, i, vec1, vec2, vec3, n);
      if (i == 0) recv[0] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::reduceDoubleInnerProductLauncher(
  const Eigen::GpuDevice& device, const int sSize, const T* vec1, const T* vec2, const T* vec3, const int n, T* recv) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  int sMemSize = 2 * sSize * sizeof(T);
  doubleInnerProduct::kernel<T><<<config.block_count, config.thread_per_block, sMemSize, device.stream()>>>(
    sSize, vec1, vec2, vec3, n, recv);
}

/* rhs = vecs[0] + vecs[1] + ... */
template <typename T>
void ComputeMethods<T>::addNLauncher(
  const Eigen::ThreadPoolDevice& device, const T* vecs[], const int m, const int n, T* rhs) {
  for (int i = 0; i < n; i++) {
    rhs[i] = 0;
    for (int j = 0; j < m; j++) {
      rhs[i] += vecs[j][i];
    }
  }
}

namespace addN{
  template <typename T>
  __global__ void initKernel(T* rhs, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      rhs[i] = 0;
    }
  }

  template <typename T>
  __global__ void kernel(const T* vec, const int n, T* rhs) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      rhs[i] += vec[i];
    }
  }
}

template <typename T>
void ComputeMethods<T>::addNLauncher(
  const Eigen::GpuDevice& device, const T* vecs[], const int m, const int n, T* rhs) {
  CudaLaunchConfig init_config = GetCudaLaunchConfig(n, device);
  addN::initKernel<T><<<init_config.block_count, init_config.thread_per_block, 0, device.stream()>>>(
    rhs, n);
  for (int i = 0; i < m; i++) {
    CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
    addN::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
      vecs[i], n, rhs);
  }
}

/* recv = mat.T */
template <typename T>
void ComputeMethods<T>::transposeLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* mat, const int m, const int n, T* recv) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      recv[j * m + i] = mat[i * n + j];
    }
  }
}

namespace matmul{
  template <typename T>
  __global__ void transposeKernel(
    const T* mat, const int m, const int n, T* recv) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
      for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n; j += blockDim.y * gridDim.y) {
        recv[j * m + i] = mat[i * n + j];
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::transposeLauncher(
  const Eigen::GpuDevice& device, const T* mat, const int m, const int n, T* recv) {
  Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(m, n, device);
  matmul::transposeKernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    mat, m, n, recv);
}

/* recv = mat1@mat2.T */
template <typename T>
void ComputeMethods<T>::matMulLauncher(
  const Eigen::ThreadPoolDevice& device, const T* mat1, const T* mat2, const int m, const int n, const int l, T* recv) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < l; j++) {
      T r = 0;
      for (int k = 0; k < n; k++) {
        r += mat1[i * n + k] * mat2[j * n + k];
      }
      recv[i * l + j] = r;
    }
  }
}

namespace matmul{
  template <typename T>
  __global__ void kernel(
    const T* mat1, const T* mat2, const int m, const int n, const int l, T* recv) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
      for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < l; j += blockDim.y * gridDim.y) {
        T r = 0;
        for (int k = 0; k < n; k++) {
          r += mat1[i * n + k] * mat2[j * n + k];
        }
        recv[i * l + j] = r;
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::matMulLauncher(
  const Eigen::GpuDevice& device, 
  const T* mat1, const T* mat2, const int m, const int n, const int l, T* recv) {
  Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(m, l, device);
  matmul::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    mat1, mat2, m, n, l, recv);
}

/* recv = mat1 X mat2 */
template <typename T>
void ComputeMethods<T>::kroneckerProductLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* mat1, const T* mat2, const int m, const int n, const int p, const int q, T* recv) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int u = 0; u < p; u++) {
        for (int v = 0; v < q; v++) {
          int row = i * p + u;
          int col = j * q + v;
          recv[row * n * q + col] = mat1[i * n + j] * mat2[u * q + v];
        }
      }
    }
  }
}

namespace kroneckerProduct{
  template <typename T>
  __global__ void kernel(
    const T* mat1, const T* mat2, const int m, const int n, const int p, const int q, T* recv) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < m * p; row += blockDim.x * gridDim.x) {
      for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < n * q; col += blockDim.y * gridDim.y) {
        int i = row / p;
        int u = row % p;
        int j = col / q;
        int v = col % q;
        recv[row * n * q + col] = mat1[i * n + j] * mat2[u * q + v];
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::kroneckerProductLauncher(
  const Eigen::GpuDevice& device, 
  const T* mat1, const T* mat2, const int m, const int n, const int p, const int q, T* recv) {
  Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(m * p, n * q, device);
  kroneckerProduct::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    mat1, mat2, m, n, p, q, recv);
}

/* onehot: (Permutation matrix) P -> (indices) pi */
template <typename T>
void ComputeMethods<T>::oneHotLauncher(const Eigen::ThreadPoolDevice& device, const int* P, const int n, int* pi){
  for(int i = 0; i < n; i ++) {
    for (int j = 0; j < n; j++) {
      pi[i * n + j] = (j == P[i]) ? 1 : 0;
    }
  }
}

namespace plu{
  __global__ void oneHotKernel(const int* P, const int n, int* pi){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      for (int j = 0; j < n; j++) {
        pi[i * n + j] = (j == P[i]) ? 1 : 0;
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::oneHotLauncher(const Eigen::GpuDevice& device, const int* P, const int n, int* pi){
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  plu::oneHotKernel<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    P, n, pi);
}

/* L@U  = pi -> M */
template <typename T>
void ComputeMethods<T>::pluLauncher (
  const Eigen::ThreadPoolDevice& device, const T* M, const int n, int* pi, T* L, T* U){
  // initialize pi
  for (int i = 0; i < n; i++) {
    pi[i] = i;
  }

  // initialize U
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			U[i * n + j] = M[i * n + j];
		}
	}

  // plu
  for (int k = 0; k < n; k++) {
    T p = 0;
    int k_ = 0;

    for (int i = k; i < n; i++) {
      if (U[i * n + k] > p) {
        p = U[i * n + k];
        k_ = i;
      }
      else {
        if (U[i * n + k] < -p) {
          p = -U[i * n + k];
          k_ = i;
        }
      }
    }

    if (p == 0) {
      throw "singmatrixUlar matrix";
    }

    T temp = pi[k];
    pi[k] = pi[k_];
    pi[k_] = temp;

    for (int i = 0; i < n; i++) {
      temp = U[k * n + i];
      U[k * n + i] = U[k_ * n + i];
      U[k_ * n + i] = temp;
    }


    for (int i = k + 1; i < n; i++) {
      U[i * n + k] /= U[k * n + k];
      for (int j = k + 1; j < n; j++) {
        U[i * n + j] -= U[i * n + k] * U[k * n + j];
      }
    }
  }

  // initialize L
  for (int i = 0; i < n; i++){
    L[i * n + i] = 1;
    for (int j = 0; j < i; j++) {
      L[i * n + j] = U[i * n + j];
      U[i * n + j] = 0;
    }
    for (int j = i + 1; j < n; j++) {
      L[i * n + j] = 0;
    }
  }
}

namespace plu {
  template <typename T>
  __global__ void InitKernel(const T* M, const int n, int* pi, T* L, T* U) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      pi[i] = i;
      L[i * n + i] = 1;
      for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n; j += blockDim.y * gridDim.y) {
        U[i * n + j] = M[i * n + j];
      }
    }
  }

  // global variable, the index of the max elements in U[k:, k].
  __device__ int k_;

  // compute the index of the max elements in U[k:, k] by reduction method, 
  // A total of n - k threads and (n - k) * sizeof(int) share memory size.
  template <typename T>
  __global__ void ArgMaxKernel(T* U, const int n, int k) {
    int r = n - k;

    extern __shared__ int sInd[];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n - k; i += blockDim.x * gridDim.x) {
      sInd[i] = i + k;
    __syncthreads();

      while(r > 1) {
        int tail = r % 2;
        r = r / 2;

        if (i < r) {
          T tempi = U[sInd[i] * n + k] > 0 ? U[sInd[i] * n + k] : -U[sInd[i] * n + k];
          T tempin = U[sInd[i + r] * n + k] > 0 ? U[sInd[i + r] * n + k] : -U[sInd[i + r] * n + k];

          if (tempi < tempin) {
            sInd[i] = sInd[i + r];
          }
        }

        __syncthreads();
        if (tail == 1) {
          T temp0 = U[sInd[0] * n + k] > 0 ? U[sInd[0] * n + k] : -U[sInd[0] * n + k];
          T temp2n = U[sInd[2 * r] * n + k] > 0 ? U[sInd[2 * r] * n + k] : -U[sInd[2 * r] * n + k];

          if (temp0 < temp2n) {
            sInd[0] = sInd[2 * r];
          }
        }
        __syncthreads();
      }
    }
    k_ = sInd[0];
  }

  // swap rows of U, A total of n threads.
  template <typename T>
  __global__ void SwapKernel(int* pi, T* U, const int n, int k) {
    int temp = pi[k];
    pi[k] = pi[k_];
    pi[k_] = temp;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      T temp = U[k * n + i];
      U[k * n + i] = U[k_ * n + i];
      U[k_ * n + i] = temp;
    }
  }

  // reset the sub-matrix of U, A total of (n - k - 1) * (n - k - 1) threads.
  template <typename T>
  __global__ void ReduceKernel(T* U, const int n, int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n - k - 1; i += blockDim.x * gridDim.x) {
      U[(i + 1 + k) * n + k] /= U[k * n + k];
      for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n - k - 1; j += blockDim.y * gridDim.y) {
        U[(i + 1 + k) * n + j + 1 + k] -= U[(i + 1 + k) * n + k] * U[k * n + j + 1 + k]; 
      }
    }
  }

  // generate L and U, A total of n * n threads.
  template <typename T>
  __global__ void FillKernel(T* L, T* U, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n; j += blockDim.y * gridDim.y) {
        if (i > j) {
          L[i * n + j] = U[i * n + j];
          U[i * n + j] = 0;
        }
        else {
          L[i * n + j] = (i == j) ? 1 : 0;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::pluLauncher (
  const Eigen::GpuDevice& device, const T* M, const int n, int* pi, T* L, T* U){
  using namespace plu;
  Cuda2DLaunchConfig initConfig = GetCuda2DLaunchConfig(n, n, device);
  InitKernel<T><<<initConfig.block_count, initConfig.thread_per_block, 0, device.stream()>>>(
    M, n, pi, L, U);

  for (int k = 0; k < n - 1; k++) {
    CudaLaunchConfig argMaxConfig = GetCudaLaunchConfig(n - k, device);
    int sMemSize = (n - k) * sizeof(int);
    ArgMaxKernel<T><<<argMaxConfig.block_count, argMaxConfig.thread_per_block, sMemSize, device.stream()>>>(
      U, n, k);

    CudaLaunchConfig swapConfig = GetCudaLaunchConfig(n, device);
    SwapKernel<T><<<swapConfig.block_count, swapConfig.thread_per_block, 0, device.stream()>>>(
      pi, U, n, k);
    
    Cuda2DLaunchConfig reduceConfig = GetCuda2DLaunchConfig(n - k - 1, n - k - 1, device);
    ReduceKernel<T><<<reduceConfig.block_count, reduceConfig.thread_per_block, 0, device.stream()>>>(
      U, n, k);
  }

  Cuda2DLaunchConfig fillConfig = GetCuda2DLaunchConfig(n, n, device);
    FillKernel<T><<<fillConfig.block_count, fillConfig.thread_per_block, 0, device.stream()>>>(
      L, U, n);
}

/* recv: L@U@recv = pi -> rhs */
template <typename T>
void ComputeMethods<T>::pluSolveLauncher(
  const Eigen::ThreadPoolDevice& device, const int* pi, const T* L, const T* U, const T* rhs, const int n, T* recv){
  for(int i = 0; i < n; i ++) {
    recv[i] = rhs[pi[i]];
  }

  for (int j = 0; j < n - 1; j ++) {
    for (int i = j + 1; i < n; i++) {
      recv[i] -= L[i * n + j] * recv[j];
    }
  }

  for (int j = n - 1; j > -1; j--) {
    recv[j] /= U[j * n + j];
    for (int i = j - 1; i > -1; i--) {
      recv[i] -= U[i * n + j] * recv[j];
    }
  }
}

namespace plu{
  template <typename T>
  __global__ void solveKernel(
    const int* pi, const T* L, const T* U, const T* rhs, const int n, T* recv){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      recv[i] = rhs[pi[i]];
      __syncthreads();

      for (int j = 0; j < n - 1; j ++) {
        if (i > j) {
          recv[i] -= L[i * n + j] * recv[j];
        }
        __syncthreads();
      }

      for (int j = n - 1; j > -1; j--) {
        recv[j] /= U[j * n + j];
        if (i < j) {
          recv[i] -= U[i * n + j] * recv[j];
        }
        __syncthreads();
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::pluSolveLauncher(
  const Eigen::GpuDevice& device, 
  const int* pi, const T* L, const T* U, const T* rhs, const int n, T* recv){
  CudaLaunchConfig config = GetCudaLaunchConfig(n, device);
  plu::solveKernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    pi, L, U, rhs, n, recv);
}

/* generate linear algebra equations for plu gradients. see more detail in README.md */
template <typename T>
void ComputeMethods<T>::pluGradEqsLauncher(
  const Eigen::ThreadPoolDevice& device, const int* pi, const T* L, const T* U, const T* gradL, const T* gradU, const int n, T* Mat, T* rhs) {
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < n; c++) {
      rhs[r * n + c] = (r > c) ? gradL[r * n + c] : gradU[r * n + c];
      if (r > c) {
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            Mat[(r * n + c) * n * n + i * n + j] = pi[r] == i ? U[c * n + j] : 0;
          }
        }
      }
      else {
        for (int i = 0; i < n; i++) {
          T result = 0;
          for (int k = 0; k < n; k++) {
            result += pi[k] == i ? L[k * n + r] : 0;
          }
          Mat[(r * n + c) * n * n + i * n + c] = result;
        }
      }
    }
  }
}

namespace plu{
  template <typename T>
  __global__ void gradEqsKernel(
    const int* pi, const T* L, const T* U, const T* gradL, const T* gradU, const int n, T* Mat, T* rhs) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += blockDim.x * gridDim.x) {
      for (int c = blockIdx.y * blockDim.y + threadIdx.y; c < n; c += blockDim.y * gridDim.y) {
        rhs[r * n + c] = (r > c) ? gradL[r * n + c] : gradU[r * n + c];
        if (r > c) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              Mat[(r * n + c) * n * n + i * n + j] = pi[r] == i ? U[c * n + j] : 0;
            }
          }
        }
        else {
          for (int i = 0; i < n; i++) {
            T result = 0;
            for (int k = 0; k < n; k++) {
              result += pi[k] == i ? L[k * n + r] : 0;
            }
            Mat[(r * n + c) * n * n + i * n + c] = result;
          }
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::pluGradEqsLauncher(
  const Eigen::GpuDevice& device, const int* pi, const T* L, const T* U, const T* gradL, const T* gradU, const int n, T* Mat, T* rhs) {
  Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(n, n, device);
  plu::gradEqsKernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    pi, L, U, gradL, gradU, n, Mat, rhs);
}

/* padTensor = SAME(inputTensor) */
template <typename T>
void ComputeMethods<T>::conv2dPaddingLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
  const int padH, const int padW, const int startH, const int startW, T* padTensor) {
  for (int padi = 0; padi < N * padH * padW * inputC; padi++) {
    int padn = padi / (padH * padW * inputC);
    int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
    int padw = (padi % (padW * inputC)) / inputC;
    int padc = padi % inputC;

    int inputn = padn;
    int inputh = padh - startH;
    int inputw = padw - startW;
    int inputc = padc;

    if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
      int inputi = inputn * (inputH * inputW * inputC) + inputh * (inputW * inputC) + inputw * inputC + inputc;
      padTensor[padi] = inputTensor[inputi];
    }
    else {
      padTensor[padi] = 0;
    }
  }
}

namespace conv2d{
  namespace padding{
    template <typename T>
    __global__ void kernel(
      const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
      const int padH, const int padW, const int startH, const int startW, T* padTensor) {
      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * inputC; padi += blockDim.x * gridDim.x) {
        int padn = padi / (padH * padW * inputC);
        int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
        int padw = (padi % (padW * inputC)) / inputC;
        int padc = padi % inputC;

        int inputn = padn;
        int inputh = padh - startH;
        int inputw = padw - startW;
        int inputc = padc;

        if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
          int inputi = inputn * (inputH * inputW * inputC) + inputh * (inputW * inputC) + inputw * inputC + inputc;
          padTensor[padi] = inputTensor[inputi];
        }
        else {
          padTensor[padi] = 0;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::conv2dPaddingLauncher(
  const Eigen::GpuDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
  const int padH, const int padW, const int startH, const int startW, T* padTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::padding::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, N, inputH, inputW, inputC, padH, padW, startH, startW, padTensor);
}

/* padTensor = VALID(inputTensor) */
template <typename T>
void ComputeMethods<T>::conv2dPaddingLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
  const int padH, const int padW, T* padTensor) {
  for (int padi = 0; padi < N * padH * padW * inputC; padi++) {
    int padn = padi / (padH * padW * inputC);
    int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
    int padw = (padi % (padW * inputC)) / inputC;
    int padc = padi % inputC;

    int inputi = padn * (inputH * inputW * inputC) + padh * (inputW * inputC) + padw * inputC + padc;
    padTensor[padi] = inputTensor[inputi];
  }
}

namespace conv2d{
  namespace padding{
    template <typename T>
    __global__ void kernel(
      const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
      const int padH, const int padW, T* padTensor) {
      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * inputC; padi += blockDim.x * gridDim.x) {
        int padn = padi / (padH * padW * inputC);
        int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
        int padw = (padi % (padW * inputC)) / inputC;
        int padc = padi % inputC;

        int inputi = padn * (inputH * inputW * inputC) + padh * (inputW * inputC) + padw * inputC + padc;
        padTensor[padi] = inputTensor[inputi];
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::conv2dPaddingLauncher(
  const Eigen::GpuDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
  const int padH, const int padW, T* padTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::padding::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, N, inputH, inputW, inputC, padH, padW, padTensor);
}

/* outputTensor = conv2d(padTensor) */
template <typename T>
void ComputeMethods<T>::conv2dLauncher(
  const Eigen::ThreadPoolDevice& device, 
  T* padTensor, const int N, const int padH, const int padW, const int inputC, 
  const T* kernel, const int kernelH, const int kernelW, const int outputC, 
  const int strideH, const int strideW, 
  const int outputH, const int outputW, T* outputTensor) {
  for (int outputi = 0; outputi < N * outputH * outputW * outputC; outputi++) {
    int outputn = outputi / (outputH * outputW * outputC);
    int outputh = (outputi % (outputH * outputW * outputC)) / (outputW * outputC);
    int outputw = (outputi % (outputW * outputC)) / outputC;
    int outputc = outputi % outputC;

    int padn = outputn;
    int padh = outputh * strideH;
    int padw = outputw * strideW;

    T r = 0;
    for (int h = 0; h < kernelH; h++) {
      for (int w = 0; w < kernelW; w++) {
        for (int c = 0; c < inputC; c++) {
          int padi = padn * (padH * padW * inputC) + (padh + h) * (padW * inputC) + (padw + w) * inputC + c;
          int kerneli = h * (kernelW * inputC * outputC) + w * (inputC * outputC) + c * outputC + outputc;
          r += padTensor[padi] * kernel[kerneli];
        }
      }
    }
    outputTensor[outputi] = r;
  }
}

namespace conv2d{
  template <typename T>
  __global__ void kernel(
    T* padTensor, const int N, const int padH, const int padW, const int inputC, 
    const T* kernel, const int kernelH, const int kernelW, const int outputC, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor) {
    for (int outputi = blockIdx.x * blockDim.x + threadIdx.x; outputi < N * outputH * outputW * outputC; outputi += blockDim.x * gridDim.x) {
      int outputn = outputi / (outputH * outputW * outputC);
      int outputh = (outputi % (outputH * outputW * outputC)) / (outputW * outputC);
      int outputw = (outputi % (outputW * outputC)) / outputC;
      int outputc = outputi % outputC;

      int padn = outputn;
      int padh = outputh * strideH;
      int padw = outputw * strideW;

      T r = 0;
      for (int h = 0; h < kernelH; h++) {
        for (int w = 0; w < kernelW; w++) {
          for (int c = 0; c < inputC; c++) {
            int padi = padn * (padH * padW * inputC) + (padh + h) * (padW * inputC) + (padw + w) * inputC + c;
            int kerneli = h * (kernelW * inputC * outputC) + w * (inputC * outputC) + c * outputC + outputc;
            r += padTensor[padi] * kernel[kerneli];
          }
        }
      }
      outputTensor[outputi] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::conv2dLauncher(
  const Eigen::GpuDevice& device, 
  T* padTensor, const int N, const int padH, const int padW, const int inputC, 
  const T* kernel, const int kernelH, const int kernelW, const int outputC, 
  const int strideH, const int strideW, 
  const int outputH, const int outputW, T* outputTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * outputH * outputW * outputC, device);
  conv2d::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    padTensor, N, padH, padW, inputC, kernel, kernelH, kernelW, outputC, strideH, strideW, outputH, outputW, outputTensor);
}

/* gradPadTensor = grad->(gradTensor, kernel) */
template <typename T>
void ComputeMethods<T>::gradConv2dInputLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
  const T* kernel, const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const int padH, const int padW, const int inputC, T* gradPadTensor) {
  for (int padi = 0; padi < N * padH * padW * inputC; padi++) {
    gradPadTensor[padi] = 0;
  }

  for (int outputi = 0; outputi < N * outputH * outputW * outputC; outputi++) {
    int outputn = outputi / (outputH * outputW * outputC);
    int outputh = (outputi % (outputH * outputW * outputC)) / (outputW * outputC);
    int outputw = (outputi % (outputW * outputC)) / outputC;
    int outputc = outputi % outputC;

    int padn = outputn;
    int padh = outputh * strideH;
    int padw = outputw * strideW;

    for (int h = 0; h < kernelH; h++) {
      for (int w = 0; w < kernelW; w++) {
        for (int c = 0; c < inputC; c++) {
          int padi = padn * (padH * padW * inputC) + (padh + h) * (padW * inputC) + (padw + w) * inputC + c;
          int kerneli = h * (kernelW * inputC * outputC) + w * (inputC * outputC) + c * outputC + outputc;
          gradPadTensor[padi] += gradTensor[outputi] * kernel[kerneli];
        }
      }
    }
  }
}

namespace conv2d{
  namespace grad{
    namespace input{
      template <typename T>
      __global__ void kernel(
        const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
        const T* kernel, const int kernelH, const int kernelW, 
        const int strideH, const int strideW, 
        const int padH, const int padW, const int inputC, T* gradPadTensor) {
        for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * inputC; padi += blockDim.x * gridDim.x) {
          int n = padi / (padH * padW * inputC);
          int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
          int padw = (padi % (padW * inputC)) / inputC;
          int inputc = padi % inputC;

          T r = 0;
          for (
            int outputh = padh / strideH < outputH ? padh / strideH : outputH - 1; 
            outputh >= 0 && padh - outputh * strideH < kernelH; 
            outputh -= 1) {
            int kernelh = padh - outputh * strideH;
            for (
              int outputw = padw / strideW < outputW ? padw / strideW : outputW - 1; 
              outputw >= 0 && padw - outputw * strideW < kernelW; 
              outputw -= 1) {
              int kernelw = padw - outputw * strideW;
              for (int outputc = 0; outputc < outputC; outputc++) {
                int outputi = n * (outputH * outputW * outputC) + outputh * (outputW * outputC) + outputw * outputC + outputc;
                int kerneli = kernelh * (kernelW * inputC * outputC) + kernelw * (inputC * outputC) + inputc * outputC + outputc;
                r += gradTensor[outputi] * kernel[kerneli];
              }
            }
          }
          gradPadTensor[padi] = r;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::gradConv2dInputLauncher(
  const Eigen::GpuDevice& device, 
  const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
  const T* kernel, const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const int padH, const int padW, const int inputC, T* gradPadTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::grad::input::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradTensor, N, outputH, outputW, outputC, 
    kernel, kernelH, kernelW, 
    strideH, strideW, 
    padH, padW, inputC, gradPadTensor);
}

/* gradInputTensor = UNSAME(gradPadTensor) */
template <typename T>
void ComputeMethods<T>::conv2dUnpaddingLauncher(
  const Eigen::ThreadPoolDevice& device, 
  T* gradPadTensor, const int padH, const int padW, const int startH, const int startW, 
  const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {

  for (int padi = 0; padi < N * padH * padW * inputC; padi++) {
    int padn = padi / (padH * padW * inputC);
    int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
    int padw = (padi % (padW * inputC)) / inputC;
    int padc = padi % inputC;

    int inputn = padn;
    int inputh = padh - startH;
    int inputw = padw - startW;
    int inputc = padc;

    if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
      int inputi = inputn * (inputH * inputW * inputC) + inputh * (inputW * inputC) + inputw * inputC + inputc;
      gradInputTensor[inputi] = gradPadTensor[padi];
    }
  }
}

namespace conv2d{
  namespace unpadding{
    template <typename T>
    __global__ void kernel(
      T* gradPadTensor, const int padH, const int padW, const int startH, const int startW, 
      const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {

      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * inputC; padi += blockDim.x * gridDim.x) {
        int padn = padi / (padH * padW * inputC);
        int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
        int padw = (padi % (padW * inputC)) / inputC;
        int padc = padi % inputC;

        int inputn = padn;
        int inputh = padh - startH;
        int inputw = padw - startW;
        int inputc = padc;

        if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
          int inputi = inputn * (inputH * inputW * inputC) + inputh * (inputW * inputC) + inputw * inputC + inputc;
          gradInputTensor[inputi] = gradPadTensor[padi];
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::conv2dUnpaddingLauncher(
  const Eigen::GpuDevice& device, 
  T* gradPadTensor, const int padH, const int padW, const int startH, const int startW, 
  const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::unpadding::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradPadTensor, padH, padW, startH, startW, N, inputH, inputW, inputC, gradInputTensor);
}

/* gradInputTensor = UNVALID(gradPadTensor) */
template <typename T>
void ComputeMethods<T>::conv2dUnpaddingLauncher(
  const Eigen::ThreadPoolDevice& device, 
  T* gradPadTensor, const int padH, const int padW, 
  const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {

  for (int inputi = 0; inputi < N * inputH * inputW * inputC; inputi++) {
    gradInputTensor[inputi] = 0;
  }

  for (int padi = 0; padi < N * padH * padW * inputC; padi++) {
    int padn = padi / (padH * padW * inputC);
    int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
    int padw = (padi % (padW * inputC)) / inputC;
    int padc = padi % inputC;

    int inputi = padn * (inputH * inputW * inputC) + padh * (inputW * inputC) + padw * inputC + padc;
    gradInputTensor[inputi] = gradPadTensor[padi];
  }
}

namespace conv2d{
  namespace unpadding{
    template <typename T>
    __global__ void initKernel(
      const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {
      for (int inputi = blockIdx.x * blockDim.x + threadIdx.x; inputi < N * inputH * inputW * inputC; inputi += blockDim.x * gridDim.x) {
        gradInputTensor[inputi] = 0;
      }
    }

    template <typename T>
    __global__ void kernel(
      T* gradPadTensor, const int padH, const int padW, 
      const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {
      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * inputC; padi += blockDim.x * gridDim.x) {

        int padn = padi / (padH * padW * inputC);
        int padh = (padi % (padH * padW * inputC)) / (padW * inputC);
        int padw = (padi % (padW * inputC)) / inputC;
        int padc = padi % inputC;

        int inputi = padn * (inputH * inputW * inputC) + padh * (inputW * inputC) + padw * inputC + padc;
        gradInputTensor[inputi] = gradPadTensor[padi];
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::conv2dUnpaddingLauncher(
  const Eigen::GpuDevice& device, 
  T* gradPadTensor, const int padH, const int padW, 
  const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor) {
  
  CudaLaunchConfig initConfig = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::unpadding::initKernel<T><<<initConfig.block_count, initConfig.thread_per_block, 0, device.stream()>>>(
    N, inputH, inputW, inputC, gradInputTensor);

  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * inputC, device);
  conv2d::unpadding::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradPadTensor, padH, padW, N, inputH, inputW, inputC, gradInputTensor);
}

/* gradKernelTensor = grad->(gradTensor, padTensor) */
template <typename T>
void ComputeMethods<T>::gradConv2dKernelLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
  T* padTensor, const int padH, const int padW, const int inputC, 
  const int strideH, const int strideW, 
  const int kernelH, const int kernelW, T* gradKernelTensor) {

  for (int kerneli = 0; kerneli < kernelH * kernelW * inputC * outputC; kerneli++) {
    gradKernelTensor[kerneli] = 0;
  }

  for (int kerneli = 0; kerneli < kernelH * kernelW * inputC * outputC; kerneli++) {
    int kernelh = kerneli / (kernelW * inputC * outputC);
    int kernelw = (kerneli % (kernelW * inputC * outputC)) / (inputC * outputC);
    int inputc = (kerneli % (inputC * outputC)) / outputC;
    int outputc = kerneli % outputC;

    for (int outputh = 0; outputh < outputH; outputh++) {
      for (int outputw = 0; outputw < outputW; outputw++) {
        int padh = outputh * strideH + kernelh;
        int padw = outputw * strideW + kernelw;
        for (int n = 0; n < N; n++) {
          int padi = n * (padH * padW * inputC) + padh * (padW * inputC) + padw * inputC + inputc;
          int outputi = n * (outputH * outputW * outputC) + outputh * (outputW * outputC) + outputw * outputC + outputc;
          gradKernelTensor[kerneli] += padTensor[padi] * gradTensor[outputi];
        }
      }
    }
  }
}

namespace conv2d{
  namespace grad{
    namespace kernel{
      template <typename T>
        __global__ void kernel(
        const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
        T* padTensor, const int padH, const int padW, const int inputC, 
        const int strideH, const int strideW, 
        const int kernelH, const int kernelW, T* gradKernelTensor) {
        
        for (int kerneli = blockIdx.x * blockDim.x + threadIdx.x; kerneli < kernelH * kernelW * inputC * outputC; kerneli += blockDim.x * gridDim.x) {
          int kernelh = kerneli / (kernelW * inputC * outputC);
          int kernelw = (kerneli % (kernelW * inputC * outputC)) / (inputC * outputC);
          int inputc = (kerneli % (inputC * outputC)) / outputC;
          int outputc = kerneli % outputC;

          T r = 0;
          for (int outputh = 0; outputh < outputH; outputh++) {
            for (int outputw = 0; outputw < outputW; outputw++) {
              int padh = outputh * strideH + kernelh;
              int padw = outputw * strideW + kernelw;
              for (int n = 0; n < N; n++) {
                int padi = n * (padH * padW * inputC) + padh * (padW * inputC) + padw * inputC + inputc;
                int outputi = n * (outputH * outputW * outputC) + outputh * (outputW * outputC) + outputw * outputC + outputc;
                r += padTensor[padi] * gradTensor[outputi];
              }
            }
          }
          gradKernelTensor[kerneli] = r;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::gradConv2dKernelLauncher(
  const Eigen::GpuDevice& device, 
  const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
  T* padTensor, const int padH, const int padW, const int inputC, 
  const int strideH, const int strideW, 
  const int kernelH, const int kernelW, T* gradKernelTensor) {

  CudaLaunchConfig config = GetCudaLaunchConfig(kernelH * kernelW * inputC * outputC, device);
  conv2d::grad::kernel::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradTensor, N, outputH, outputW, outputC, 
    padTensor, padH, padW, inputC, 
    strideH, strideW, 
    kernelH, kernelW, gradKernelTensor);
}

/* outputTensor = inputTensor + bias */
template <typename T>
void ComputeMethods<T>::biasAddLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* inputTensor, const int N, const int C, 
  const T* bias, T* outputTensor) {

  for (int i = 0; i < N * C; i++) {
    outputTensor[i] = inputTensor[i] + bias[i % C];
  }
}

namespace biasAdd{
  template <typename T>
  __global__ void kernel(
    const T* inputTensor, const int N, const int C, 
    const T* bias, T* outputTensor) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C; i += blockDim.x * gridDim.x) {
      outputTensor[i] = inputTensor[i] + bias[i % C];
    }
  }
}

template <typename T>
void ComputeMethods<T>::biasAddLauncher(
  const Eigen::GpuDevice& device, 
  const T* inputTensor, const int N, const int C, 
  const T* bias, T* outputTensor) {

  CudaLaunchConfig config = GetCudaLaunchConfig(N * C, device);
  biasAdd::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, N, C, bias, outputTensor);
}

/* padTensor = SAME(inputTensor) */
template <typename T>
void ComputeMethods<T>::maxPoolPaddingLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int C,
  const int padH, const int padW, const int startH, const int startW, T* padTensor) {
  for (int padi = 0; padi < N * padH * padW * C; padi++) {
    int n = padi / (padH * padW * C);
    int padh = (padi % (padH * padW * C)) / (padW * C);
    int padw = (padi % (padW * C)) / C;
    int c = padi % C;

    int inputh = padh - startH;
    int inputw = padw - startW;

    if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
      int inputi = n * (inputH * inputW * C) + inputh * (inputW * C) + inputw * C + c;
      padTensor[padi] = inputTensor[inputi];
    }
    else {
      padTensor[padi] = -10000;
    }
  }
}

namespace maxPool{
  namespace padding{
    template <typename T>
    __global__ void kernel(
      const T* inputTensor, const int N, const int inputH, const int inputW, const int C,
      const int padH, const int padW, const int startH, const int startW, T* padTensor) {
      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * C; padi += blockDim.x * gridDim.x) {
        int n = padi / (padH * padW * C);
        int padh = (padi % (padH * padW * C)) / (padW * C);
        int padw = (padi % (padW * C)) / C;
        int c = padi % C;

        int inputh = padh - startH;
        int inputw = padw - startW;

        if (0 <= inputh && inputh < inputH && 0 <= inputw && inputw < inputW) {
          int inputi = n * (inputH * inputW * C) + inputh * (inputW * C) + inputw * C + c;
          padTensor[padi] = inputTensor[inputi];
        }
        else {
          padTensor[padi] = -10000;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::maxPoolPaddingLauncher(
  const Eigen::GpuDevice& device, 
  const T* inputTensor, const int N, const int inputH, const int inputW, const int C,
  const int padH, const int padW, const int startH, const int startW, T* padTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * padH * padW * C, device);
  maxPool::padding::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, N, inputH, inputW, C, padH, padW, startH, startW, padTensor);
}

/* outputTensor = maxpool(padTensor) */
template <typename T>
void ComputeMethods<T>::maxPoolLauncher(
  const Eigen::ThreadPoolDevice& device, 
  T* padTensor, const int N, const int padH, const int padW, const int C, 
  const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const int outputH, const int outputW, T* outputTensor) {
  for (int outputi = 0; outputi < N * outputH * outputW * C; outputi++) {
    int n = outputi / (outputH * outputW * C);
    int outputh = (outputi % (outputH * outputW * C)) / (outputW * C);
    int outputw = (outputi % (outputW * C)) / C;
    int c = outputi % C;

    int padh = outputh * strideH;
    int padw = outputw * strideW;

    T r = padTensor[n * (padH * padW * C) + padh * (padW * C) + padw * C + c];
    for (int h = 0; h < kernelH; h++) {
      for (int w = 0; w < kernelW; w++) {
        int padi = n * (padH * padW * C) + (padh + h) * (padW * C) + (padw + w) * C + c;
        r = (padTensor[padi] > r) ? padTensor[padi] : r;
      }
    }
    outputTensor[outputi] = r;
  }
}

namespace maxPool{
  template <typename T>
  __global__ void kernel(
    T* padTensor, const int N, const int padH, const int padW, const int C, 
    const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor) {

    for (int outputi = blockIdx.x * blockDim.x + threadIdx.x; outputi < N * outputH * outputW * C; outputi += blockDim.x * gridDim.x) {
      int n = outputi / (outputH * outputW * C);
      int outputh = (outputi % (outputH * outputW * C)) / (outputW * C);
      int outputw = (outputi % (outputW * C)) / C;
      int c = outputi % C;

      int padh = outputh * strideH;
      int padw = outputw * strideW;

      T r = padTensor[n * (padH * padW * C) + padh * (padW * C) + padw * C + c];
      for (int h = 0; h < kernelH; h++) {
        for (int w = 0; w < kernelW; w++) {
          int padi = n * (padH * padW * C) + (padh + h) * (padW * C) + (padw + w) * C + c;
          r = (padTensor[padi] > r) ? padTensor[padi] : r;
        }
      }
      outputTensor[outputi] = r;
    }
  }
}

template <typename T>
void ComputeMethods<T>::maxPoolLauncher(
  const Eigen::GpuDevice& device, 
  T* padTensor, const int N, const int padH, const int padW, const int C, 
  const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const int outputH, const int outputW, T* outputTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * outputH * outputW * C, device);
  maxPool::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    padTensor, N, padH, padW, C, kernelH, kernelW, strideH, strideW, outputH, outputW, outputTensor);
}

/* gradPadTensor = grad->(gradOutputTensor) */
template <typename T>
void ComputeMethods<T>::gradMaxPoolLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* gradOutputTensor, const int N, const int outputH, const int outputW, const int C, 
  const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const T* outputTensor,
  const int padH, const int padW, T* padTensor, 
  T* gradPadTensor) {

  for (int padi = 0; padi < N * padH * padW * C; padi++) {
    gradPadTensor[padi] = 0;
  }

  for (int outputi = 0; outputi < N * outputH * outputW * C; outputi++) {
    int n = outputi / (outputH * outputW * C);
    int outputh = (outputi % (outputH * outputW * C)) / (outputW * C);
    int outputw = (outputi % (outputW * C)) / C;
    int c = outputi % C;

    int padh = outputh * strideH;
    int padw = outputw * strideW;

    for (int h = 0; h < kernelH; h++) {
      for (int w = 0; w < kernelW; w++) {
        int padi = n * (padH * padW * C) + (padh + h) * (padW * C) + (padw + w) * C + c;
        gradPadTensor[padi] += (padTensor[padi] == outputTensor[outputi]) ? gradOutputTensor[outputi] : 0;
      }
    }
  }
}

namespace maxpool{
  namespace grad{
    template <typename T>
    __global__ void initKernel(
      const int N, const int padH, const int padW, const int C, T* gradPadTensor) {
      for (int padi = blockIdx.x * blockDim.x + threadIdx.x; padi < N * padH * padW * C; padi += blockDim.x * gridDim.x) {
        gradPadTensor[padi] = 0;
      }
    }

    template <typename T>
      __global__ void kernel(
      const T* gradOutputTensor, const int N, const int outputH, const int outputW, const int C, 
      const int kernelH, const int kernelW, 
      const int strideH, const int strideW, 
      const T* outputTensor,
      const int padH, const int padW, T* padTensor, 
      T* gradPadTensor) {
      for (int outputi = blockIdx.x * blockDim.x + threadIdx.x; outputi < N * outputH * outputW * C; outputi += blockDim.x * gridDim.x) {
        int n = outputi / (outputH * outputW * C);
        int outputh = (outputi % (outputH * outputW * C)) / (outputW * C);
        int outputw = (outputi % (outputW * C)) / C;
        int c = outputi % C;

        int padh = outputh * strideH;
        int padw = outputw * strideW;

        for (int h = 0; h < kernelH; h++) {
          for (int w = 0; w < kernelW; w++) {
            int padi = n * (padH * padW * C) + (padh + h) * (padW * C) + (padw + w) * C + c;
            gradPadTensor[padi] += (padTensor[padi] == outputTensor[outputi]) ? gradOutputTensor[outputi] : 0;
            __syncthreads();
          }
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::gradMaxPoolLauncher(
  const Eigen::GpuDevice& device, 
  const T* gradOutputTensor, const int N, const int outputH, const int outputW, const int C, 
  const int kernelH, const int kernelW, 
  const int strideH, const int strideW, 
  const T* outputTensor,
  const int padH, const int padW, T* padTensor, 
  T* gradPadTensor) {
  
  CudaLaunchConfig initConfig = GetCudaLaunchConfig(N * padH * padW * C, device);
  maxpool::grad::initKernel<T><<<initConfig.block_count, initConfig.thread_per_block, 0, device.stream()>>>(
    N, padH, padW, C, gradPadTensor);

  CudaLaunchConfig config = GetCudaLaunchConfig(N * outputH * outputW * C, device);
  maxpool::grad::kernel<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradOutputTensor, N, outputH, outputW, C, 
    kernelH, kernelW, 
    strideH, strideW, 
    outputTensor,
    padH, padW, padTensor, 
    gradPadTensor);
}

/* outputTensor = \\(\frac{scale(inputTensor - mean)}{sqrt(variance + varianceEpsilon)} + offset\\) */
template <typename T>
void ComputeMethods<T>::batchNormLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* inputTensor, const int N, const int C, 
  const T* meanTensor, 
  const T* varianceTensor, 
  const T* offsetTensor, 
  const T* scaleTensor, 
  const T varianceEpsilon, 
  T* outputTensor) {
  for (int i = 0; i < N * C; i++) {
    int c = i % C;
    outputTensor[i] = scaleTensor[c] * (inputTensor[i] - meanTensor[c]) / sqrt(varianceTensor[c] + varianceEpsilon) + offsetTensor[c];
  }
}

namespace  batchNorm{
  __global__ void kernel(const float* inputTensor, const int N, const int C, 
    const float* meanTensor, 
    const float* varianceTensor, 
    const float* offsetTensor, 
    const float* scaleTensor, 
    const float varianceEpsilon, 
    float* outputTensor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C; i += blockDim.x * gridDim.x) {
      int c = i % C;
      outputTensor[i] = scaleTensor[c] * (inputTensor[i] - meanTensor[c]) * rsqrtf(varianceTensor[c] + varianceEpsilon) + offsetTensor[c];
    }
  }
  
  __global__ void kernel(const double* inputTensor, const int N, const int C, 
    const double* meanTensor, 
    const double* varianceTensor, 
    const double* offsetTensor, 
    const double* scaleTensor, 
    const double varianceEpsilon, 
    double* outputTensor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C; i += blockDim.x * gridDim.x) {
      int c = i % C;
      outputTensor[i] = scaleTensor[c] * (inputTensor[i] - meanTensor[c]) * rsqrt(varianceTensor[c] + varianceEpsilon) + offsetTensor[c];
    }
  }
}

template <typename T>
void ComputeMethods<T>::batchNormLauncher(
  const Eigen::GpuDevice& device, 
  const T* inputTensor, const int N, const int C, 
  const T* meanTensor, 
  const T* varianceTensor, 
  const T* offsetTensor, 
  const T* scaleTensor, 
  const T varianceEpsilon, 
  T* outputTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * C, device);
  batchNorm::kernel<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    inputTensor, N, C, meanTensor, varianceTensor, offsetTensor, scaleTensor, varianceEpsilon, outputTensor);
}

/* grad_input_tensor, grad_mean, grad_variance, grad_offset, grad_scale = grad->BN(grad, input_tensor, mean, variance, scale) */
template <typename T>
void ComputeMethods<T>::gradBatchNormLauncher(
  const Eigen::ThreadPoolDevice& device, 
  const T* gradTensor, const int N, const int C, const T varianceEpsilon, 
  const T* inputTensor, 
  const T* meanTensor, 
  const T* varianceTensor, 
  const T* scaleTensor,
  T* gradInputTensor,
  T* gradMeanTensor, 
  T* gradVarianceTensor, 
  T* gradOffsetTensor, 
  T* gradScaleTensor) {
  for (int i = 0; i < N * C; i++) {
    int c = i % C;
    gradInputTensor[i] = gradTensor[i] * scaleTensor[c] * rsqrtf(varianceTensor[c] + varianceEpsilon);
  }
  for (int c = 0; c < C; c++) {
    float sumGrad = 0;
    float sumGradInput = 0;
    float sumInput = 0;
    for (int k = 0; k < N; k++) {
      sumGrad += gradTensor[k * C + c];
      sumGradInput += gradTensor[k * C + c] * (inputTensor[k * C + c] - meanTensor[c]);
      sumInput += inputTensor[k * C + c];
    }
    float rsqrtVal = rsqrtf(varianceTensor[c] + varianceEpsilon);
    float rsqrt3Val = rsqrtVal * rsqrtVal * rsqrtVal;

    gradMeanTensor[c] = -sumGrad * scaleTensor[c] * rsqrtVal;
    gradVarianceTensor[c] = -0.5 * sumGradInput * scaleTensor[c] * rsqrt3Val;
    gradOffsetTensor[c] = sumGrad;
    gradScaleTensor[c] = sumGradInput * rsqrtVal;
  }
}

namespace  batchNorm{
  namespace grad {
    __global__ void kernel(
      const float* gradTensor, const int N, const int C, const float varianceEpsilon, 
      const float* inputTensor, 
      const float* meanTensor, 
      const float* varianceTensor, 
      const float* scaleTensor,
      float* gradInputTensor,
      float* gradMeanTensor, 
      float* gradVarianceTensor, 
      float* gradOffsetTensor, 
      float* gradScaleTensor) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C + C; i += blockDim.x * gridDim.x) {
        if (i < N * C) {
          int c = i % C;
          gradInputTensor[i] = gradTensor[i] * scaleTensor[c] * rsqrtf(varianceTensor[c] + varianceEpsilon);
        }
        else {
          int c = i - N * C;
          float sumGrad = 0;
          float sumGradInput = 0;
          float sumInput = 0;
          for (int k = 0; k < N; k++) {
            sumGrad += gradTensor[k * C + c];
            sumGradInput += gradTensor[k * C + c] * (inputTensor[k * C + c] - meanTensor[c]);
            sumInput += inputTensor[k * C + c];
          }
          float rsqrtVal = rsqrtf(varianceTensor[c] + varianceEpsilon);
          float rsqrt3Val = rsqrtVal * rsqrtVal * rsqrtVal;

          gradMeanTensor[c] = -sumGrad * scaleTensor[c] * rsqrtVal;
          gradVarianceTensor[c] = -0.5 * sumGradInput * scaleTensor[c] * rsqrt3Val;
          gradOffsetTensor[c] = sumGrad;
          gradScaleTensor[c] = sumGradInput * rsqrtVal;
        }
      }
    }
    
    __global__ void kernel(
      const double* gradTensor, const int N, const int C, const double varianceEpsilon, 
      const double* inputTensor, 
      const double* meanTensor, 
      const double* varianceTensor, 
      const double* scaleTensor,
      double* gradInputTensor,
      double* gradMeanTensor, 
      double* gradVarianceTensor, 
      double* gradOffsetTensor, 
      double* gradScaleTensor) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C + C; i += blockDim.x * gridDim.x) {
        if (i < N * C) {
          int c = i % C;
          gradInputTensor[i] = gradTensor[i] * scaleTensor[c] * rsqrt(varianceTensor[c] + varianceEpsilon);
        }
        else {
          int c = i - N * C;
          double sumGrad = 0;
          double sumGradInput = 0;
          double sumInput = 0;
          for (int k = 0; k < N; k++) {
            sumGrad += gradTensor[k * C + c];
            sumGradInput += gradTensor[k * C + c] * (inputTensor[k * C + c] - meanTensor[c]);
            sumInput += inputTensor[k * C + c];
          }
          double rsqrtVal = rsqrt(varianceTensor[c] + varianceEpsilon);
          double rsqrt3Val = rsqrtVal * rsqrtVal * rsqrtVal;

          gradMeanTensor[c] = -sumGrad * scaleTensor[c] * rsqrtVal;
          gradVarianceTensor[c] = -0.5 * sumGradInput * scaleTensor[c] * rsqrt3Val;
          gradOffsetTensor[c] = sumGrad;
          gradScaleTensor[c] = sumGradInput * rsqrtVal;
        }
      }
    }
  }
}

template <typename T>
void ComputeMethods<T>::gradBatchNormLauncher(
  const Eigen::GpuDevice& device, 
  const T* gradTensor, const int N, const int C, const T varianceEpsilon, 
  const T* inputTensor, 
  const T* meanTensor, 
  const T* varianceTensor, 
  const T* scaleTensor,
  T* gradInputTensor,
  T* gradMeanTensor, 
  T* gradVarianceTensor, 
  T* gradOffsetTensor, 
  T* gradScaleTensor) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N * C + C, device);
  batchNorm::grad::kernel<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
    gradTensor, N, C, varianceEpsilon, 
    inputTensor, meanTensor, varianceTensor, scaleTensor,
    gradInputTensor, gradMeanTensor, gradVarianceTensor, gradOffsetTensor, gradScaleTensor);
}

// -------------------------------- instantiate all methods ----------------------------------
template class ComputeMethods<float>;
template class ComputeMethods<double>;

// #endif  // GOOGLE_CUDA
