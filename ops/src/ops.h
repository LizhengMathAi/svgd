#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


template <typename T>
class ComputeMethods {
public:
  // inputTensor = outputTensor
  static void identityLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int n, T* outputTensor);
  static void identityLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int n, T* outputTensor);
  // compute k = argmax(vec)
  static void reduceArgmaxLauncher(
    const Eigen::ThreadPoolDevice& device, const int sSize, 
    const T* vec, const int n, int* k);
  static void reduceArgmaxLauncher(
    const Eigen::GpuDevice& device, const int sSize, 
    const T* vec, const int n, int* k);

  // compute recv = sum(vec)
  static void reduceSumLauncher(
    const Eigen::ThreadPoolDevice& device, const int sSize, 
    const T* vec, const int n, T* recv);
  static void reduceSumLauncher(
    const Eigen::GpuDevice& device, const int sSize, 
    const T* vec, const int n, T* recv);

  // compute recv = sum(vec1 * vec2)
  static void reduceInnerProductLauncher(
    const Eigen::ThreadPoolDevice& device, const int sSize, 
    const T* vec1, const T* vec2, const int n, T* recv);
  static void reduceInnerProductLauncher(
    const Eigen::GpuDevice& device, const int sSize, 
    const T* vec1, const T* vec2, const int n, T* recv);

  // compute recv = sum(vec1 * vec3) / sum(vec2 * vec3)
  static void reduceDoubleInnerProductLauncher(
    const Eigen::ThreadPoolDevice& device, const int sSize, 
    const T* vec1, const T* vec2, const T* vec3, const int n, T* recv);
  static void reduceDoubleInnerProductLauncher(
    const Eigen::GpuDevice& device, const int sSize, 
    const T* vec1, const T* vec2, const T* vec3, const int n, T* recv);

  // compute rhs = vecs[0] + vecs[1] + ...
  static void addNLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* vecs[], const int m, const int n, T* rhs);
  static void addNLauncher(
    const Eigen::GpuDevice& device, 
    const T* vecs[], const int m, const int n, T* rhs);

  // compute recv = mat.T
  static void transposeLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* mat, const int m, const int n, T* recv);
  static void transposeLauncher(
    const Eigen::GpuDevice& device, 
    const T* mat, const int m, const int n, T* recv);
  // compute recv = mat1@mat2.T
  static void matMulLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* mat1, const T* mat2, const int m, const int n, const int l, T* recv);
  static void matMulLauncher(
    const Eigen::GpuDevice& device, 
    const T* mat1, const T* mat2, const int m, const int n, const int l, T* recv);

  // compute recv = mat1 X mat2
  static void kroneckerProductLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* mat1, const T* mat2, const int m, const int n, const int p, const int q, T* recv);
  static void kroneckerProductLauncher(
    const Eigen::GpuDevice& device, 
    const T* mat1, const T* mat2, const int m, const int n, const int p, const int q, T* recv);


  // compute (Permutation matrix) P -> (indices) pi
  static void oneHotLauncher(
    const Eigen::ThreadPoolDevice& device, const int* P, const int n, int* pi);
  static void oneHotLauncher(
    const Eigen::GpuDevice& device, const int* P, const int n, int* pi);
  // compute L@U = M <- pi
  static void pluLauncher(
    const Eigen::ThreadPoolDevice& device, const T* M, const int n, int* pi, T* L, T* U);
  static void pluLauncher(
    const Eigen::GpuDevice& device, const T* M, const int n, int* pi, T* L, T* U);
  // compute recv: L@U@recv = pi -> rhs 
  static void pluSolveLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const int* pi, const T* L, const T* U, const T* rhs, const int n, T* recv);
  static void pluSolveLauncher(
    const Eigen::GpuDevice& device, 
    const int* pi, const T* L, const T* U, const T* rhs, const int n, T* recv);
  // generate linear algebra equations for plu gradients. see more detail in README.md
  static void pluGradEqsLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const int* pi, const T* L, const T* U, const T* gradL, const T* gradU, const int n, T* Mat, T* rhs);
  static void pluGradEqsLauncher(
    const Eigen::GpuDevice& device, 
    const int* pi, const T* L, const T* U, const T* gradL, const T* gradU, const int n, T* Mat, T* rhs);

  // padding images by "SAME" type.
  static void conv2dPaddingLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
    const int padH, const int padW, const int startH, const int startW, T* padTensor);
  static void conv2dPaddingLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
    const int padH, const int padW, const int startH, const int startW, T* padTensor);
  // padding images by "VALID" type.
  static void conv2dPaddingLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
    const int padH, const int padW, T* padTensor);
  static void conv2dPaddingLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int inputC,
    const int padH, const int padW, T* padTensor);
  // compute outputTensor = conv2d(padTensor, kernel, strides).
  static void conv2dLauncher(
    const Eigen::ThreadPoolDevice& device, 
    T* padTensor, const int N, const int padH, const int padW, const int inputC, 
    const T* kernel, const int kernelH, const int kernelW, const int outputC, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor);
  static void conv2dLauncher(
    const Eigen::GpuDevice& device, 
    T* padTensor, const int N, const int padH, const int padW, const int inputC, 
    const T* kernel, const int kernelH, const int kernelW, const int outputC, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor);
  // compute gradPadTensor = grad->(gradTensor, kernel).
  static void gradConv2dInputLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
    const T* kernel, const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const int padH, const int padW, const int inputC, T* gradPadTensor);
  static void gradConv2dInputLauncher(
    const Eigen::GpuDevice& device, 
    const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
    const T* kernel, const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const int padH, const int padW, const int inputC, T* gradPadTensor);
  // unpadding images by "SAME" type.
  static void conv2dUnpaddingLauncher(
    const Eigen::ThreadPoolDevice& device, 
    T* gradPadTensor, const int padH, const int padW, const int startH, const int startW, 
    const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor);
  static void conv2dUnpaddingLauncher(
    const Eigen::GpuDevice& device, 
    T* gradPadTensor, const int padH, const int padW, const int startH, const int startW, 
    const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor);
  // unpadding images by "VALID" type.
  static void conv2dUnpaddingLauncher(
    const Eigen::ThreadPoolDevice& device, 
    T* gradPadTensor, const int padH, const int padW, 
    const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor);
  static void conv2dUnpaddingLauncher(
    const Eigen::GpuDevice& device, 
    T* gradPadTensor, const int padH, const int padW, 
    const int N, const int inputH, const int inputW, const int inputC, T* gradInputTensor);
  // compute gradKernelTensor = grad->(gradTensor, padTensor).
  static void gradConv2dKernelLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
    T* padTensor, const int padH, const int padW, const int inputC, 
    const int strideH, const int strideW, 
    const int kernelH, const int kernelW, T* gradKernelTensor);
  static void gradConv2dKernelLauncher(
    const Eigen::GpuDevice& device, 
    const T* gradTensor, const int N, const int outputH, const int outputW, const int outputC, 
    T* padTensor, const int padH, const int padW, const int inputC, 
    const int strideH, const int strideW, 
    const int kernelH, const int kernelW, T* gradKernelTensor);

  // compute outputTensor = inputTensor + bias.
  static void biasAddLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int N, const int C, 
    const T* bias, T* outputTensor);
  static void biasAddLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int N, const int C, 
    const T* bias, T* outputTensor);

  // padding images by "SAME" type.
  static void maxPoolPaddingLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int C,
    const int padH, const int padW, const int startH, const int startW, T* padTensor);
  static void maxPoolPaddingLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int N, const int inputH, const int inputW, const int C,
    const int padH, const int padW, const int startH, const int startW, T* padTensor);
  // compute outputTensor = maxPool(padTensor, ksize, strides).
  static void maxPoolLauncher(
    const Eigen::ThreadPoolDevice& device, 
    T* padTensor, const int N, const int padH, const int padW, const int C, 
    const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor);
  static void maxPoolLauncher(
    const Eigen::GpuDevice& device, 
    T* padTensor, const int N, const int padH, const int padW, const int C, 
    const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const int outputH, const int outputW, T* outputTensor);
  // compute gradPadTensor = grad->(gradOutputTensor).
  static void gradMaxPoolLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* gradOutputTensor, const int N, const int outputH, const int outputW, const int C, 
    const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const T* outputTensor,
    const int padH, const int padW, T* padTensor, 
    T* gradPadTensor);
  static void gradMaxPoolLauncher(
    const Eigen::GpuDevice& device, 
    const T* gradOutputTensor, const int N, const int outputH, const int outputW, const int C, 
    const int kernelH, const int kernelW, 
    const int strideH, const int strideW, 
    const T* outputTensor,
    const int padH, const int padW, T* padTensor, 
    T* gradPadTensor);
  // compute outputTensor = \\(\frac{scale(inputTensor - mean)}{sqrt(variance + varianceEpsilon)} + offset\\).
  static void batchNormLauncher(
    const Eigen::ThreadPoolDevice& device, 
    const T* inputTensor, const int N, const int C, 
    const T* meanTensor, 
    const T* varianceTensor, 
    const T* offsetTensor, 
    const T* scaleTensor,
    const T varianceEpsilon, 
    T* outputTensor);
  static void batchNormLauncher(
    const Eigen::GpuDevice& device, 
    const T* inputTensor, const int N, const int C, 
    const T* meanTensor, 
    const T* varianceTensor, 
    const T* offsetTensor, 
    const T* scaleTensor,
    const T varianceEpsilon, 
    T* outputTensor);
  // compute gradTensor = grad->BN(outputTensor).
  static void gradBatchNormLauncher(
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
    T* gradScaleTensor);
  static void gradBatchNormLauncher(
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
    T* gradScaleTensor);
};
