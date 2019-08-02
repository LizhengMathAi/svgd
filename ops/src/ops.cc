#include "ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using shape_inference::Dimension;
using shape_inference::DimensionOrConstant;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


// -------------------------------- identity ------------------------------------
REGISTER_OP("DefIdentity")
  .Input("input_tensor: T")
  .Output("output_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("dim: int = 4")
  .SetShapeFn([](InferenceContext* context) {
    int dim;
    TF_RETURN_IF_ERROR(context->GetAttr("dim", &dim));

    ShapeHandle inputShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), dim, &inputShape));

    context->set_output(0, inputShape);

    return Status::OK();
  })
  .Doc(R"doc(
    output_tensor = input_tensor
    )doc");

template <typename Device, typename T>
class DefIdentityOp : public OpKernel {
  private:
    int dim_;

  public:
    explicit DefIdentityOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& inputTensor = context->input(0);

      int n;
      Tensor* outputTensor = nullptr;
      if (dim_ == 1) {
        int N = inputTensor.shape().dim_size(0);
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &outputTensor));
        n = N;
      }
      else if (dim_ == 2) {
        int N = inputTensor.shape().dim_size(0);
        int C = inputTensor.shape().dim_size(1);
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, C}), &outputTensor));
        n = N * C;
      }
      else {
        int N = inputTensor.shape().dim_size(0);
        int H = inputTensor.shape().dim_size(1);
        int W = inputTensor.shape().dim_size(2);
        int C = inputTensor.shape().dim_size(3);
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, H, W, C}), &outputTensor));
        n = N * H * W * C;
      }

      ComputeMethods<T>::identityLauncher(
        context->eigen_device<Device>(), 
        inputTensor.flat<T>().data(), n, outputTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefIdentity").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefIdentityOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefIdentity").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefIdentityOp<Eigen::ThreadPoolDevice, double>);
  
REGISTER_KERNEL_BUILDER(
  Name("DefIdentity").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefIdentityOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefIdentity").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefIdentityOp<Eigen::GpuDevice, double>);

// ------------------------------------ reduce argmax --------------------------------------
REGISTER_OP("DefReduceArgmax")
  .Input("vec: T")
  .Output("k: int32")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .Attr("share_memory_size: int = 128")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle vecShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &vecShape));

    context->set_output(0, context->Scalar());

    return Status::OK();
  })
  .Doc(R"doc(
    k = argmax(vec)
    )doc");

template <typename Device, typename T>
class DefReduceArgmaxOp : public OpKernel {
  private:
    int sSize_;

  public:
    explicit DefReduceArgmaxOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("share_memory_size", &sSize_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& vec = context->input(0);

      const int n = vec.shape().dim_size(0);

      // Create output tensors
      Tensor* k = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &k));

      ComputeMethods<T>::reduceArgmaxLauncher(
        context->eigen_device<Device>(), sSize_, 
        vec.flat<T>().data(), n, k->template flat<int>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefReduceArgmax").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefReduceArgmaxOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceArgmax").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefReduceArgmaxOp<Eigen::ThreadPoolDevice, double>);
  
REGISTER_KERNEL_BUILDER(
  Name("DefReduceArgmax").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefReduceArgmaxOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceArgmax").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefReduceArgmaxOp<Eigen::GpuDevice, double>);

// ------------------------------------ reduce sum --------------------------------------
REGISTER_OP("DefReduceSum")
  .Input("vec: T")
  .Output("recv: T")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .Attr("share_memory_size: int = 128")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle vecShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &vecShape));

    context->set_output(0, context->Scalar());

    return Status::OK();
  })
  .Doc(R"doc(
    recv = sum(vec)
    )doc");

template <typename Device, typename T>
class DefReduceSumOp : public OpKernel {
  private:
    int sSize_;

  public:
    explicit DefReduceSumOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("share_memory_size", &sSize_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& vec = context->input(0);

      const int n = vec.shape().dim_size(0);

      // Create output tensors
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &recv));

      ComputeMethods<T>::reduceSumLauncher(
        context->eigen_device<Device>(), sSize_, 
        vec.flat<T>().data(), n, recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefReduceSum").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefReduceSumOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceSum").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefReduceSumOp<Eigen::ThreadPoolDevice, double>);

REGISTER_KERNEL_BUILDER(
  Name("DefReduceSum").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefReduceSumOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceSum").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefReduceSumOp<Eigen::GpuDevice, double>);

// -------------------------------- reduce inner product ------------------------------------
REGISTER_OP("DefReduceInnerProduct")
  .Input("vec1: T")
  .Input("vec2: T")
  .Output("recv: T")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .Attr("share_memory_size: int = 128")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle vec1Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &vec1Shape));
    ShapeHandle vec2Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &vec2Shape));

    context->set_output(0, context->Scalar());

    return Status::OK();
  })
  .Doc(R"doc(
    recv = sum(vec1 * vec2)
    )doc");

template <typename Device, typename T>
class DefReduceInnerProductOp : public OpKernel {
  private:
    int sSize_;

  public:
    explicit DefReduceInnerProductOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("share_memory_size", &sSize_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& vec1 = context->input(0);
      const Tensor& vec2 = context->input(1);

      const int n = vec1.shape().dim_size(0);
      DCHECK_EQ(n, vec2.shape().dim_size(0));

      // Create output tensors
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &recv));

      ComputeMethods<T>::reduceInnerProductLauncher(
        context->eigen_device<Device>(), sSize_, 
        vec1.flat<T>().data(), vec2.flat<T>().data(), n, recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefReduceInnerProduct").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefReduceInnerProductOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceInnerProduct").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefReduceInnerProductOp<Eigen::ThreadPoolDevice, double>);

REGISTER_KERNEL_BUILDER(
  Name("DefReduceInnerProduct").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefReduceInnerProductOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceInnerProduct").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefReduceInnerProductOp<Eigen::GpuDevice, double>);

// ---------------------------- reduce double inner product ------------------------------------
REGISTER_OP("DefReduceDoubleInnerProduct")
  .Input("vec1: T")
  .Input("vec2: T")
  .Input("vec3: T")
  .Output("recv: T")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .Attr("share_memory_size: int = 128")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle vec1Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &vec1Shape));
    ShapeHandle vec2Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &vec2Shape));
    ShapeHandle vec3Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 1, &vec3Shape));

    context->set_output(0, context->Scalar());

    return Status::OK();
  })
  .Doc(R"doc(
    recv = sum(vec)
    )doc");

template <typename Device, typename T>
class DefReduceDoubleInnerProductOp : public OpKernel {
  private:
    int sSize_;

  public:
    explicit DefReduceDoubleInnerProductOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("share_memory_size", &sSize_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& vec1 = context->input(0);
      const Tensor& vec2 = context->input(1);
      const Tensor& vec3 = context->input(2);

      const int n = vec1.shape().dim_size(0);
      DCHECK_EQ(n, vec2.shape().dim_size(0));
      DCHECK_EQ(n, vec3.shape().dim_size(0));

      // Create output tensors
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &recv));

      ComputeMethods<T>::reduceDoubleInnerProductLauncher(
        context->eigen_device<Device>(), sSize_, 
        vec1.flat<T>().data(), vec2.flat<T>().data(), vec3.flat<T>().data(), n, recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefReduceDoubleInnerProduct").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefReduceDoubleInnerProductOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceDoubleInnerProduct").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefReduceDoubleInnerProductOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceDoubleInnerProduct").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefReduceDoubleInnerProductOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefReduceDoubleInnerProduct").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefReduceDoubleInnerProductOp<Eigen::GpuDevice, double>);

// -------------------------------- add n vectors ------------------------------------
REGISTER_OP("DefAddN")
  .Input("vecs: m * T")
  .Output("rhs: T")
  .Attr("m: int >= 2")
  .Attr("T: {float, double} = DT_FLOAT")
  .SetShapeFn([](InferenceContext* context) {
    for (int i = 0; i < context->num_inputs(); i++) {
      ShapeHandle shape;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(i), 1, &shape));
    }

    std::vector<DimensionHandle> rhsDims;
    rhsDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle rhsShape = context->MakeShape(rhsDims);
    context->set_output(0, rhsShape);

    return Status::OK();
  })
  .Doc(R"doc(
    vec1 + vec2 + ...
  )doc");

template <typename Device, typename T>
class DefAddNOp : public OpKernel {
  private:
    int m_;

  public:
    explicit DefAddNOp(OpKernelConstruction* context) : OpKernel(context) {
      m_ = context->num_inputs();
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors  TODO: fix here.
      const T* vecs[m_];
      for (int i = 0; i < m_; i++) {
        const T* data = context->input(i).flat<T>().data();
        vecs[i] = data;
      }

      const int n = context->input(0).shape().dim_size(0);
      for (int i = 0; i < m_; i ++) {
        DCHECK_EQ(n, context->input(i).shape().dim_size(0));
      }

      // Create an output tensor
      Tensor* rhs = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n}), &rhs));

      // compute
      ComputeMethods<T>::addNLauncher(
        context->eigen_device<Device>(), 
        vecs, m_, n, rhs->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefAddN").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefAddNOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefAddN").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefAddNOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefAddN").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefAddNOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefAddN").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefAddNOp<Eigen::GpuDevice, double>);

// -------------------------------- matrix Multiply ------------------------------------
REGISTER_OP("DefMatMul")
  .Input("mat1: T")
  .Input("mat2: T")
  .Output("recv: T")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .Attr("transpose: bool = false")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle mat1Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &mat1Shape));
    ShapeHandle mat2Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &mat2Shape));

    std::vector<DimensionHandle> recv_dims;
    recv_dims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    bool transpose;
    TF_RETURN_IF_ERROR(context->GetAttr("transpose", &transpose));
    DimensionHandle l = transpose ? context->Dim(context->input(1), 0) : context->Dim(context->input(1), 1);
    recv_dims.push_back(context->MakeDim(l));
    ShapeHandle recv_shape = context->MakeShape(recv_dims);
    context->set_output(0, recv_shape);

    return Status::OK();
  })
  .Doc(R"doc(
    recv = transpose ? mat1@mat2.T : mat1@mat2
    )doc");

template <typename Device, typename T>
class DefMatMulOp : public OpKernel {
  private:
    bool transpose_;

  public:
    explicit DefMatMulOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("transpose", &transpose_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& mat1 = context->input(0);
      const Tensor& mat2 = context->input(1);

      const int m = mat1.shape().dim_size(0);
      const int n = mat1.shape().dim_size(1);
      const int l = transpose_ ? mat2.shape().dim_size(0) : mat2.shape().dim_size(1);
      if (transpose_) {
        DCHECK_EQ(n, mat2.shape().dim_size(1));
      }
      else {
        DCHECK_EQ(n, mat2.shape().dim_size(0));
      }

      // Create output tensors
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({m, l}), &recv));

      if (transpose_) {
        // compute
        ComputeMethods<T>::matMulLauncher(
          context->eigen_device<Device>(), 
          mat1.flat<T>().data(), mat2.flat<T>().data(), m, n, l, recv->template flat<T>().data()
        );
      }
      else {
        // Create temp tensor
        Tensor temp;
        OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({l, n}), &temp));

        // compute
        ComputeMethods<T>::transposeLauncher(
          context->eigen_device<Device>(), 
          mat2.flat<T>().data(), n, l, temp.template flat<T>().data()
        );

        ComputeMethods<T>::matMulLauncher(
          context->eigen_device<Device>(), 
          mat1.flat<T>().data(), temp.template flat<T>().data(), m, n, l, recv->template flat<T>().data()
        );
      }
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefMatMul").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefMatMulOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMatMul").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefMatMulOp<Eigen::ThreadPoolDevice, double>);
  
REGISTER_KERNEL_BUILDER(
  Name("DefMatMul").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefMatMulOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMatMul").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefMatMulOp<Eigen::GpuDevice, double>);

// -------------------------------- Kronecker Product ------------------------------------
REGISTER_OP("DefKroneckerProduct")
  .Input("mat1: T")
  .Input("mat2: T")
  .Output("recv: T")
  .Attr("T: {int32, float, double} = DT_FLOAT")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle mat1Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &mat1Shape));
    ShapeHandle mat2Shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &mat2Shape));

    DimensionHandle m = context->Dim(context->input(0), 0);
    DimensionHandle n = context->Dim(context->input(0), 1);
    DimensionHandle p = context->Dim(context->input(1), 0);
    DimensionHandle q = context->Dim(context->input(1), 1);

    DimensionHandle mp;
    TF_RETURN_IF_ERROR(context->Multiply(m, p, &mp));
    DimensionHandle nq;
    TF_RETURN_IF_ERROR(context->Multiply(n, q, &nq));

    std::vector<DimensionHandle> recv_dims;
    recv_dims.push_back(context->MakeDim(mp));
    recv_dims.push_back(context->MakeDim(nq));
    ShapeHandle recv_shape = context->MakeShape(recv_dims);
    context->set_output(0, recv_shape);

    return Status::OK();
  })
  .Doc(R"doc(
    recv = mat1 X mat2
    )doc");

template <typename Device, typename T>
class DefKroneckerProductOp : public OpKernel {
  public:
    explicit DefKroneckerProductOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& mat1 = context->input(0);
      const Tensor& mat2 = context->input(1);

      const int m = mat1.shape().dim_size(0);
      const int n = mat1.shape().dim_size(1);
      const int p = mat2.shape().dim_size(0);
      const int q = mat2.shape().dim_size(1);

      // Create output tensors
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({m * p, n * q}), &recv));

      // compute
      ComputeMethods<T>::kroneckerProductLauncher(
        context->eigen_device<Device>(), 
        mat1.flat<T>().data(), mat2.flat<T>().data(), m, n, p, q, recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefKroneckerProduct").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefKroneckerProductOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefKroneckerProduct").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefKroneckerProductOp<Eigen::ThreadPoolDevice, double>);
  
REGISTER_KERNEL_BUILDER(
  Name("DefKroneckerProduct").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefKroneckerProductOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefKroneckerProduct").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefKroneckerProductOp<Eigen::GpuDevice, double>);

// -------------------------------- PLU decomposition ------------------------------------
REGISTER_OP("DefPlu")
  .Input("m: T")
  .Output("pi: int32")
  .Output("l: T")
  .Output("u: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("one_hot: bool = false")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle MShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &MShape));

    std::vector<DimensionHandle> piDims;
    piDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    bool oneHot;
    TF_RETURN_IF_ERROR(context->GetAttr("one_hot", &oneHot));
    if (oneHot) 
      piDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle piShape = context->MakeShape(piDims);
    context->set_output(0, piShape);

    std::vector<DimensionHandle> LDims;
    LDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    LDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle LShape = context->MakeShape(LDims);
    context->set_output(1, LShape);

    std::vector<DimensionHandle> UDims;
    UDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    UDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle UShape = context->MakeShape(UDims);
    context->set_output(2, UShape);

    return Status::OK();
  })
  .Doc(R"doc(
    pi is a permutation group. P is permutation matrix.
    L * U = P * M
  )doc");

template <typename Device, typename T>
class DefPluOp : public OpKernel {
  private:
    bool oneHot_;

  public:
    explicit DefPluOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("one_hot", &oneHot_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& M = context->input(0);
      int n = M.shape().dim_size(0);
      DCHECK_EQ(M.shape().dim_size(1), n);

      if (oneHot_) {
        // Create temp tensor
        Tensor tempPi;
        OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape({n}), &tempPi));

        // Create output tensors
        Tensor* pi = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n, n}), &pi));
        Tensor* L = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({n, n}), &L));
        Tensor* U = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({n, n}), &U));

        // compute
        ComputeMethods<T>::pluLauncher(
          context->eigen_device<Device>(), 
          M.flat<T>().data(), 
          n, 
          tempPi.template flat<int>().data(), 
          L->template flat<T>().data(), 
          U->template flat<T>().data()
        );

        ComputeMethods<T>::oneHotLauncher(
          context->eigen_device<Device>(), 
          tempPi.template flat<int>().data(), 
          n,
          pi->template flat<int>().data()
        );
      }
      else {
        // Create output tensors
        Tensor* pi = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n}), &pi));
        Tensor* L = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({n, n}), &L));
        Tensor* U = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({n, n}), &U));

        // compute
        ComputeMethods<T>::pluLauncher(
          context->eigen_device<Device>(), 
          M.flat<T>().data(), 
          n, 
          pi->template flat<int>().data(), 
          L->template flat<T>().data(), 
          U->template flat<T>().data()
        );
      }
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefPlu").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefPluOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPlu").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefPluOp<Eigen::ThreadPoolDevice, double>);

REGISTER_KERNEL_BUILDER(
  Name("DefPlu").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefPluOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPlu").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefPluOp<Eigen::GpuDevice, double>);

// -------------------------------- PLU solve ------------------------------------
REGISTER_OP("DefPluSolve")
  .Input("mat: T")
  .Input("rhs: T")
  .Output("recv: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle MatShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &MatShape));
    ShapeHandle rhsShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &rhsShape));

    std::vector<DimensionHandle> recvDims;
    recvDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle recvShape = context->MakeShape(recvDims);
    context->set_output(0, recvShape);

    return Status::OK();
  })
  .Doc(R"doc(
    Mat@recv = rhs
  )doc");

template <typename Device, typename T>
class DefPluSolveOp : public OpKernel {
  public:
    explicit DefPluSolveOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& Mat = context->input(0);
      const Tensor& rhs = context->input(1);

      int n = Mat.shape().dim_size(0);
      DCHECK_EQ(Mat.shape().dim_size(1), n);
      DCHECK_EQ(rhs.shape().dim_size(0), n);

      // Create temp tensors
      Tensor pi;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape({n}), &pi));
      Tensor L;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n, n}), &L));
      Tensor U;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n, n}), &U));

      // compute
      ComputeMethods<T>::pluLauncher(
        context->eigen_device<Device>(), 
        Mat.flat<T>().data(), 
        n, 
        pi.template flat<int>().data(), 
        L.template flat<T>().data(), 
        U.template flat<T>().data()
      );

      // Create an output tensor
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n}), &recv));

      // compute
      ComputeMethods<T>::pluSolveLauncher(
        context->eigen_device<Device>(), 
        pi.template flat<int>().data(), 
        L.template flat<T>().data(), 
        U.template flat<T>().data(),
        rhs.flat<T>().data(),
        n, 
        recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefPluSolve").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefPluSolveOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPluSolve").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefPluSolveOp<Eigen::ThreadPoolDevice, double>);

REGISTER_KERNEL_BUILDER(
  Name("DefPluSolve").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefPluSolveOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPluSolve").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefPluSolveOp<Eigen::GpuDevice, double>);

// -------------------------------- PLU gradients ------------------------------------
REGISTER_OP("DefPluGrad")
  .Input("pi: int32")
  .Input("l: T")
  .Input("u: T")
  .Input("grad_l: T")
  .Input("grad_u: T")
  .Output("recv: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle piShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &piShape));

    ShapeHandle LShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &LShape));

    ShapeHandle UShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 2, &UShape));

    ShapeHandle gradLShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 2, &gradLShape));

    ShapeHandle gradUShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(4), 2, &gradUShape));

    std::vector<DimensionHandle> recvDims;
    recvDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    recvDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    ShapeHandle recvShape = context->MakeShape(recvDims);
    context->set_output(0, recvShape);

    return Status::OK();
  })
  .Doc(R"doc(
    einsum('ijkl,kl->ij', A, gradM) = b
  )doc");

template <typename Device, typename T>
class DefPluGradOp : public OpKernel {
  public:
    explicit DefPluGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& pi = context->input(0);
      const Tensor& L = context->input(1);
      const Tensor& U = context->input(2);
      const Tensor& gradL = context->input(3);
      const Tensor& gradU = context->input(4);

      // const TensorShape& piShape = pi.shape();
      int n = pi.shape().dim_size(0);
      DCHECK_EQ(L.shape().dim_size(0), n);
      DCHECK_EQ(L.shape().dim_size(1), n);
      DCHECK_EQ(U.shape().dim_size(0), n);
      DCHECK_EQ(U.shape().dim_size(1), n);
      DCHECK_EQ(gradL.shape().dim_size(0), n);
      DCHECK_EQ(gradL.shape().dim_size(1), n);
      DCHECK_EQ(gradU.shape().dim_size(0), n);
      DCHECK_EQ(gradU.shape().dim_size(1), n);

      // Create temp tensors
      Tensor Mat;
      OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n * n, n * n}), &Mat));
      Tensor rhs;
      OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n * n}), &rhs));

      // compute
      ComputeMethods<T>::pluGradEqsLauncher(
        context->eigen_device<Device>(), 
        pi.flat<int>().data(), 
        L.flat<T>().data(), 
        U.flat<T>().data(), 
        gradL.flat<T>().data(), 
        gradU.flat<T>().data(), 
        n, 
        Mat.template flat<T>().data(), 
        rhs.template flat<T>().data()
      );

      // Create temp tensors
      Tensor squarePi;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape({n * n}), &squarePi));
      Tensor squareL;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n * n, n * n}), &squareL));
      Tensor squareU;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({n * n, n * n}), &squareU));

      // compute
      ComputeMethods<T>::pluLauncher(
        context->eigen_device<Device>(), 
        Mat.template flat<T>().data(), 
        n * n, 
        squarePi.template flat<int>().data(), 
        squareL.template flat<T>().data(), 
        squareU.template flat<T>().data()
      );

      // Create output tensor
      Tensor* recv = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n, n}), &recv));

      ComputeMethods<T>::pluSolveLauncher(
        context->eigen_device<Device>(), 
        squarePi.template flat<int>().data(), 
        squareL.template flat<T>().data(), 
        squareU.template flat<T>().data(),
        rhs.template flat<T>().data(),
        n * n, 
        recv->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefPluGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefPluGradOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPluGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefPluGradOp<Eigen::ThreadPoolDevice, double>);

REGISTER_KERNEL_BUILDER(
  Name("DefPluGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefPluGradOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefPluGrad").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefPluGradOp<Eigen::GpuDevice, double>);

// -------------------------------- conv2d ------------------------------------
REGISTER_OP("DefConv2d")
  .Input("input_tensor: T")
  .Input("kernel: T")
  .Output("output_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &inputTensorShape));

    ShapeHandle kernelShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &kernelShape));

    std::vector<int> strides;
    TF_RETURN_IF_ERROR(context->GetAttr("strides", &strides));
    std::string padding;
    TF_RETURN_IF_ERROR(context->GetAttr("padding", &padding));

    // compute output shape.
    int inputH = context->Value(context->Dim(context->input(0), 1));
    int inputW = context->Value(context->Dim(context->input(0), 2));
    int inputC = context->Value(context->Dim(context->input(0), 3));
    int kernelH = context->Value(context->Dim(context->input(1), 0));
    int kernelW = context->Value(context->Dim(context->input(1), 1));
    
    int outputH, outputW;
    if (padding == "SAME") {
      outputH = ceil(float(inputH) / float(strides[1]));
      outputW = ceil(float(inputW) / float(strides[2]));
    } 
    else {
      outputH = ceil(float(inputH - kernelH + 1) / float(strides[1]));
      outputW = ceil(float(inputW - kernelW + 1) / float(strides[2]));
    }

    std::vector<DimensionHandle> outputDims;
    outputDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    outputDims.push_back(context->MakeDim(outputH));
    outputDims.push_back(context->MakeDim(outputW));
    outputDims.push_back(context->MakeDim(context->Dim(context->input(1), 3)));
    ShapeHandle outputShape = context->MakeShape(outputDims);
    context->set_output(0, outputShape);

    return Status::OK();
  })
  .Doc(R"doc(
    outTensor = conv2d(inTensor, kernel, strides, padding)
  )doc");

template <typename Device, typename T>
class DefConv2dOp : public OpKernel {
  private:
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefConv2dOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& inputTensor = context->input(0);
      const Tensor& kernel = context->input(1);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int inputC = inputTensor.shape().dim_size(3);

      int kernelH = kernel.shape().dim_size(0);
      int kernelW = kernel.shape().dim_size(1);
      DCHECK_EQ(kernel.shape().dim_size(2), inputC);
      int outputC = kernel.shape().dim_size(3);

      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      // compute padding tensor.
      int outputH, outputW, padH, padW;
      Tensor padTensor;
      if (padding_ == "SAME") {
        outputH = ceil(float(inputH) / float(strides_[1]));
        outputW = ceil(float(inputW) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, inputC}), &padTensor));

        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;
        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, startH, startW, padTensor.template flat<T>().data()
        );

      }
      else {
        DCHECK_GE(inputH, kernelH);
        DCHECK_GE(inputW, kernelW);

        outputH = ceil(float(inputH - kernelH + 1) / float(strides_[1]));
        outputW = ceil(float(inputW - kernelW + 1) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, inputC}), &padTensor));

        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, padTensor.template flat<T>().data()
        );
      }

      // compute conv2d
      Tensor* outputTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, outputH, outputW, outputC}), &outputTensor));
      ComputeMethods<T>::conv2dLauncher(
        context->eigen_device<Device>(), 
        padTensor.template flat<T>().data(), N, padH, padW, inputC, 
        kernel.flat<T>().data(), kernelH, kernelW, outputC, 
        strides_[1], strides_[2], 
        outputH, outputW, outputTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefConv2d").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefConv2dOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2d").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefConv2dOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2d").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefConv2dOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2d").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefConv2dOp<Eigen::GpuDevice, double>);

// -------------------------------- conv2d grad ------------------------------------
REGISTER_OP("DefConv2dGrad")
  .Input("grad_tensor: T")
  .Input("input_tensor: T")
  .Input("kernel: T")
  .Output("grad_input_tensor: T")
  .Output("grad_kernel: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle gradTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &gradTensorShape));

    ShapeHandle inputShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &inputShape));

    ShapeHandle kernelShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &kernelShape));

    context->set_output(0, context->input(1));
    context->set_output(1, context->input(2));

    return Status::OK();
  })
  .Doc(R"doc(
    grad_input_tensor, grad_kernel = grad->(gradTensor, input_tensor, kernel)
  )doc");

template <typename Device, typename T>
class DefConv2dGradOp : public OpKernel {
  private:
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefConv2dGradOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradTensor = context->input(0);
      const Tensor& inputTensor = context->input(1);
      const Tensor& kernel = context->input(2);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int inputC = inputTensor.shape().dim_size(3);

      int kernelH = kernel.shape().dim_size(0);
      int kernelW = kernel.shape().dim_size(1);
      DCHECK_EQ(kernel.shape().dim_size(2), inputC);
      int outputC = kernel.shape().dim_size(3);

      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      /* compute gradient of input tensor */
      int outputH, outputW, padH, padW;
      if (padding_ == "SAME") {
        outputH = ceil(float(inputH) / float(strides_[1]));
        outputW = ceil(float(inputW) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }
      else {
        outputH = ceil(float(inputH - kernelH + 1) / float(strides_[1]));
        outputW = ceil(float(inputW - kernelW + 1) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }
      // compute the padding grad of input tensor
      Tensor gradPadTensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, inputC}), &gradPadTensor));
      ComputeMethods<T>::gradConv2dInputLauncher(
        context->eigen_device<Device>(), 
        gradTensor.flat<T>().data(), N, outputH, outputW, outputC, 
        kernel.flat<T>().data(), kernelH, kernelW, 
        strides_[1], strides_[2], 
        padH, padW, inputC, gradPadTensor.template flat<T>().data()
      );
      // compute unpadding of grad input tensor
      Tensor* gradInputTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, inputH, inputW, inputC}), &gradInputTensor));
      if (padding_ == "SAME") {
        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;

        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, startH, startW, 
          N, inputH, inputW, inputC, gradInputTensor->template flat<T>().data()
        );
      }
      else {
        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, 
          N, inputH, inputW, inputC, gradInputTensor->template flat<T>().data()
        );
      }

      // compute the padding of input tensor
      Tensor padTensor = gradPadTensor;
      if (padding_ == "SAME") {
        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;
        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, startH, startW, padTensor.template flat<T>().data()
        );
      }
      else {
        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, padTensor.template flat<T>().data()
        );
      }
      // compute the gradient of kernel tensor
      Tensor* gradKernelTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({kernelH, kernelW, inputC, outputC}), &gradKernelTensor));
      ComputeMethods<T>::gradConv2dKernelLauncher(
        context->eigen_device<Device>(), 
        gradTensor.flat<T>().data(), N, outputH, outputW, outputC, 
        padTensor.template flat<T>().data(), padH, padW, inputC, 
        strides_[1], strides_[2], 
        kernelH, kernelW, gradKernelTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefConv2dGradOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefConv2dGradOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefConv2dGradOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGrad").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefConv2dGradOp<Eigen::GpuDevice, double>);

// -------------------------------- conv2d grad input ------------------------------------
REGISTER_OP("DefConv2dGradInput")
  .Input("grad_tensor: T")
  .Input("input_tensor: T")
  .Input("kernel: T")
  .Output("grad_input_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle gradTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &gradTensorShape));

    ShapeHandle inputShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &inputShape));

    ShapeHandle kernelShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &kernelShape));

    context->set_output(0, context->input(1));

    return Status::OK();
  })
  .Doc(R"doc(
    grad_input_tensor = grad->(gradTensor, input_tensor, kernel)
  )doc");

template <typename Device, typename T>
class DefConv2dGradInputOp : public OpKernel {
  private:
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefConv2dGradInputOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradTensor = context->input(0);
      const Tensor& inputTensor = context->input(1);
      const Tensor& kernel = context->input(2);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int inputC = inputTensor.shape().dim_size(3);

      int kernelH = kernel.shape().dim_size(0);
      int kernelW = kernel.shape().dim_size(1);
      DCHECK_EQ(kernel.shape().dim_size(2), inputC);
      int outputC = kernel.shape().dim_size(3);

      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      /* compute gradient of input tensor */
      int outputH, outputW, padH, padW;
      if (padding_ == "SAME") {
        outputH = ceil(float(inputH) / float(strides_[1]));
        outputW = ceil(float(inputW) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }
      else {
        outputH = ceil(float(inputH - kernelH + 1) / float(strides_[1]));
        outputW = ceil(float(inputW - kernelW + 1) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }
      // compute the padding grad of input tensor
      Tensor gradPadTensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, inputC}), &gradPadTensor));
      ComputeMethods<T>::gradConv2dInputLauncher(
        context->eigen_device<Device>(), 
        gradTensor.flat<T>().data(), N, outputH, outputW, outputC, 
        kernel.flat<T>().data(), kernelH, kernelW, 
        strides_[1], strides_[2], 
        padH, padW, inputC, gradPadTensor.template flat<T>().data()
      );
      // compute unpadding of grad input tensor
      Tensor* gradInputTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, inputH, inputW, inputC}), &gradInputTensor));
      if (padding_ == "SAME") {
        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;

        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, startH, startW, 
          N, inputH, inputW, inputC, gradInputTensor->template flat<T>().data()
        );
      }
      else {
        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, 
          N, inputH, inputW, inputC, gradInputTensor->template flat<T>().data()
        );
      }
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradInput").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefConv2dGradInputOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradInput").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefConv2dGradInputOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradInput").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefConv2dGradInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradInput").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefConv2dGradOp<Eigen::GpuDevice, double>);

// -------------------------------- conv2d grad kernel ------------------------------------
REGISTER_OP("DefConv2dGradKernel")
  .Input("grad_tensor: T")
  .Input("input_tensor: T")
  .Input("kernel: T")
  .Output("grad_kernel: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle gradTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &gradTensorShape));

    ShapeHandle inputShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &inputShape));

    ShapeHandle kernelShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &kernelShape));

    context->set_output(0, context->input(2));

    return Status::OK();
  })
  .Doc(R"doc(
    grad_kernel = grad->(gradTensor, input_tensor, kernel)
  )doc");

template <typename Device, typename T>
class DefConv2dGradKernelOp : public OpKernel {
  private:
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefConv2dGradKernelOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradTensor = context->input(0);
      const Tensor& inputTensor = context->input(1);
      const Tensor& kernel = context->input(2);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int inputC = inputTensor.shape().dim_size(3);

      int kernelH = kernel.shape().dim_size(0);
      int kernelW = kernel.shape().dim_size(1);
      DCHECK_EQ(kernel.shape().dim_size(2), inputC);
      int outputC = kernel.shape().dim_size(3);

      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      /* compute gradient of input tensor */
      int outputH, outputW, padH, padW;
      if (padding_ == "SAME") {
        outputH = ceil(float(inputH) / float(strides_[1]));
        outputW = ceil(float(inputW) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }
      else {
        outputH = ceil(float(inputH - kernelH + 1) / float(strides_[1]));
        outputW = ceil(float(inputW - kernelW + 1) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + kernelH;
        padW = (outputW - 1) * strides_[2] + kernelW;
      }

      // compute the padding of input tensor
      Tensor padTensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, inputC}), &padTensor));
      if (padding_ == "SAME") {
        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;
        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, startH, startW, padTensor.template flat<T>().data()
        );
      }
      else {
        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, inputC, 
          padH, padW, padTensor.template flat<T>().data()
        );
      }
      // compute the gradient of kernel tensor
      Tensor* gradKernelTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({kernelH, kernelW, inputC, outputC}), &gradKernelTensor));
      ComputeMethods<T>::gradConv2dKernelLauncher(
        context->eigen_device<Device>(), 
        gradTensor.flat<T>().data(), N, outputH, outputW, outputC, 
        padTensor.template flat<T>().data(), padH, padW, inputC, 
        strides_[1], strides_[2], 
        kernelH, kernelW, gradKernelTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradKernel").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefConv2dGradKernelOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradKernel").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefConv2dGradKernelOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradKernel").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefConv2dGradKernelOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefConv2dGradKernel").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefConv2dGradOp<Eigen::GpuDevice, double>);

// -------------------------------- bias add ------------------------------------
REGISTER_OP("DefBiasAdd")
  .Input("input_tensor: T")
  .Input("bias: T")
  .Output("output_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("dim: int = 2")
  .SetShapeFn([](InferenceContext* context) {
    int dim;
    TF_RETURN_IF_ERROR(context->GetAttr("dim", &dim));

    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), dim, &inputTensorShape));

    ShapeHandle biasShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &biasShape));

    context->set_output(0, context->input(0));

    return Status::OK();
  })
  .Doc(R"doc(
    outTensor = conv2d(inTensor, kernel, strides, padding)
  )doc");

template <typename Device, typename T>
class DefBiasAddOp : public OpKernel {
  private:
    int dim_;

  public:
    explicit DefBiasAddOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& inputTensor = context->input(0);
      const Tensor& bias = context->input(1);

      if (dim_ == 4) {
        int N = inputTensor.shape().dim_size(0);
        int H = inputTensor.shape().dim_size(1);
        int W = inputTensor.shape().dim_size(2);
        int C = inputTensor.shape().dim_size(3);

        DCHECK_EQ(bias.shape().dim_size(0), C);

        Tensor* outputTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, H, W, C}), &outputTensor));
        ComputeMethods<T>::biasAddLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N * H * W, C, 
          bias.flat<T>().data(), outputTensor->template flat<T>().data()
        );
      }
      else {
        int N = inputTensor.shape().dim_size(0);
        int C = inputTensor.shape().dim_size(1);

        DCHECK_EQ(bias.shape().dim_size(0), C);

        Tensor* outputTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, C}), &outputTensor));
        ComputeMethods<T>::biasAddLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, C, 
          bias.flat<T>().data(), outputTensor->template flat<T>().data()
        );
      }
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefBiasAdd").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefBiasAddOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefBiasAdd").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefBiasAddOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefBiasAdd").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefBiasAddOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefBiasAdd").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefBiasAddOp<Eigen::GpuDevice, double>);

// -------------------------------- max pool ------------------------------------
REGISTER_OP("DefMaxPool")
  .Input("input_tensor: T")
  .Output("output_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("ksize: list(int)")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &inputTensorShape));

    std::vector<int> strides;
    TF_RETURN_IF_ERROR(context->GetAttr("strides", &strides));
    std::vector<int> ksize;
    TF_RETURN_IF_ERROR(context->GetAttr("ksize", &ksize));
    std::string padding;
    TF_RETURN_IF_ERROR(context->GetAttr("padding", &padding));

    // compute output shape.
    int inputH = context->Value(context->Dim(context->input(0), 1));
    int inputW = context->Value(context->Dim(context->input(0), 2));
    
    int outputH, outputW;
    if (padding == "SAME") {
      outputH = ceil(float(inputH) / float(strides[1]));
      outputW = ceil(float(inputW) / float(strides[2]));
    } 
    else {
      outputH = ceil(float(inputH - ksize[1] + 1) / float(strides[1]));
      outputW = ceil(float(inputW - ksize[2] + 1) / float(strides[2]));
    }

    std::vector<DimensionHandle> outputDims;
    outputDims.push_back(context->MakeDim(context->Dim(context->input(0), 0)));
    outputDims.push_back(context->MakeDim(outputH));
    outputDims.push_back(context->MakeDim(outputW));
    outputDims.push_back(context->MakeDim(context->Dim(context->input(0), 3)));
    ShapeHandle outputShape = context->MakeShape(outputDims);
    context->set_output(0, outputShape);

    return Status::OK();
  })
  .Doc(R"doc(
    outTensor = maxpool(inputTensor, ksize, strides, padding)
  )doc");

template <typename Device, typename T>
class DefMaxPoolOp : public OpKernel {
  private:
    std::vector<int> ksize_;
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefMaxPoolOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& inputTensor = context->input(0);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int C = inputTensor.shape().dim_size(3);

      DCHECK_EQ(ksize_[0], 1);
      DCHECK_EQ(ksize_[3], 1);
      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      // compute padding tensor.
      int outputH, outputW, padH, padW;
      Tensor padTensor;
      if (padding_ == "SAME") {
        outputH = ceil(float(inputH) / float(strides_[1]));
        outputW = ceil(float(inputW) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + ksize_[1];
        padW = (outputW - 1) * strides_[2] + ksize_[2];
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, C}), &padTensor));

        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;
        ComputeMethods<T>::maxPoolPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, C, 
          padH, padW, startH, startW, padTensor.template flat<T>().data()
        );
      }
      else {
        DCHECK_GE(inputH, ksize_[1]);
        DCHECK_GE(inputW, ksize_[2]);

        outputH = ceil(float(inputH - ksize_[1] + 1) / float(strides_[1]));
        outputW = ceil(float(inputW - ksize_[2] + 1) / float(strides_[2]));

        padH = (outputH - 1) * strides_[1] + ksize_[1];
        padW = (outputW - 1) * strides_[2] + ksize_[2];
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, C}), &padTensor));

        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, C, 
          padH, padW, padTensor.template flat<T>().data()
        );
      }

      // compute max pool
      Tensor* outputTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, outputH, outputW, C}), &outputTensor));
      ComputeMethods<T>::maxPoolLauncher(
        context->eigen_device<Device>(), 
        padTensor.template flat<T>().data(), N, padH, padW, C, 
        ksize_[1], ksize_[2], 
        strides_[1], strides_[2], 
        outputH, outputW, outputTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefMaxPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefMaxPoolOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPool").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefMaxPoolOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefMaxPoolOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPool").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefMaxPoolOp<Eigen::GpuDevice, double>);


// -------------------------------- max pool grad ------------------------------------
REGISTER_OP("DefMaxPoolGrad")
  .Input("grad_output_tensor: T")
  .Input("input_tensor: T")
  .Input("output_tensor: T")
  .Output("grad_input_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("ksize: list(int)")
  .Attr("strides: list(int)")
  .Attr("padding: {'SAME', 'VALID'}")
  .SetShapeFn([](InferenceContext* context) {
    ShapeHandle gradOutputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &gradOutputTensorShape));

    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &inputTensorShape));

    ShapeHandle outputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &outputTensorShape));

    context->set_output(0, context->input(1));

    return Status::OK();
  })
  .Doc(R"doc(
    grad_input_tensor = grad->(gradInputTensor, input_tensor, output_tensor)
  )doc");

template <typename Device, typename T>
class DefMaxPoolGradOp : public OpKernel {
  private:
    std::vector<int> ksize_;
    std::vector<int> strides_;
    std::string padding_;

  public:
    explicit DefMaxPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradOutputTensor = context->input(0);
      const Tensor& inputTensor = context->input(1);
      const Tensor& outputTensor = context->input(2);

      int N = inputTensor.shape().dim_size(0);
      int inputH = inputTensor.shape().dim_size(1);
      int inputW = inputTensor.shape().dim_size(2);
      int C = inputTensor.shape().dim_size(3);

      int outputH = outputTensor.shape().dim_size(1);
      int outputW = outputTensor.shape().dim_size(2);

      DCHECK_EQ(ksize_[0], 1);
      DCHECK_EQ(ksize_[3], 1);
      DCHECK_EQ(strides_[0], 1);
      DCHECK_EQ(strides_[3], 1);

      /* compute gradient of input tensor */
      // compute the padding of input tensor
      int padH = (outputH - 1) * strides_[1] + ksize_[1];
      int padW = (outputW - 1) * strides_[2] + ksize_[2];
      Tensor padTensor;
      if (padding_ == "SAME") {
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, C}), &padTensor));

        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;
        ComputeMethods<T>::maxPoolPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, C, 
          padH, padW, startH, startW, padTensor.template flat<T>().data()
        );
      }
      else {
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, C}), &padTensor));

        ComputeMethods<T>::conv2dPaddingLauncher(
          context->eigen_device<Device>(), 
          inputTensor.flat<T>().data(), N, inputH, inputW, C, 
          padH, padW, padTensor.template flat<T>().data()
        );
      }

      // compute the padding grad of input tensor
      Tensor gradPadTensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({N, padH, padW, C}), &gradPadTensor));
      ComputeMethods<T>::gradMaxPoolLauncher(
        context->eigen_device<Device>(), 
        gradOutputTensor.flat<T>().data(), N, outputH, outputW, C, 
        ksize_[1], ksize_[2], 
        strides_[1], strides_[2], 
        outputTensor.flat<T>().data(), 
        padH, padW, padTensor.template flat<T>().data(), 
        gradPadTensor.template flat<T>().data()
      );

      // compute unpadding of grad input tensor
      Tensor* gradInputTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, inputH, inputW, C}), &gradInputTensor));
      if (padding_ == "SAME") {
        int startH = (padH - inputH) / 2;
        int startW = (padW - inputW) / 2;

        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, startH, startW, 
          N, inputH, inputW, C, gradInputTensor->template flat<T>().data()
        );
      }
      else {
        ComputeMethods<T>::conv2dUnpaddingLauncher(
          context->eigen_device<Device>(), 
          gradPadTensor.template flat<T>().data(), padH, padW, 
          N, inputH, inputW, C, gradInputTensor->template flat<T>().data()
        );
      }

    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefMaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefMaxPoolGradOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefMaxPoolGradOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefMaxPoolGradOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefMaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefMaxPoolGradOp<Eigen::GpuDevice, double>);

// -------------------------------- batch norm ------------------------------------
REGISTER_OP("DefBatchNorm")
  .Input("input_tensor: T")
  .Input("mean: T")
  .Input("variance: T")
  .Input("offset: T")
  .Input("scale: T")
  .Output("output_tensor: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("variance_epsilon: float")
  .Attr("dim: int = 4")
  .SetShapeFn([](InferenceContext* context) {
    int dim;
    TF_RETURN_IF_ERROR(context->GetAttr("dim", &dim));

    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), dim, &inputTensorShape));
    ShapeHandle meanShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &meanShape));
    ShapeHandle varianceShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 1, &varianceShape));
    ShapeHandle offsetShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 1, &offsetShape));
    ShapeHandle scaleShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(4), 1, &scaleShape));

    context->set_output(0, inputTensorShape);

    return Status::OK();
  })
  .Doc(R"doc(
    output_tensor = \\(\frac{scale(input_tensor - mean)}{sqrt(variance + variance_epsilon)} + offset\\)
  )doc");

template <typename Device, typename T>
class DefBatchNormOp : public OpKernel {
  private:
    float varianceEpsilon_;
    int dim_;

  public:
    explicit DefBatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("variance_epsilon", &varianceEpsilon_));
      OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& inputTensor = context->input(0);
      const Tensor& meanTensor = context->input(1);
      const Tensor& varianceTensor = context->input(2);
      const Tensor& offsetTensor = context->input(3);
      const Tensor& scaleTensor = context->input(4);

      int N, C;
      if (dim_ == 4) {
        N = inputTensor.shape().dim_size(0) * inputTensor.shape().dim_size(1) * inputTensor.shape().dim_size(2);
        C = inputTensor.shape().dim_size(3);
      }
      else {
        N = inputTensor.shape().dim_size(0);
        C = inputTensor.shape().dim_size(1);
      }

      DCHECK_EQ(meanTensor.shape().dim_size(0), C);
      DCHECK_EQ(varianceTensor.shape().dim_size(0), C);
      DCHECK_EQ(offsetTensor.shape().dim_size(0), C);
      DCHECK_EQ(scaleTensor.shape().dim_size(0), C);

      // compute batch normalization
      Tensor* outputTensor = nullptr;
      if (dim_ == 4) {
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
          inputTensor.shape().dim_size(0), inputTensor.shape().dim_size(1), inputTensor.shape().dim_size(2), C
          }), &outputTensor));
      }
      else {
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, C}), &outputTensor));
      }
      ComputeMethods<T>::batchNormLauncher(
        context->eigen_device<Device>(), 
        inputTensor.flat<T>().data(), N, C, 
        meanTensor.flat<T>().data(), 
        varianceTensor.flat<T>().data(), 
        offsetTensor.flat<T>().data(), 
        scaleTensor.flat<T>().data(), 
        (T)varianceEpsilon_, 
        outputTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefBatchNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefBatchNormOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefBatchNorm").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefBatchNormOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefBatchNorm").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefBatchNormOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefBatchNorm").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefBatchNormOp<Eigen::GpuDevice, double>);

// -------------------------------- batch norm grad ------------------------------------
REGISTER_OP("DefGradBatchNorm")
  .Input("grad_tensor: T")
  .Input("input_tensor: T")
  .Input("mean: T")
  .Input("variance: T")
  .Input("scale: T")
  .Output("grad_input_tensor: T")
  .Output("grad_mean: T")
  .Output("grad_variance: T")
  .Output("grad_offset: T")
  .Output("grad_scale: T")
  .Attr("T: {float, double} = DT_FLOAT")
  .Attr("variance_epsilon: float")
  .Attr("dim: int = 4")
  .SetShapeFn([](InferenceContext* context) {
    int dim;
    TF_RETURN_IF_ERROR(context->GetAttr("dim", &dim));

    ShapeHandle gradTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), dim, &gradTensorShape));
    ShapeHandle inputTensorShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), dim, &inputTensorShape));
    ShapeHandle meanShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 1, &meanShape));
    ShapeHandle varianceShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(3), 1, &varianceShape));
    ShapeHandle scaleShape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(4), 1, &scaleShape));

    context->set_output(0, gradTensorShape);
    context->set_output(1, meanShape);
    context->set_output(2, meanShape);
    context->set_output(3, meanShape);
    context->set_output(4, meanShape);

    return Status::OK();
  })
  .Doc(R"doc(
    grad_input_tensor, grad_mean, grad_variance, grad_offset, grad_scale = grad->BN(grad, input_tensor, mean, variance, scale)
  )doc");

template <typename Device, typename T>
class DefGradBatchNormOp : public OpKernel {
  private:
    float varianceEpsilon_;
    int dim_;

  public:
    explicit DefGradBatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("variance_epsilon", &varianceEpsilon_));
      OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradTensor = context->input(0);

      const Tensor& inputTensor = context->input(1);
      const Tensor& meanTensor = context->input(2);
      const Tensor& varianceTensor = context->input(3);
      const Tensor& scaleTensor = context->input(4);

      int N, C;
      if (dim_ == 4) {
        N = gradTensor.shape().dim_size(0) * gradTensor.shape().dim_size(1) * gradTensor.shape().dim_size(2);
        C = gradTensor.shape().dim_size(3);

        DCHECK_EQ(inputTensor.shape().dim_size(0), gradTensor.shape().dim_size(0));
        DCHECK_EQ(inputTensor.shape().dim_size(1), gradTensor.shape().dim_size(1));
        DCHECK_EQ(inputTensor.shape().dim_size(2), gradTensor.shape().dim_size(2));
        DCHECK_EQ(inputTensor.shape().dim_size(3), gradTensor.shape().dim_size(3));
        DCHECK_EQ(meanTensor.shape().dim_size(0), C);
        DCHECK_EQ(varianceTensor.shape().dim_size(0), C);
        DCHECK_EQ(scaleTensor.shape().dim_size(0), C);
      }
      else {
        N = gradTensor.shape().dim_size(0);
        C = gradTensor.shape().dim_size(1);

        DCHECK_EQ(inputTensor.shape().dim_size(0), N);
        DCHECK_EQ(inputTensor.shape().dim_size(1), C);
        DCHECK_EQ(meanTensor.shape().dim_size(0), C);
        DCHECK_EQ(varianceTensor.shape().dim_size(0), C);
        DCHECK_EQ(scaleTensor.shape().dim_size(0), C);
      }

      // compute batch normalization
      Tensor* gradInputTensor = nullptr;
      if (dim_ == 4) {
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
          gradTensor.shape().dim_size(0), gradTensor.shape().dim_size(1), gradTensor.shape().dim_size(2), C
          }), &gradInputTensor));
      }
      else {
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, C}), &gradInputTensor));
      }
      Tensor* gradMeanTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({C}), &gradMeanTensor));
      Tensor* gradVarianceTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({C}), &gradVarianceTensor));
      Tensor* gradOffsetTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({C}), &gradOffsetTensor));
      Tensor* gradScaleTensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({C}), &gradScaleTensor));
      ComputeMethods<T>::gradBatchNormLauncher(
        context->eigen_device<Device>(), 
        gradTensor.flat<T>().data(), N , C, (T)varianceEpsilon_, 
        inputTensor.flat<T>().data(), 
        meanTensor.flat<T>().data(), 
        varianceTensor.flat<T>().data(), 
        scaleTensor.flat<T>().data(), 
        gradInputTensor->template flat<T>().data(), 
        gradMeanTensor->template flat<T>().data(), 
        gradVarianceTensor->template flat<T>().data(), 
        gradOffsetTensor->template flat<T>().data(), 
        gradScaleTensor->template flat<T>().data()
      );
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DefGradBatchNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"), 
  DefGradBatchNormOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefGradBatchNorm").Device(DEVICE_CPU).TypeConstraint<double>("T"), 
  DefGradBatchNormOp<Eigen::ThreadPoolDevice, double>);
REGISTER_KERNEL_BUILDER(
  Name("DefGradBatchNorm").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  DefGradBatchNormOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
  Name("DefGradBatchNorm").Device(DEVICE_GPU).TypeConstraint<double>("T"), 
  DefGradBatchNormOp<Eigen::GpuDevice, double>);
