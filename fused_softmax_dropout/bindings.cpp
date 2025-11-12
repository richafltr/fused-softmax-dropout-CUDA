#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declaration
void launch_fused_softmax_dropout(
    const at::Tensor& x,
    const at::Tensor& mask,
    at::Tensor& y,
    float p,
    bool training,
    at::PhiloxCudaState philox_state
);

torch::Tensor fused_softmax_dropout_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> mask,
    double p,
    bool training
) {
    // Check input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D (batch, seq, d_model)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    
    // Ensure contiguous
    x = x.contiguous();
    
    // Handle mask
    at::Tensor mask_contig;
    if (mask.has_value() && mask->defined()) {
        TORCH_CHECK(mask->is_cuda(), "Mask tensor must be on CUDA");
        mask_contig = mask->to(x.dtype()).contiguous();
        // Broadcast mask if needed
        if (mask_contig.sizes() != x.sizes()) {
            mask_contig = mask_contig.expand_as(x).contiguous();
        }
    }
    
    // Create output tensor
    at::Tensor y = torch::empty_like(x);
    
    // Setup RNG
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    int64_t numel = x.numel();
    at::PhiloxCudaState philox_state = at::cuda::detail::getPhiloxState(gen, numel);
    
    // Launch kernel
    at::Tensor mask_for_kernel = mask_contig.defined() ? mask_contig : at::Tensor();
    launch_fused_softmax_dropout(
        x,
        mask_for_kernel,
        y,
        static_cast<float>(p),
        training,
        philox_state
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_softmax_dropout", &fused_softmax_dropout_forward,
          "Fused softmax + dropout (CUDA)",
          py::arg("x"),
          py::arg("mask") = nullptr,
          py::arg("p") = 0.1,
          py::arg("training") = true);
}

