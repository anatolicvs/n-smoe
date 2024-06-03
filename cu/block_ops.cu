#include <torch/extension.h>

__global__ void extract_blocks_kernel(const float* __restrict__ src, float* __restrict__ dst, int batch, int channels, int height, int width, int block_size, int step) {
    int xBlock = blockIdx.x * blockDim.x + threadIdx.x;
    int yBlock = blockIdx.y * blockDim.y + threadIdx.y;
    int bChannel = blockIdx.z * blockDim.z + threadIdx.z;

    int bBlock = bChannel / channels;
    int zBlock = bChannel % channels;

    if (bBlock < batch && xBlock < (width - block_size + 1) / step && yBlock < (height - block_size + 1) / step && zBlock < channels) {
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                int src_index = bBlock * channels * width * height + zBlock * width * height + (yBlock * step + i) * width + (xBlock * step + j);
                int dst_index = bBlock * channels * (width - block_size + 1) * (height - block_size + 1) + zBlock * (width - block_size + 1) * (height - block_size + 1) + yBlock * ((width - block_size + 1) / step) + xBlock * block_size * block_size + i * block_size + j;
                dst[dst_index] = src[src_index];
            }
        }
    }
}

__global__ void reconstruct_kernel(const float* __restrict__ blocks, float* __restrict__ dst, int batch, int channels, int height, int width, int block_size, int step, int num_blocks_per_row, int num_blocks_per_column) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;  

    if (x < width && y < height && z < batch * channels) {
        int b = z / channels;
        int c = z % channels;

        float pixel_value = 0.0;
        float count = 0.0;

        int start_row = max(0, (y - block_size + 1 + step - 1) / step);
        int end_row = min(num_blocks_per_column - 1, y / step);
        int start_col = max(0, (x - block_size + 1 + step - 1) / step);
        int end_col = min(num_blocks_per_row - 1, x / step);

        for (int by = start_row; by <= end_row; ++by) {
            for (int bx = start_col; bx <= end_col; ++bx) {
                int block_idx = b * num_blocks_per_row * num_blocks_per_column * channels + c * num_blocks_per_row * num_blocks_per_column + by * num_blocks_per_row + bx;
                int in_block_x = x - bx * step;
                int in_block_y = y - by * step;
                pixel_value += blocks[block_idx * block_size * block_size + in_block_y * block_size + in_block_x];
                count += 1.0;
            }
        }

        if (count > 0) {
            int index = b * channels * width * height + c * width * height + y * width + x;
            dst[index] = pixel_value / count;
        }
    }
}

void extract_blocks(const torch::Tensor& src, torch::Tensor& dst, int batch, int channels, int height, int width, int block_size, int step) {
    int num_blocks_x = (width - block_size + step) / step;
    int num_blocks_y = (height - block_size + step) / step;
    int num_blocks_z = batch * channels;

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((num_blocks_x + threadsPerBlock.x - 1) / threadsPerBlock.x, (num_blocks_y + threadsPerBlock.y - 1) / threadsPerBlock.y, (num_blocks_z + threadsPerBlock.z - 1) / threadsPerBlock.z);

    extract_blocks_kernel<<<numBlocks, threadsPerBlock>>>(src.data_ptr<float>(), dst.data_ptr<float>(), batch, channels, height, width, block_size, step);
}

void reconstruct(const torch::Tensor& blocks, torch::Tensor& dst, int batch, int channels, int height, int width, int block_size, int step) {
    int num_blocks_per_row = (width - block_size + step - 1) / step + 1;
    int num_blocks_per_column = (height - block_size + step - 1) / step + 1;

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y, batch * channels);

    reconstruct_kernel<<<numBlocks, threadsPerBlock>>>(blocks.data_ptr<float>(), dst.data_ptr<float>(), batch, channels, height, width, block_size, step, num_blocks_per_row, num_blocks_per_column);
}

PYBIND11_MODULE(cuda_block_ops, m) {
    m.def("extract_blocks", &extract_blocks, "Extract blocks from an image tensor", py::arg("src"), py::arg("dst"), py::arg("batch"), py::arg("channels"), py::arg("height"), py::arg("width"), py::arg("block_size"), py::arg("step"));
    m.def("reconstruct", &reconstruct, "Reconstruct image from blocks", py::arg("blocks"), py::arg("dst"), py::arg("batch"), py::arg("channels"), py::arg("height"), py::arg("width"), py::arg("block_size"), py::arg("step"));
}
