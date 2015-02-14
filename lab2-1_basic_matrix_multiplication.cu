#include <wb.h>

#define wbCheck(stmt)                                                          \
do {                                                                         \
cudaError_t err = stmt;                                                    \
if (err != cudaSuccess) {                                                  \
wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
return -1;                                                               \
}                                                                          \
} while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
int numAColumns, int numBRows, int numBColumns,
int numCRows, int numCColumns) {
//@@ Insert code to implement matrix multiplication here
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;
if (row < numCRows && col < numCColumns) {
float value = 0.0;
for (int i = 0; i < numAColumns; ++i) {
value += A[row * numAColumns + i] * B[i * numBColumns + col];
}
C[row * numCColumns + col] = value;
}

}

int main(int argc, char **argv) {
wbArg_t args;
float *hostA; // The A matrix
float *hostB; // The B matrix
float *hostC; // The output C matrix
float *deviceA;
float *deviceB;
float *deviceC;
int numARows;    // number of rows in the matrix A
int numAColumns; // number of columns in the matrix A
int numBRows;    // number of rows in the matrix B
int numBColumns; // number of columns in the matrix B
int numCRows;    // number of rows in the matrix C (you have to set this)
int numCColumns; // number of columns in the matrix C (you have to set this)

args = wbArg_read(argc, argv);

wbTime_start(Generic, "Importing data and creating memory on host");
hostA =
( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
hostB =
( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
//@@ Set numCRows and numCColumns
numCRows = numARows;
numCColumns = numBColumns;
//@@ Allocate the hostC matrix
wbTime_stop(Generic, "Importing data and creating memory on host");
hostC = (float*)malloc(sizeof(float) * numCColumns * numCRows);
memset(hostC, 0, sizeof(float) * numCColumns * numCRows);
wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

wbTime_start(GPU, "Allocating GPU memory.");
//@@ Allocate GPU memory here
cudaMalloc((void**)&deviceA, sizeof(float) * numARows * numAColumns);
cudaMalloc((void**)&deviceB, sizeof(float) * numBRows * numBColumns);
cudaMalloc((void**)&deviceC, sizeof(float) * numCRows * numCColumns);
wbLog(TRACE, "[Felica]The dimensions of C are ", numCRows, " x ", numCColumns, ", size = ", sizeof(float) * numCRows * numCColumns);

wbTime_stop(GPU, "Allocating GPU memory.");

wbTime_start(GPU, "Copying input memory to the GPU.");
//@@ Copy memory to the GPU here
cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
cudaMemcpy(deviceC, hostC, sizeof(float) * numCRows * numCColumns, cudaMemcpyHostToDevice);
wbTime_stop(GPU, "Copying input memory to the GPU.");

//@@ Initialize the grid and block dimensions here
int tile_width = 32;  //stangely, if I change 32 into 64, it will go wrong. I can not see the system requirement.
dim3 GridDim((numCColumns - 1) / tile_width + 1, (numCRows - 1) / tile_width + 1);  //NOTE: order of parameter, column first, then row.
dim3 BlockDIm(tile_width, tile_width);
wbTime_start(Compute, "Performing CUDA computation");
//@@ Launch the GPU Kernel here
matrixMultiply<<<GridDim, BlockDIm>>>(
deviceA, deviceB, deviceC,
numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
cudaDeviceSynchronize();
wbTime_stop(Compute, "Performing CUDA computation");

wbTime_start(Copy, "Copying output memory to the CPU");
//@@ Copy the GPU memory back to the CPU here
cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
wbTime_stop(Copy, "Copying output memory to the CPU");

wbTime_start(GPU, "Freeing GPU Memory");
//@@ Free the GPU memory here
cudaFree(deviceA);
cudaFree(deviceB);
cudaFree(deviceC);
wbTime_stop(GPU, "Freeing GPU Memory");

wbSolution(args, hostC, numCRows, numCColumns);

free(hostA);
free(hostB);
free(hostC);

return 0;
}


