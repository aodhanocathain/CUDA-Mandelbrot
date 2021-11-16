#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include <iostream>

#define WIDTH 80
#define HEIGHT 45
#define SPAN 4 //length of real axis visible on screen
#define MAX_ITERATIONS 100000
#define IN_SET '0'
#define NOT_IN_SET ' '

void __global__ iterate(char* points)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = index % WIDTH;
	int y = index / WIDTH;

	//value = (axis scale) * (leftmost value + portion of axis covered)
	float real = (SPAN) * (-(1/2) + (x/(float)WIDTH));
	//value = (axis scale) * (top value + portion of axis covered)
	float imaginary = (WIDTH / (float)HEIGHT) * ((1 / 2) - (y / (float)HEIGHT));

	float realCopy; bool escape;
	for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++)
	{
		realCopy = real;

		/* intended algorithm
		real = (real * real) - (imaginary * imaginary);
		imaginary = 2 * realCopy * imaginary;
		if(absolute value of real or imaginary exceeds 2){return;}
		*/

		//must use branchless code to avoid divergence
		escape = real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2;
		real = 2 * (escape)+((real * real) - (imaginary * imaginary)) * (!escape);
		imaginary = 2 * (escape)+(2 * realCopy * imaginary) * (!escape);
	}

	escape = real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2;
	points[index] = (IN_SET * !escape) + (NOT_IN_SET * escape);
}

int main()
{
	using namespace std;

	char* host_points = (char*)malloc(sizeof(char)*HEIGHT*WIDTH);
	char* dev_points = nullptr;

	if (cudaMalloc(&dev_points, sizeof(char)*HEIGHT*WIDTH))
	{
		cout << "Could not allocate memory on the device";
		return(-1);
	}

	iterate<<<HEIGHT, WIDTH>>>(dev_points);

	if (cudaMemcpy(host_points, dev_points, sizeof(char)*HEIGHT*WIDTH, cudaMemcpyDeviceToHost))
	{
		cout << "Could not copy memory from the device :(";
		return(-1);
	}

	return 0;
}