#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <chrono>

#include <iostream>
#include <ctime>

#define WIDTH 107
#define HEIGHT 60
#define SPAN 3 //length of real axis visible on screen
#define MAX_ITERATIONS 1000000
#define IN_SET '0'
#define NOT_IN_SET ' '
#define ESCAPED_VALUE 3

char* globalPoints;

void __global__ iterate(char* points)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = index % WIDTH;
	int y = index / WIDTH;

	//value = (axis scale) * (leftmost value + portion of axis covered)
	float real = (SPAN) * (-(1/(float)2) + (x/(float)WIDTH));
	//value = (axis scale) * (top value + portion of axis covered)
	float imaginary = (SPAN * WIDTH / (float)HEIGHT) * ((1 / (float)2) - (y / (float)HEIGHT));

	float realCopy; bool escape; 
	float addend_real = real, addend_imaginary = imaginary;
	for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++)
	{
		realCopy = real;

		/* intended algorithm
		real = (real * real) - (imaginary * imaginary) + original real;
		imaginary = 2 * realCopy * imaginary + original imaginary;
		if(absolute value of real or imaginary exceeds 2){return;}
		*/

		//must use branchless code to avoid divergence
		escape = real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2;
		real = (ESCAPED_VALUE*(escape)) + ( (((real * real) - (imaginary * imaginary)) + addend_real) * (!escape));
		imaginary = (ESCAPED_VALUE*(escape)) + ( ((2 * realCopy * imaginary) + addend_imaginary) * (!escape));
	}

	escape = real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2;
	points[index] = (IN_SET * !escape) + (NOT_IN_SET * escape);
}

long long time_device()
{

	using namespace std;
	char* host_points = (char*)malloc(sizeof(char) * HEIGHT * WIDTH);
	char* dev_points = nullptr;

	auto start = chrono::steady_clock::now();

	if (cudaMalloc(&dev_points, sizeof(char) * HEIGHT * WIDTH))
	{
		cout << "Could not allocate memory on the device";
		return(-1);
	}

	iterate << <HEIGHT, WIDTH >> > (dev_points);

	cudaDeviceSynchronize();

	if (cudaMemcpy(host_points, dev_points, sizeof(char) * HEIGHT * WIDTH, cudaMemcpyDeviceToHost))
	{
		cout << "Could not copy memory from the device :(";
		return(-1);
	}

	if (cudaFree(dev_points))
	{
		cout << "Could not free memory on the device :(";
		return(-1);
	}
	auto end = chrono::steady_clock::now();
	globalPoints = host_points;
	return (long long)(chrono::duration_cast<chrono::milliseconds>(end - start).count());
}

long long time_host()
{
	using namespace std;
	char* points = (char*)malloc(sizeof(char) * HEIGHT * WIDTH);

	auto start = chrono::steady_clock::now();

	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			//value = (axis scale) * (leftmost value + portion of axis covered)
			float real = (SPAN) * (-(1 / (float)2) + (x / (float)WIDTH));
			//value = (axis scale) * (top value + portion of axis covered)
			float imaginary = (SPAN * WIDTH / (float)HEIGHT) * ((1 / (float)2) - (y / (float)HEIGHT));

			float realCopy;
			float addend_real = real, addend_imaginary = imaginary;
			for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++)
			{
				realCopy = real;

				real = (real * real) - (imaginary * imaginary) + addend_real;
				imaginary = 2 * realCopy * imaginary + addend_imaginary;
				if(real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2){break;}
			}

			bool escape = real <= -2 || real >= 2 || imaginary <= -2 || imaginary >= 2;
			points[y*WIDTH + x] = (IN_SET * !escape) + (NOT_IN_SET * escape);
		}
	}

	free(points);

	auto end = chrono::steady_clock::now();
	return (long long)(chrono::duration_cast<chrono::milliseconds>(end - start).count());
}

int main()
{
	using namespace std;
	cout << "Timing the host..." << endl;
	float host_time = time_host()/(double)1000;
	cout << "Timing the device..." << endl;
	float device_time = time_device()/(double)1000;
	for (int row = 0; row < HEIGHT; row++)
	{
		for (int column = 0; column < WIDTH; column++)
		{
			cout << globalPoints[(row * WIDTH) + column];
		}
		cout << endl;
	}
	printf("real axis span: %d, width: %d, height: %d, max iterations: %d\n", SPAN, WIDTH, HEIGHT, MAX_ITERATIONS);
	cout << "Rendering in parallel on the graphics card takes " << device_time << " seconds." << endl;
	cout << "Rendering on one CPU thread takes " << host_time << " seconds." << endl;
	return 0;
}