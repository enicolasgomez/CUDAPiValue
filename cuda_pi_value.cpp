//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

//enicolasgomez@gmail.com

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <stdio.h>
#include <math.h>

#include <curand_kernel.h>
#include <ctime>

const int blocks  = 100 ; 
const int threads = 100 ; //each kernel will compute a 100 x 100 square 
                          //program will do that 10,000 times

__device__ int is_in_circle(int circle_x, int circle_y, int rad, int x, int y)
{
  int d = (pow((double)x - circle_x, 2) + pow((double)y - circle_y, 2));
  d = sqrt((double)d);
  return (int) d <= rad;
}

__global__ void sum_vector(int* total, int* result )
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if ( r > 0 )
    result[0] += result[r];
  __syncthreads();
  *total = result[0] ;
}
__global__ void compute_pi_parallel(int* result)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x; //linear index currently being processed (block[x] * block.length + block.threads[y])

  int local_square_size = 100 ;
  int initial_y = blockIdx.x  * local_square_size;
  int initial_x = threadIdx.x * local_square_size;

  //process a sub-square relative to the linear index
  for ( int x = initial_x ; x < initial_x + local_square_size; x ++ )
  {
    for ( int y = initial_y ; y < initial_y + local_square_size; y ++ )
    {
      result[r] += is_in_circle( 5000, 5000, 5000, x, y )  ;
    }
  }
}

int main()
{
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    const int array_length = blocks * threads ; //each array position will store the number of "in circle" matches for each x,y pair 

    int result_host[array_length] = {0};
    int *result_device ;

    printf("Calculating pi value in parallel\n");

    cudaMalloc( (void**)&result_device, sizeof(int)*array_length ); 

    compute_pi_parallel<<<blocks, threads>>>(result_device);

    cudaMemcpy( result_host, result_device, sizeof(int)*array_length, cudaMemcpyDeviceToHost );

    int total = 0;
    int *total_device = 0 ;
    cudaMalloc( (void**)&total_device, sizeof(int) ); 
    //TODO sum with CUDA using a shared variable
    sum_vector<<<blocks, threads>>>( total_device,  result_device );
    cudaMemcpy( &total, total_device, sizeof(int), cudaMemcpyDeviceToHost );
    // total = ( pi * r ^ 2 ) / ( ( 2 * r ) ^ 2 )
    // [total * ( ( 2 * r ) ^ 2 ) ] / ( r ^ 2 ) = pi

    float pi = (4 * total ) / (float)( pow(10000, 2) ) ;

    //double pi = ( total * ( pow( 2 * 100, 2) ) ) / ( pow(100, 2) ) / 10000 ;

    printf("Pi value: %f\n",  pi);

    cudaFree( &result_device );

    return 0;
}