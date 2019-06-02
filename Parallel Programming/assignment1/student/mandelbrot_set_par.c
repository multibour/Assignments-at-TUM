#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>

#include "mandelbrot_set.h"


// the struct for input to the threads
struct Input {
	int x_resolution;
	int y_end;
	int max_iter;
	double view_x0; 
	double view_x1; 
	double view_y0; 
	double view_y1;
	double x_stepsize; 
	double y_stepsize;
	int palette_shift;
	unsigned char (*image)[][3];

	int y_start;
};

// the thread function
void* drawer_thread(void* args){
	struct Input *input;
	input = (struct Input*) args;

	double y;
	double x;

	complex double Z;
	complex double C;

	int k;

	// create a local variable to define the boundaries of the array
	unsigned char (*img)[input->x_resolution][3] = input->image;

	// same operations as sequential
	for (int i = input->y_start; i < input->y_end; ++i)
	{
		for (int j = 0; j < input->x_resolution; ++j)
		{
			y = input->view_y1 - i * input->y_stepsize;
			x = input->view_x0 + j * input->x_stepsize;
 
			Z = 0 + 0 * I;
			C = x + y * I;

			k = 0;

			do
			{
				Z = Z * Z + C;
				k++;
			} while (cabs(Z) < 2 && k < input->max_iter);

			if (k == input->max_iter)
			{
				memcpy(img[i][j], "\0\0\0", 3);
			}
			else
			{
				int index = (k + input->palette_shift)
				            % (sizeof(colors) / sizeof(colors[0]));
				memcpy(img[i][j], colors[index], 3);
			}
		}
	}
	return NULL;
}


void mandelbrot_draw(int x_resolution, int y_resolution, int max_iter,
	                double view_x0, double view_x1, double view_y0, double view_y1,
	                double x_stepsize, double y_stepsize,
	                int palette_shift, unsigned char (*image)[x_resolution][3],
						 int num_threads) {
	// TODO:
	// implement your solution in this file.
	int i;
	struct Input *inputs = (struct Input*) malloc(num_threads * sizeof(struct Input)); // array of inputs for threads
	pthread_t *threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t)); // array of threads
	
	// create inputs
	for (i = 0; i < num_threads; ++i){
		struct Input* temp;
		temp = &inputs[i];
		temp->x_resolution = x_resolution;

		temp->y_end = (i + 1) * (y_resolution / num_threads);
		if (i == num_threads - 1){ // if it is the last thread
			temp->y_end += (y_resolution % num_threads); // add the remainder so the whole image is considered
		}

		temp->max_iter = max_iter;
		temp->view_x0 = view_x0;
		temp->view_x1 = view_x1;
		temp->view_y0 = view_y0;
		temp->view_y1 = view_y1;
		temp->x_stepsize = x_stepsize;
		temp->y_stepsize = y_stepsize;
		temp->palette_shift = palette_shift;
		temp->image = image;

		temp->y_start = i * (y_resolution / num_threads);
	}
	
	// run threads
	for (i = 0; i < num_threads; ++i){
		pthread_create(&threads[i], NULL, drawer_thread, &inputs[i]);
	}
	
	// wait for threads
	for (i = 0; i < num_threads; ++i){
		pthread_join(threads[i], NULL);
	}

	// free memory
	free(inputs);
	free(threads);
}
