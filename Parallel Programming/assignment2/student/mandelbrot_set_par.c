#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>

#include "mandelbrot_set.h"

//-- Structs --//

typedef struct {
	int x_resolution;
	int y_resolution;
	int max_iter;
	double view_x0; 
	double view_x1; 
	double view_y0; 
	double view_y1;
	double x_stepsize; 
	double y_stepsize;
	int palette_shift;
	unsigned char (*image)[][3];	
} Input;

typedef struct{
	int y_start;
	int y_end;
} Chunk;

typedef struct{
	Chunk* buffer;
	int index;
	unsigned int size;
} ChunkBuffer;


//-- Common Variables --//

pthread_mutex_t bufferLock = PTHREAD_MUTEX_INITIALIZER;
ChunkBuffer chunkBuffer;

pthread_mutex_t checkMutex = PTHREAD_MUTEX_INITIALIZER;
int allChunksGenerated = 0;


//-- Thread Definitions --//

void* producer_thread(void* args){

	//intialize stuff
	Input* input = (Input*) args;

	const int chunkSize = 10;
	const int chunkCount = (input->y_resolution / chunkSize) + (input->y_resolution % chunkSize != 0 ? 1 : 0);

	chunkBuffer.buffer = (Chunk*) calloc(chunkCount, sizeof(Chunk));
	chunkBuffer.size = 0;
	chunkBuffer.index = 0;

	// create chunks
	for (int i = 0; i < chunkCount; ++i){
		pthread_mutex_lock(&bufferLock);

		chunkBuffer.buffer[i].y_start = i * chunkSize;
		chunkBuffer.buffer[i].y_end = (i + 1) * chunkSize;
		if (i == chunkCount - 1){
			chunkBuffer.buffer[i].y_end = input->y_resolution;
		}

		chunkBuffer.size++;

		pthread_mutex_unlock(&bufferLock);
	}

	pthread_mutex_lock(&checkMutex);
	allChunksGenerated = 1;
	pthread_mutex_unlock(&checkMutex);

	return NULL;
}


void* worker_thread(void* args){
	Input* input = (Input*) args;

	// create a local variable to define the boundaries of the array
	unsigned char (*img)[input->x_resolution][3] = input->image;

	// variables for processing
	double y;
	double x;
	complex double Z;
	complex double C;
	int k, y_start, y_end;

	while (1){
		// try to get a chunk off the buffer
		pthread_mutex_lock(&bufferLock);
		if (chunkBuffer.index >= chunkBuffer.size){

			pthread_mutex_lock(&checkMutex);
			if (allChunksGenerated){
				pthread_mutex_unlock(&checkMutex);
				pthread_mutex_unlock(&bufferLock);
				break; // break out of infinite while loop
			}
			pthread_mutex_unlock(&checkMutex);

			pthread_mutex_unlock(&bufferLock);
			continue; // try again checking whether there are available chunks
		}
		else{ // chunkBuffer.index < chunkBuffer.size
			Chunk chunk = chunkBuffer.buffer[chunkBuffer.index];
			y_start = chunk.y_start;
			y_end = chunk.y_end;

			chunkBuffer.index++;
		}
		pthread_mutex_unlock(&bufferLock);


		// start processing pixels in the chunk
		for (int i = y_start; i < y_end; ++i){
			for (int j = 0; j < input->x_resolution; ++j){
				y = input->view_y1 - i * input->y_stepsize;
				x = input->view_x0 + j * input->x_stepsize;
	
				Z = 0 + 0 * I;
				C = x + y * I;

				k = 0;

				do{
					Z = Z * Z + C;
					k++;
				} while (cabs(Z) < 2 && k < input->max_iter);

				if (k == input->max_iter){
					memcpy(img[i][j], "\0\0\0", 3);
				}
				else{
					int index = (k + input->palette_shift)
								% (sizeof(colors) / sizeof(colors[0]));
					memcpy(img[i][j], colors[index], 3);
				}
			}
		} // end of for loops

	} //end of infinite while loop

	return NULL;
}


//-- Main --//

void mandelbrot_draw(int x_resolution, int y_resolution, int max_iter,
			        double view_x0, double view_x1, double view_y0, double view_y1,
					double x_stepsize, double y_stepsize,
					int palette_shift, unsigned char (*image)[x_resolution][3],
					int num_threads) {
	// TODO: implement your solution in this file.

	// create input struct
	Input input;
	input.x_resolution = x_resolution;
	input.y_resolution = y_resolution;
	input.max_iter = max_iter;
	input.view_x0 = view_x0;
	input.view_x1 = view_x1;
	input.view_y0 = view_y0;
	input.view_y1 = view_y1;
	input.x_stepsize = x_stepsize;
	input.y_stepsize = y_stepsize;
	input.palette_shift = palette_shift;
	input.image = image;

	// initialize threads
	num_threads--;
	pthread_t* threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
	pthread_t producer;

	// run threads
	pthread_create(&producer, NULL, producer_thread, &input);
	for (int i = 0; i < num_threads; ++i){
		pthread_create(&threads[i], NULL, worker_thread, &input);
	}

	// wait for threads
	pthread_join(producer, NULL);
	for (int i = 0; i < num_threads; ++i){
		pthread_join(threads[i], NULL);
	}

	free(threads);
	free(chunkBuffer.buffer); // Bad programming practice! Always deallocate in the same scope where it was allocated!
}
