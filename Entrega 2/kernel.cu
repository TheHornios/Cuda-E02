/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Entrega 2
*
* Alumno: Rodrigo Pascual Arnaiz y Villar Solla, Alejandro
* Fecha: 16/11/2022
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"

// estrutura de Color, por no declarar las partes de un color en un array
struct Color {
	float R, G, B, A;
};

// Defines
#define LADO_CUADRADO 512 // Dimension vertical
#define REJILLA_TAMNYO 8 // Bloques a mostrar
#define DIM ( LADO_CUADRADO / REJILLA_TAMNYO ) // Dimension de los bloques
#define BLANCO Color { 255, 255, 255, 0 } // Color BLANCO
#define NEGRO Color { 0, 0, 0, 0 } // Color NEGRO


// GLOBAL: funcion llamada desde el host y ejecutada en el device (generateImage)
/**
* Funcion: generateImage ( GLOBAL )
* Objetivo: Funcion que genera la rejilla de colores blancos y negros 
* La rejilla es de un 8x8
*
* Param: char* imagen -> Bitmap del device
* Return: void
*/

__global__ void generateImage(unsigned char* imagen)
{	
	Color color;

	// coordenada vertical de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada horizontal de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada global de cada hilo (indice para acceder a la memoria)
	int posicion = x + y * blockDim.x * gridDim.x;
	// cada hilo obtiene la posicion de un pixel
	int pixel = posicion * 4;

	// Obtenemos la posición teniendo encuenta la dimension de cada cuadrado
	// dimension_factor = LADO_CUADRADO / REJILLA_TAMNYO 
	// 512 / 8 = 64   |  dimension_factor = 64
	// posX = CEIL( Posicion X  / dimension_factor )
	// posY = CEIL( Posicion Y  / dimension_factor )
	int posX = (int) ceil( (float)x / DIM );
	int posY = (int) ceil( (float)y / DIM );

	// Sabemos el color según X e Y
	// Si Y es par es = 1, si Y es impar es 1
	// Si X + (Y % 2 ? 1 : 0)
	// Par colorea en BLANCO y Impar en NEGRO.
	if ( (posX + (posY % 2)) % 2 ) {
		color = BLANCO;
	}
	else {
		color = NEGRO;
	}

	// Como son el mismo color, recorremos los tres canales (R,G,B)
	imagen[pixel + 0] = color.R; 
	imagen[pixel + 1] = color.G; 
	imagen[pixel + 2] = color.B; 
	imagen[pixel + 3] = color.A; 


}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Declaracion del bitmap:
	// Inicializacion de la estructura RenderGPU
	RenderGPU foto(LADO_CUADRADO, LADO_CUADRADO);
	// Tamaño del bitmap en bytes
	size_t size = foto.image_size();
	// Asignacion y reserva de la memoria en el host (framebuffer) 
	unsigned char* host_bitmap = foto.get_ptr();
	// Reserva en el device 
	unsigned char* dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, size);
	// Generamos el bitmap:
	// Lanzamos un kernel con bloques de 256 hilos (16x16)
	// y tantos bloques como hagan falta
	dim3 hilosB(16, 16);
	// Calculamos el numero de bloques necesario (un hilo por cada pixel)
	dim3 Nbloques(LADO_CUADRADO / 16, LADO_CUADRADO / 16);
	// Generamos el bitmap 
	generateImage <<<Nbloques, hilosB >>> (dev_bitmap);
	// Recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
	// Visualizacion y salida
	printf("\n...pulsa ESC para finalizar...");
	foto.display_and_exit();
	return 0;
}
///////////////////////////////////////////////////////////////////////////
