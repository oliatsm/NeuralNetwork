CC = gcc
CFLAGS = -I headers/ -fopenmp -lm -Wall

fmnist:
	$(CC) nn_fmnist.c -o fmnist $(CFLAGS)

xor:
	$(CC) nn_xor.c -o xor $(CFLAGS)

random:
	$(CC) nn_random.c -o random $(CFLAGS)

clean:
	rm -f fmnist xor random
