#include "pch.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

int malloc2dint(int ***matrix, int n) {

    // allocate the n*m contiguous items
    int *aux = (int *)malloc(n*n * sizeof(int));
    if (!aux) return -1;

    // allocate the row pointers into the memory
    (*matrix) = (int **)malloc(n * sizeof(int*));
    if (!(*matrix)) {
        free(aux);
        return -1;
    }

    // set up the pointers into the contiguous memory
    for (int i = 0; i < n; i++)
        (*matrix)[i] = &(aux[i*n]);

    return 0;
}

void prettyprint(int ***matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", (*matrix)[i][j]);
        }
        std::cout << std::endl;
    }
}

int free2dint(int ***array) {
    // free the memory - the first element of the array is at the start 
    free(&((*array)[0][0]));

    // free the pointers into the memory 
    free(*array);

    return 0;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int i, j, k;
    int **A = NULL;
    int **B = NULL;
    int **C = NULL;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    if (world_rank == 0) {
        t1 = std::chrono::high_resolution_clock::now();
    }

    if (argc != 2) {
        if (world_rank == 0) {
            printf("Usage: %s infilename\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    char* filename = argv[1];

    //works with world sizes of perfect squares
    const int p = sqrt(world_size);
    int n = 0;
    int n_block;

    if (p*p != world_size) {
        printf("World size must be a perfect square!\n");
        MPI_Finalize();
        exit(1);
    }

    ///READ MATRICES FROM FILE///
    if (world_rank == 0)
    {
        std::ifstream infile;
        infile.open(filename);

        //first line has to contain n=rows=columns
        infile >> n;
        if (n == 0) {
            printf("Matrix is empty!\n");
            MPI_Finalize();
            exit(1);
        }

        if (n % p != 0) {
            printf("Number of matrix elements is not divisible by p!\n");
            MPI_Finalize();
            exit(1);
        }

        malloc2dint(&A, n);
        malloc2dint(&B, n);
        malloc2dint(&C, n);

        //file must have two nxn matrices one after another
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                infile >> A[i][j];
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                infile >> B[i][j];
        }

        infile.close();

        n_block = n / p;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //blocks
    int **a_temp, **b_temp, **c_temp;
    malloc2dint(&a_temp, n_block);
    malloc2dint(&b_temp, n_block);
    malloc2dint(&c_temp, n_block);

    for (i = 0; i < n_block; i++) {
        for (j = 0; j < n_block; j++) {
            c_temp[i][j] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ///SEND BLOCKS TO PROCESSORS///
    //global size
    int sizes[2] = { n, n };
    //local size
    int subsizes[2] = { n_block, n_block };
    //first start
    int starts[2] = { 0, 0 };
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *pA = NULL;
    int *pB = NULL;
    int *pC = NULL;
    if (world_rank == 0) {
        pA = &(A[0][0]);
        pB = &(B[0][0]);
        pC = &(C[0][0]);
    }

    //define the displacement and nr to send for scatter
    int* sendcounts;
    sendcounts = (int*)malloc(p*p * sizeof(int));
    int* displacements;
    displacements = (int*)malloc(p*p * sizeof(int));

    if (world_rank == 0) {
        for (i = 0; i < p*p; i++) {
            sendcounts[i] = 1;
        }
        int disp = 0;
        for (i = 0; i < p; i++) {
            for (j = 0; j < p; j++) {
                displacements[i*p + j] = disp;
                //next block on the row starts at current+n_block
                disp += n_block;
            }
            //next block on the next row starts in n_block-1 rows
            disp += (n_block-1)*n;
        }
    }

    /*int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        MPI_Datatype recvtype,
        int root, MPI_Comm comm)*/
    MPI_Scatterv(pA, sendcounts, displacements,
        subarrtype, &(a_temp[0][0]), n_block*n_block,
        MPI_INT,
        0, MPI_COMM_WORLD);
    MPI_Scatterv(pB, sendcounts, displacements,
        subarrtype, &(b_temp[0][0]), n_block*n_block,
        MPI_INT,
        0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int n_left, n_right;
    int n_up, n_down;

    ///ORGANIZE WORLD///
    if (world_rank == 0)
    {
        int** processors;
        processors = (int**)malloc(p * sizeof(int*));
        for (i = 0; i < p; i++) {
            processors[i] = (int*)malloc(p * sizeof(int));
        }

        //put the processor ranks in a matrix
        for (i = 0; i < p; i++) {
            for (j = 0; j < p; j++) {
                processors[i][j] = i * p + j;
            }
        }

        if (processors[p - 1][p - 1] != world_size - 1) {
            printf("World size wasn't evenly distributed!\n");
            MPI_Finalize();
            exit(1);
        }

        int n_left_temp, n_right_temp, n_up_temp, n_down_temp;

        //Send to all processors their neighbor numbers
        for (i = 0; i < p; i++) {
            for (j = 0; j < p; j++) {

                //left neighbor
                if (j > 0) {
                    n_left = processors[i][j - 1];
                }
                else {
                    n_left = processors[i][p - 1];
                }
                if (processors[i][j] != 0) {
                    MPI_Send(&n_left, 1, MPI_INT, processors[i][j], 0, MPI_COMM_WORLD);
                }
                else {
                    n_left_temp = n_left;
                }

                //right neighbor
                if (j < p - 1) {
                    n_right = processors[i][j + 1];
                }
                else {
                    n_right = processors[i][0];
                }
                if (processors[i][j] != 0) {
                    MPI_Send(&n_right, 1, MPI_INT, processors[i][j], 0, MPI_COMM_WORLD);
                }
                else {
                    n_right_temp = n_right;
                }

                //up neighbor
                if (i > 0) {
                    n_up = processors[i - 1][j];
                }
                else {
                    n_up = processors[p - 1][j];
                }
                if (processors[i][j] != 0) {
                    MPI_Send(&n_up, 1, MPI_INT, processors[i][j], 0, MPI_COMM_WORLD);
                }
                else {
                    n_up_temp = n_up;
                }

                //down neighbor
                if (i < p - 1) {
                    n_down = processors[i + 1][j];
                }
                else {
                    n_down = processors[0][j];
                }
                if (processors[i][j] != 0) {
                    MPI_Send(&n_down, 1, MPI_INT, processors[i][j], 0, MPI_COMM_WORLD);
                }
                else {
                    n_down_temp = n_down;
                }
            }
        }
        n_left = n_left_temp;
        n_right = n_right_temp;
        n_up = n_up_temp;
        n_down = n_down_temp;
    }
    else {
        MPI_Recv(&n_left, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_right, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_up, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_down, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ///MULTIPLY BLOCKS AND SEND-RECEIVE
    int nr_op = 0;
    while(nr_op < p) {
        //multiply
        for (i = 0; i < n_block; i++) {
            for (j = 0; j < n_block; j++) {
                for (k = 0; k < n_block; k++)
                    c_temp[i][j] = c_temp[i][j] + a_temp[i][k] * b_temp[k][j];
            }
        }

        //send further
        //A to left
        MPI_Send(a_temp[0], n_block*n_block, MPI_INT, n_left, 0, MPI_COMM_WORLD);
        //B to up
        MPI_Send(b_temp[0], n_block*n_block, MPI_INT, n_up, 0, MPI_COMM_WORLD);

        //receive
        //A from right
        MPI_Recv(a_temp[0], n_block*n_block, MPI_INT, n_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //B from down
        MPI_Recv(b_temp[0], n_block*n_block, MPI_INT, n_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        nr_op++;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ///GATHER ALL SUBRESULTS///
    /*int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)*/
    MPI_Gatherv(&(c_temp[0][0]), n_block*n_block, MPI_INT,
        pC, sendcounts, displacements, subarrtype,
        0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    ///PRINT RESULT///
    if (world_rank == 0) {
        std::cout << "Matrix A: " << std::endl;
        prettyprint(&A, n);
        std::cout << "Matrix B: " << std::endl;
        prettyprint(&B, n);

        std::cout << "Result matrix C: " << std::endl;
        prettyprint(&C, n);

        free2dint(&A);
        free2dint(&B);
        free2dint(&C);
    }

    if (world_rank == 0) {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        std::cout << "It took me " << time_span.count() << " seconds.";
    }
       
    MPI_Barrier(MPI_COMM_WORLD);

    free2dint(&a_temp);
    free2dint(&b_temp);
    free2dint(&c_temp);

    MPI_Finalize();
}