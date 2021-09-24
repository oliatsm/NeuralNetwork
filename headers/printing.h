#ifndef HEADER_FILE_NAME
#define HEADER_FILE_NAME

#include <stdio.h>

#define PRINT_STEP 1


//*********************************************************
void PrintArray(int n,double a[]){

    for (int i=0;i<n;i+=PRINT_STEP)
            printf("%lf ",a[i]);
        printf("\n");
}

void Print2dArray(int n,int m,double a[][m]){

    for (int i=0;i<n;i+=PRINT_STEP){
        for (int j=0;j<m;j+=PRINT_STEP)
            printf("%lf ",a[i][j]);
        printf("\n");
        }
}


#endif
