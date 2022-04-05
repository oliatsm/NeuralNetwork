# NeuralNetwork
Error Back Propagation
---

### Αρχεία:
**nn1-5.c** 	: νευρωνικό δίκτυο με τυχαίες τιμές εισόδου- εξόδου  
**nn_xor-3.c** 	: νευρωνικό δίκτυο πύλης XOR  
**nn_fmnist-5.c**	:νευρωνικό δίκτυο με τιμές εισόδου-εξόδου fashion mnist  
**mnist.h**		: συναρτήσεις για εισαγωγή των δεδομένων από τα αρχεία του fmnist  
**printing.h**	: συναρτήσεις για εκτύπωση πινάκων
* PrintArray(int n,double a[]): για μονοδιάστατο πίνακα a[m]
* Print2dArray(int n,int m,double a[][m]): για δισδιάστατο πίνακα a[n][m]

---

## Υπολογισμός Mean Squared Error – Συνάρτηση MSE()
Για τον υπολογισμό του μέσου τετραγωνικού σφάλματος, αφού υπολογίσω το μέσω τετραγωνικό σφάλμα για κάθε παράδειγμα ξεχωριστά,  υπολογίζω το άθροισμά τους στη μεταβλητή error και στη συνέχεια διαιρώ με τον αριθμό των παραδειγμάτων (μεταβλητή Μ) για να πάρω το τελικό MSE.


```c
int main(){
	//…
double error=0;
for(int d=0;d<M;d++){

            activateNN((double *)X[d]);
            trainNN((double *)X[d],(double *)Y[d]);
            error+=MSE((double*)Y[d]);
        }
            printf("%4.0d MSE: %lf , ",epoch,error/(M));
            printf("time : %lf\n",t);
        }
	//...
}

```

```c
double MSE(double *desired){

    double sum = 0;
    for(int n=0;n<NL2;n++){
        double dif=desired[n]-OL2[n];
        sum+=(dif*dif);
    }
    return sum/NL2;
   }

```
---

## Υπολογισμός Σφαλμάτων 
Από το σύνολο των δεδομένων που χρησιμοποιώ για test, βρίσκω την κατηγορία που ανήκει το κάθε παράδειγμα υπολογίζοντας τον νευρώνα εξόδου (**estimated**) με τη μεγαλύτερη τιμή και συγκρίνω αν είναι ο ίδιος νευρώνας με την κατηγορία που ανήκει πραγματικά το παραδειγμα (**category**). 


```c
int main(){
	//…
//Test
    int count =0;
    int estimated=0;
    int category =0;
    double err=0;
    for(int d=0;d<TEST;d++){
        activateNN((double *)X[d]);
        estimated=max_index(NL2,OL2);
        category=max_index(NL2,(double*)Y[d]);
        if(estimated==category) count++;
        err+=MSE((double*)Y[d]);
    }
    printf("Test MSE: %lf\n",err/TEST);
    printf("Test Correct: %d of %d \n",count,TEST);
	//…
}

```

---

## Αποτελέσματα κώδικα με δεδομένα fashion mnist, για 20 επαναλήψεις:

```
$ gcc nn_fmnist-5.c -o nn_fmnist -lm -fopenmp
$ time ./fmnist 

X[60000][784], L1: 100, L2: 10,  epoch: 20

Initialise fmnist data,W :  0.702933s

Training
     MSE: 0.046677, Acc: 68.281667, time : 38.386558s
   1 MSE: 0.029026, Acc: 81.086667, time : 40.930665s
   2 MSE: 0.025730, Acc: 83.178333, time : 39.162451s
   3 MSE: 0.024114, Acc: 84.215000, time : 37.570604s
   4 MSE: 0.023024, Acc: 84.978333, time : 37.321124s
   5 MSE: 0.022192, Acc: 85.581667, time : 36.369080s
   6 MSE: 0.021514, Acc: 85.971667, time : 36.452514s
   7 MSE: 0.020943, Acc: 86.373333, time : 36.322375s
   8 MSE: 0.020451, Acc: 86.680000, time : 36.347077s
   9 MSE: 0.020018, Acc: 86.958333, time : 36.297263s
  10 MSE: 0.019633, Acc: 87.190000, time : 37.876887s
  11 MSE: 0.019287, Acc: 87.421667, time : 38.476182s
  12 MSE: 0.018974, Acc: 87.588333, time : 37.036921s
  13 MSE: 0.018687, Acc: 87.793333, time : 36.514782s
  14 MSE: 0.018422, Acc: 87.941667, time : 36.377067s
  15 MSE: 0.018177, Acc: 88.105000, time : 36.308365s
  16 MSE: 0.017948, Acc: 88.216667, time : 37.814276s
  17 MSE: 0.017733, Acc: 88.346667, time : 37.328177s
  18 MSE: 0.017530, Acc: 88.460000, time : 37.479174s
  19 MSE: 0.017337, Acc: 88.585000, time : 37.176122s

Test
Test MSE: 0.159970
Test Acc: 8677 of 10000 (86.8 %)
    time : 2.777352s
Total time : 751.027950s

real    12m31,069s
user    12m29,741s
sys     0m0,612s

```

---

## Αποτελέσματα κώδικα XOR:
```
$ gcc nn_xor-3.c -o nn_xor-3 -lm -fopenmp 
$ time ./nn_xor-3 
XOR - NN

X[60000][2], L1: 100, L2: 10
Initialise X,Y,W :  0.005535s
     MSE: 0.050732 , time : 1.019057s
   1 MSE: 0.050422 , time : 0.949161s
   2 MSE: 0.050408 , time : 0.963353s
   3 MSE: 0.050399 , time : 0.944530s
   4 MSE: 0.050391 , time : 0.953095s
   5 MSE: 0.050383 , time : 0.957644s
   6 MSE: 0.050374 , time : 0.946143s
   7 MSE: 0.050359 , time : 0.947841s
   8 MSE: 0.050157 , time : 0.940041s
   9 MSE: 0.038376 , time : 0.948612s

Test
Test MSE: 0.020313
Test Acc: 10000 of 10000 (100.0 %)
    time : 0.072574s
Total time : 9.647587s

real    0m9,650s
user    0m9,609s
sys     0m0,028s
```
