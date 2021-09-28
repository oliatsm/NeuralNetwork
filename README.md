# NeuralNetwork
Error Back Propagation
---

### Αρχεία:
**nn1-5.c** 	: νευρωνικό δίκτυο με τυχαίες τιμές εισόδου- εξόδου  
**nn_xor-3.c** 	: νευρωνικό δίκτυο πύλης XOR  
**nn_fmnist-3.c**	:νευρωνικό δίκτυο με τιμές εισόδου-εξόδου fashion mnist  
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
Από το σύνολο των δεδομένων που χρησιμοποιώ για test, βρίσκω την κατηγορία που ανήκει το κάθε παράδειγμα υπολογίζοντας τον νευρώνα εξόδου (estimated) με τη μεγαλύτερη τιμή και συγκρίνω αν είναι ο ίδιος νευρώνας με την κατηγορία που ανήκει πραγματικά το παραδειγμα (category). 


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

## Αποτελέσματα κώδικα με δεδομένα fashion mnist, για 1000 επαναλήψεις:

```
$ gcc -o fmnist nn_fmnist-2.c -fopenmp -lm
$ time ./fmnist 

X[60000][784], L1: 100, L2: 10,  epoch: 1000
Initialise fmnist data,W :  0.682202s
     MSE: 0.027583, time : 5.143038s
 100 MSE: 0.013489, time : 8.138292s
 200 MSE: 0.011950, time : 5.967105s
 300 MSE: 0.009765, time : 5.306875s
 400 MSE: 0.008898, time : 5.102918s
 500 MSE: 0.009068, time : 5.110120s
 600 MSE: 0.007795, time : 5.141652s
 700 MSE: 0.007475, time : 5.126513s
 800 MSE: 0.007152, time : 5.131338s
 900 MSE: 0.007161, time : 5.966761s

Test
Test MSE: 0.173984
Test Error: 1338 of 10000 (13.4 %)
    time : 0.475545s
Total time : 5602.718424s

real    93m22,764s
user    92m59,054s
sys        0m9,103s

```

---

