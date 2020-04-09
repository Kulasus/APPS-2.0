#include <stdio.h>

long hledej_rozptyl(long *t_pole, int t_N);
void objem_valce(int t_R, int *t_vysledek, int t_Vyska);
int main(){

    // Tady bude mit nejvetsi rozptyl 9ka a 1cka, funkce vrati ale 9ku protoze je v poli na vetsim indexu
    long pole[] = {1,2,3,4,5,6,7,8,9};
    printf("prvek s nejvetsim rozptylem: %ld\n",hledej_rozptyl(pole,9));
    // Tady jsem to pro ukazku otocil
    long pole2[] = {9,2,3,4,5,6,7,8,1};
    printf("prvek s nejvetsim rozptylem: %ld\n",hledej_rozptyl(pole2,9));
    //----------------------------
    int l_vysledek[2];
    objem_valce(1,l_vysledek,3);
    printf( "Objem valce je %d.%03d\n", l_vysledek[ 0 ], l_vysledek[ 1 ] );
}