#include <stdio.h>

int mul_2char(char a, char b);
long mul_2int(int a, int b);
int skalarni(int *a, int *b, int N);
int div_2int(int a, int b, int *zbytek);
int nasobky(int *pole, int N, int cislo);
void minmax(int *pole, int N, int *min, int *max);
int is_digit(char c, char hex);
long faktorial(long n);
char vyskyt(char *str);


int main(){

    //vyskyt
    //printf("vyskyt: %c\n", vyskyt("aqzgweerjioenfjnedfijndjvfjnsdjf"));

    //faktorial
    //printf("faktorial: %ld\n", faktorial(10));

    //is_digit
    /*printf("is_digit(8): %d\n",is_digit('8',0));
    printf("is_digit(A): %d\n",is_digit('A',0));
    printf("is_digit(A): %d\n",is_digit('A',1));*/

    //minmax
    /*int min;
    int max;
    int pole[5] = {-1,5,0,-20,1};
    minmax(pole, 5, &min, &max);
    printf("minmax: min: %d | max: %d\n", min,max);*/

    //nasobky
    /*int pole[5] = {2,4,5,6,11};
    printf("nasobky 2: %d\n",nasobky(pole, 5, 2));
    printf("nasobky 3: %d\n",nasobky(pole, 5, 3));
    printf("nasobky 10: %d\n",nasobky(pole, 5, 10));*/

    //div_2int
    /*int zbytek = 0;
    printf("deleni: zbytek: %d, vysledek: %d\n", zbytek, div_2int(123456789,1000, &zbytek));*/

    //skalarni
    /*int vec1[5] = {1, 2, 3, 4, 5};
    int vec2[5] = {5, 4, 3, 2, 1};
    printf("skalarni: %d\n",skalarni(vec1,vec2,5));*/

    //mul_2int
    //printf("mul int: %ld\n",mul_2int(10000000,6666666));

    //mul_2char
    //printf("mul char: %d\n",mul_2char(-100,-100));
}