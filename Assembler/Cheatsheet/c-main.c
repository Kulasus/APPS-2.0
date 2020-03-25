#include <stdio.h>

#define delkaPoleLong 6
#define delkaPoleChar 32

// Variables
int g_val0 = -5, g_val1 = -25; 
long g_pole_long1[ delkaPoleLong ] = { 0, 0, 0, 0, 0, 0 };
long g_pole_long2[ delkaPoleLong ] = { 0, 0, 0, 0, 0, 0 };
char g_pozdrav[ delkaPoleChar ] = "_Ahoj!";
long g_long = 0xFEDCBA0987654321;
int g_lsb = 0, g_msb = 0;

void uloha1();
void uloha2();
void uloha3();
void uloha5();

void vypis_pole_long(long *g_pole_long){
    for (int i = 0; i < 6; i++)
    {
        printf("%ld", g_pole_long[i]);
        printf(",");
    }
    printf("\n");
}

int main()
{
   uloha1();
   printf("Uloha1 :");
   vypis_pole_long(g_pole_long1);
   printf("\n");
   uloha2();
   printf("Uloha2 :");
   vypis_pole_long(g_pole_long2);
   printf("\n");
   printf("Uloha3: '%s'\n",g_pozdrav);
    uloha3();
   printf("Uloha3: '%s'\n",g_pozdrav);
   uloha5();
   printf("\n");
   printf("Uloha5 :");
   printf( "long %016lX, msb %08X, lsb %08X\n", g_long, g_msb, g_lsb );
   printf("\n");
}
