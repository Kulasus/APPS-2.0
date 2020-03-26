#include <stdio.h>

long g_long = 0x2121212141414141;
char g_pocet = 0;
int g_bin_num = 214711102;
char g_bin_num_char[33] = "000000000000000000000000000000000";
char g_date[] = "Dnes je 26. 3. 2020 09:35:02";
int g_pocet_cisel = 0;
int g_int_pole[256];
int g_login = 355;
int g_pocet4x0 = 0;
int g_pocet4x1 = 0;

void spocitej_bity();
void num_to_bin_char();
void spocitej_cisla();
void spocitej_jednicky_nuly();

void init_pole(){
    for(int i = 0; i < 256; i++)
    {
        g_int_pole[i] = i * ( g_login | 1 );
    }
}

int main()
{
    printf("--------------5----------------\n");
    num_to_bin_char();
    printf("cislo(dec): %d\n",g_bin_num);
    printf("cislo binarne: %s\n",g_bin_num_char);

    printf("--------------1.1----------------\n");
    spocitej_bity();
    printf("long(hex): %04lX\n", g_long);
    printf("long(dec): %ld\n", g_long);
    printf("pocet jednicek: %d\n",g_pocet);

    g_long = 0x0355035503550355;

    printf("--------------1.2----------------\n");
    spocitej_bity();
    printf("long muj login(0355)(hex): %04lX\n", g_long);
    printf("long muj login(0355)(dec): %ld\n", g_long);
    printf("pocet jednicek: %d\n",g_pocet);

    printf("--------------4----------------\n");
    spocitej_cisla();
    printf("datum: %s\n", g_date);
    printf("pocet cisel: %d\n", g_pocet_cisel);

    printf("--------------2----------------\n");
    init_pole();
    spocitej_jednicky_nuly();
    printf("pocet 4x0: %d\n", g_pocet4x0);
    printf("pocet 4x1: %d\n", g_pocet4x1);


}