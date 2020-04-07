#include <stdio.h>


int kopiruj_velka(char * l_vysledek,char * l_jmeno, char nahrazovac);
long bitova_maska(char * t_cisla_bitu, int t_LEN);
int over_format(char * retezec);
void najdi_pozice_minmax(long *t_prvky, int t_L, long *t_indexy_minmax);

int main()
{
    printf("----START----\n");
    printf("Ukol 3.\n");
    char l_jmeno[] = "Ten Co Pase Cerne Ovce V Hustem Lese";
    char l_vysledek[ 1024 ];
    int l_preneseno = kopiruj_velka(l_vysledek, l_jmeno, '-');
    printf("Veta: %s\n",l_jmeno);
    printf("Vysledek: %s\n",l_vysledek);
    printf("Preneseno: %d\n", l_preneseno);

    //------------------------------------------------------------------

    printf("---------------\n");
    printf("Ukol 1.\n");
    char l_cisla_bitu[] = {7, 0};
    int l_LEN = 2;
    long cislo = bitova_maska(l_cisla_bitu, l_LEN);
    printf("Cislo v dekadicke soustave je: %ld\n",cislo);
    printf("Cislo v hexadecimalni soustave je: %lx\n",cislo);

    //------------------------------------------------------------------

    printf("---------------\n");
    printf("Ukol 4.\n");
    char cislo2[] = "-9_999_99.0000";
    int format = over_format(cislo2);
    printf("Cislo je: %s\n",cislo2);
    printf("Formaty: 1 - dobry format, 0 - spatny format\n");
    printf("Format: %d\n",format);


    //------------------------------------------------------------------

    printf("---------------\n");
    printf("Ukol 2.\n");
    long l_prvky[] = {333, 5, 55, 0x1};
    int l_l = 4;
    long l_index_minmax[ 2 ];
    najdi_pozice_minmax(l_prvky, l_l, l_index_minmax);
    printf("Maximum: %ld\n",l_index_minmax[0]);
    printf("Minimum: %ld\n",l_index_minmax[1]);
}