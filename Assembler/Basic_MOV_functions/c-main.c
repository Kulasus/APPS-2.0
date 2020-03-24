
#include <stdio.h>

#define numArrayLength 10
#define longArrayLength 6
#define charArrayLength 32

int g_val0 = -2;
int g_val1 = -10;
int g_valSet = 0; // OLD
int g_index = 0;
long g_pole_long[longArrayLength] = {0,0,0,0,0,0};
long g_pole_long2[longArrayLength] = {0,0,0,0,0,0};
char g_pozdrav[ charArrayLength ] = "!_Ahoj!";
long g_long = 0xFEDCBA9876543210;
int g_lsb = 0, g_msb = 0;

void setLongArray(); //NEW
void setLongArrayNoExtension(); //NEW
void setLongOnIndex(); //OLD
void setLongOnIndexNoExtension(); //OLD
void moveCharArrayLeft(); //NEW
void setCharOnIndex(); // OLD
void setMsbLsb();
/* OLD
void setValuesInLongArray(void (*function)()){
    for (int i = 0; i < longArrayLength - 1; i++)
    {
        g_index = i;
        g_valSet = i%2 == 0 ? g_val0 : g_val1;
        function(); 
    }
    g_index = 0;
}

void moveCharArrayLeft(){
    for (int i = 0; i < charArrayLength; i++)
    {
        g_index = i;
        setCharOnIndex();   
    }
    g_index = 0;
};
*/
void print_array_long(long *array){
    for (int i = 0; i < longArrayLength; i++)
    {
        printf("%ld", array[i]);
        printf(",");
    }
    printf("\n");
}

int main()
{
    //1
    printf("1.\n");
    print_array_long(g_pole_long);
    setLongArray();
    print_array_long(g_pole_long);
    printf("-------------------------\n2.\n");

    //2
    print_array_long(g_pole_long2);
    setLongArrayNoExtension();
    print_array_long(g_pole_long2);
    printf("-------------------------\n3.\n");

    //3
    printf("g_pozdrav '%s'\n",g_pozdrav);
    moveCharArrayLeft();
    printf("g_pozdrav '%s'\n",g_pozdrav);
    printf("-------------------------\n5.\n");

    //5
    printf( "long %016lX, msb %08X, lsb %08X\n", g_long, g_msb, g_lsb );
    setMsbLsb();
    printf( "long %016lX, msb %08X, lsb %08X\n", g_long, g_msb, g_lsb );
}