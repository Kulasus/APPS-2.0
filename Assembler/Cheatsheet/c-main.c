#include <stdio.h>

#define numArrayLength 10

// Variables
char g_char = 0;
int g_num32 = 0;
long g_num64 = 0;
int g_index = 2;

char g_array_char[128] = "Ok BOOMER! xdddddddd";
int g_array_num32[numArrayLength] = {0,1,2,3,4,5,6,7,8,9};
long g_array_num64[numArrayLength] = {0,1,2,3,4,5,6,7,8,9};

// Functions which are defined in assembler
void set_variables_in_asm();
void move_variables_in_asm();
void set_array_char_in_asm();
void set_array_num32_index_in_asm();
void set_array_num64_in_asm();
void set_array_num32_index_num32_in_asm();

void print_array_num64(){
    for (int i = 0; i < numArrayLength; i++)
    {
        printf("%ld", g_array_num64[i]);
        printf(",");
    }
    printf("\n");
}

void print_array_num32(){
    for (int i = 0; i < numArrayLength; i++)
    {
        printf("%d", g_array_num32[i]);
        printf(",");
    }
    printf("\n");
}

int main()
{
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);
    set_variables_in_asm();
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);
    move_variables_in_asm();
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);

    printf("g_array_char '%s'\n",g_array_char);
    set_array_char_in_asm();
    printf("g_array_char '%s'\n",g_array_char);

    print_array_num64();
    set_array_num64_in_asm();
    print_array_num64();

    print_array_num32();
    set_array_num32_index_in_asm();
    print_array_num32();
    g_num32 = 88888;
    g_index = 8;
    set_array_num32_index_num32_in_asm();
    print_array_num32();
}
