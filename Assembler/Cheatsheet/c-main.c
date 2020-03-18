
#include <stdio.h>

#define numArrayLength 10

// Variables
char g_char = 0;
int g_num32 = 0;
long g_num64 = 0;
int g_index = 2;
int g_index_plus_one = 3;
int g_firstIndexOfArray = 0;
int g_lastIndexOfArray = numArrayLength - 1;
int g_switchIndex1 = 0;
int g_switchIndex2 = 2;

char g_array_char[128] = "Some message";
int g_array_num32[numArrayLength] = {0,1,2,3,4,5,6,7,8,9};
long g_array_num64[numArrayLength] = {10,11,12,13,14,15,16,17,18,19};

// Functions which are defined in assembler
void set_variables_in_asm(); // Sets variables values in assembler
void move_variables_in_asm(); // Moves g_char -> g_num32 and g_num32 -> g_num64 in assembler 
void set_array_char_in_asm(); // Sets first 4 characters of g_array_char in assembler
void set_array_num32_index_in_asm(); // Sets certain values on index declared in c in g_array_num32 in assembler
void set_array_num64_in_asm(); // Sets certain values on certain indexes in g_array_num64 in assembler
void set_array_num32_index_num32_in_asm(); // Sets values declared in c into index declared in c in g_array_num32 in assembler
void switch_two_values_index_num32_in_asm(); // Switches two values on certain indexes in g_array_num32 in assembler
void switch_first_and_last_index_array_num32_in_asm(); // Switches values on first and last index of g_array_num32 in assembler
void switch_between_array_num32_array_num64_in_asm(); // Switches first three values from g_array_num32 to g_array_num64 and backwards in assembler
void switch_neighbors_array_num32_in_asm(); // Switches neighbors values in g_array_num32 -> WORKS ONLY FOR array lenght = 10
void set_array_num32_neighbors_value(); // Sets value of entity on certain index to value of its right neigbour

void print_array_num64(){ // Prints values in g_array_num64
    for (int i = 0; i < numArrayLength; i++)
    {
        printf("%ld", g_array_num64[i]);
        printf(",");
    }
    printf("\n");
}

void print_array_num32(){ // Prints values in g_array_num32
    for (int i = 0; i < numArrayLength; i++)
    {
        printf("%d", g_array_num32[i]);
        printf(",");
    }
    printf("\n");
}

void switch_neigbors_refactored_array_num32_in_asm(){ // Switches neighbors values in g_array_num32, works for variable array length
    for (int i = 0; i < numArrayLength; i+=2)
    {
        g_switchIndex1 = i;
        g_switchIndex2 = i + 1;
        if(g_switchIndex2 >= numArrayLength){
            break;
        }
        switch_two_values_index_num32_in_asm();
    }
}

int main()
{
    /*
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);
    set_variables_in_asm();
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);
    move_variables_in_asm();
    printf("g_char %c g_num32 %d g_num64 %ld\n",g_char,g_num32,g_num64);
    */

    /*
    printf("g_array_char '%s'\n",g_array_char);
    set_array_char_in_asm();
    printf("g_array_char '%s'\n",g_array_char);
    */

    /*
    print_array_num64();
    set_array_num64_in_asm();
    print_array_num64();
    */

    /*
    print_array_num32();
    set_array_num32_index_in_asm();
    print_array_num32();
    */

    /*
    g_num32 = 88888;
    g_index = 8;
    print_array_num32();
    set_array_num32_index_num32_in_asm();
    print_array_num32();
    */

    /*
    print_array_num32();
    switch_first_and_last_index_array_num32_in_asm();
    print_array_num32();
    */

    /*
    print_array_num32();
    print_array_num64();
    switch_between_array_num32_array_num64_in_asm();
    print_array_num32();
    print_array_num64();
    */

    /*
    print_array_num32();
    switch_neighbors_array_num32_in_asm();
    print_array_num32();
    */

    /*
    g_index_plus_one = g_index +1;
    print_array_num32();
    set_array_num32_neighbors_value();
    print_array_num32();
    */

    /*
    print_array_num32();
    switch_two_values_index_num32_in_asm();
    print_array_num32();
    */

    print_array_num32();
    switch_neigbors_refactored_array_num32_in_asm();
    print_array_num32();
}