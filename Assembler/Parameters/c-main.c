#include <stdio.h>

long g_num = 0;


void even_odd_count(int* array, int size, int* evenOdd);
void replace_chars(char* array, char replacer, char replaced);
long str_to_num(char* array);
void test(int x);


#define numArrayLength 10

void print_array_num32(int *array){ // Prints values in g_array_num32
    for (int i = 0; i < 2; i++)
    {
        printf("%d", array[i]);
        printf(",");
    }
    printf("\n");
}


int main()
{
    int arrayEvenOddCount[numArrayLength] = {1,2,3,4,5,6,0,8,9,10};
    int evenOddCount[2];
    even_odd_count(arrayEvenOddCount, numArrayLength, evenOddCount);
    print_array_num32(evenOddCount);
    printf("-------------------\n");
    char message[] = "Xozdrav z AXXS xocas xrazdnin";
    char find = 'x';
    char replace = 'p';
    printf("String %s\n", message);
    replace_chars(message,find,replace);
    printf("String %s\n", message);
    printf("-------------------\n");
    char string[] = "1000";
    long number = 0;
    number = str_to_num(string);
    printf("g_num: %ld\n", g_num);
    printf("number: %ld\n", number);
    
    int x = 10;
    test(x);
    printf("int: %d",x);

}