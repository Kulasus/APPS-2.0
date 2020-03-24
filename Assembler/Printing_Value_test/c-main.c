#include <stdio.h>

#define messageLength 15

// Variables
char g_message[messageLength] = "xxxxxxxxxxxxxxx";

void set_message();

int main()
{
    set_message();
    printf("%s\n",g_message);
}
