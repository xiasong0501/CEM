#include <stdio.h>
int x = 1234; /* file scope */
double y = 1.234567; /* file scope */
void function_1(void);
int main(void)
{
 int x = 4321; /* main block scope */
 function_1();
 printf("In main block:\n x = %d, y = %lf\n", x, y);
 /* nested block */
 {
 double y = 7.654321; /* nested block scope */
 function_1();
 printf("In nested block:\n x = %d, y = %lf\n", x, y);
 }
 return 0;
}
void function_1(void)
{
 printf("From function_1:\n x = %d, y = %lf\n", x, y);
}