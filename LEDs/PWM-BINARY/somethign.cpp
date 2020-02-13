#include "mbed.h"


// DO NOT REMOVE OR RENAME FOLLOWING GLOBAL VARIABLES!!

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// LEDs on K64F-KIT - instances of class DigitalOut
/*
DigitalOut g_led1(PTA1);
DigitalOut g_led2(PTA2);
*/

// Buttons on K64F-KIT - instances of class DigitalIn
/*
DigitalIn g_but9(PTC9);
DigitalIn g_but10(PTC10);
DigitalIn g_but11(PTC11);
DigitalIn g_but12(PTC12);
*/

DigitalIn g_buttons[4] = {PTC9,PTC10,PTC11,PTC12};
DigitalOut g_redLeds[16] = {PTA1,PTA2,PTC0,PTC1,PTC2,PTC3,PTC4,PTC5,PTC7,PTC8,PTB9,PTB3,PTB2,PTB19,PTB18,PTB11};

/*
#define numOfLEDs 8
float LEDIncrease = 0.1;
int FirstLEDPosition = 0;
int LastLEDPosition = 0;
*/
//Buttons
/*
DigitalIn NextButton(PTC9);
DigitalIn PrevButton(PTC10);
*/

/*
float brightnessIncrease = 0.1;
int T = 15;
*/
/*
struct PWM
{
	DigitalOut LED;
	float brightness;

/*
 * @param -int from 0 - T based on number of ms passed from last cycle iteration
 * */ /*
	void Update(int Tick)
	{
		if (Tick < (T * brightness))
		{
			LED = true;
		}
		else
		{
			LED = false;
		}
	}

//Return True if brightness is bigger than 1 (100%), false if otherwise
	bool HasMaxBrigthness()
	{
		return brightness > 1;
	}

//Return True if brightness is lower than 0 (0%), false if otherwise
	bool HasMinBrigthness()
	{
		return brightness < 0;
	}
};
*/
/*
PWM LEDs[numOfLEDs] =
{
	{ DigitalOut(PTC0), 0 },
	{ DigitalOut(PTC1), 0 },
	{ DigitalOut(PTC2), 0 },
	{ DigitalOut(PTC3), 0 },
	{ DigitalOut(PTC4), 0 },
	{ DigitalOut(PTC5), 0 },
	{ DigitalOut(PTC7), 0 },
	{ DigitalOut(PTC8), 0 }
};
*/





int main()
{
// Serial line initialization
	g_pc.baud(115200);

	int binaryValue = 0;
	char str[10];

	while(1){
		g_redLeds[binaryValue] = false;
		//g_pc.printf(str, "%d", binaryValue);
		binaryValue = 0;

		for(int i = 0; i < 4; i++){
			if(!g_buttons[i]){
				if(i == 0){
					binaryValue+=1;
				}
				else if(i == 1){
					binaryValue+=2;
				}
				else if(i == 2){
					binaryValue+=4;
				}
				else if(i == 3){
					binaryValue+=8;
				}
			}
		}
		g_redLeds[binaryValue] = true;

	}


}

