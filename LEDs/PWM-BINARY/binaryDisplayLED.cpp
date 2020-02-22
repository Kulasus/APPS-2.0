#include "mbed.h"

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// All buttons on K64F-KIT
DigitalIn g_buttons[4] = {PTC9,PTC10,PTC11,PTC12};

// All leds on K64F-KIT
DigitalOut g_leds[16] = {PTA1,PTA2,PTC0,PTC1,PTC2,PTC3,PTC4,PTC5,PTC7,PTC8,PTB9,PTB3,PTB2,PTB19,PTB18,PTB11};

int main()
{
	// Serial line initialization
	g_pc.baud(115200);

	// binaryValue stores current value to display
	int binaryValue = 0;
	int previousBinaryValue = 0;

	/* displays current binaryValue on led -> You have to hold buttons:
	 	 PTC9 -> 1
	 	 PTC10 -> 2
	 	 PTC11 -> 4
	 	 PTC12 -> 8
	*/
	while(1){

		// turns off led which was on on start of every cycle
		g_leds[binaryValue] = false;

		// sets binaryValue to 0 on start of every cycle
		previousBinaryValue = binaryValue;
		binaryValue = 0;

		// add value to binaryValue for each pressed button
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

		// displays selected binaryValue to console, if binaryValue is different from previous one
		// this alsou prevents serial output from killing microcontrollers power storage ... 
		if(binaryValue != previousBinaryValue){
			g_pc.printf("%d\r\n", binaryValue);
		}

		// turns on led which equals to current binaryValue
		g_leds[binaryValue] = true;
	}


}

