#include "mbed.h"

// Constant variables
static const int g_numOfPwmLeds = 10;
static const float ledIncrease = 0.1;
static const int g_numOfRedLeds = 10;
static const int g_numOfButtons = 4;

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// All redLEDs on K64F-KIT
DigitalOut g_redLeds[g_numOfRedLeds] = {(PTA1),(PTA1),(PTC0),(PTC1),(PTC2),(PTC3),(PTC4),(PTC5),(PTC7),(PTC8)};

// All Buttons on K64F-KIT
DigitalIn g_buttons[g_numOfButtons] = {(PTC9),(PTC10),(PTC11),(PTC12)};

// PwmLed struct definition
struct PwmLed{
		DigitalOut led;
		float brightness;
		static const int timeUnit = 10; //Number of all states
		int timeFrame;

		//Just a setter
		void setBrightness(float brightness){
			this->brightness = brightness;
		}
		void update() //timeFrame is one state of all states in timeUnit
		{
			if(timeFrame == 15){
				timeFrame = 0;
			}
			if (timeFrame < (timeUnit * brightness))
			{
				led = true;
			}
			else
			{
				led= false;
			}
			timeFrame++;
		}

		//Just a getter
		int getTimeUnit(){
			return timeUnit;
		}
		//Just a getter
		float getBrightness(){
			return brightness;
		}
};

// Top leds on K64F-KIT -> creation of PwmStructs...
PwmLed pwmLeds[g_numOfPwmLeds] =
{
	{ PTC0, 0, 0},
	{ PTC1, 0, 0 },
	{ PTC2, 0, 0 },
	{ PTC3, 0, 0 },
	{ PTC4, 0, 0 },
	{ PTC5, 0, 0 },
	{ PTC7, 0, 0 },
	{ PTC8, 0, 0 },
	{ PTB9, 0, 0},
	{ PTB19, 0, 0}
};

void updater(){
	for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++){
		pwmLeds[l_currentLed].update();
	}
}

int main()
{
	// Serial line initialization
 	g_pc.baud(115200);
	int currentIndex = 0;

	// Increasing led brightness from 0% to 100% by 10% repetitively
	Ticker ticker;
	ticker.attach_us(callback(updater),1000);

	while (1)
	{
		if(!g_buttons[3]){
			if(currentIndex == g_numOfPwmLeds - 1){
				currentIndex = 0;
			}
			else{
				currentIndex++;
			}
			g_pc.printf("Current INDEX: ");
			g_pc.printf("%d\r\n", currentIndex);
			wait_ms(150);
		}
		if(!g_buttons[2]){
			if(currentIndex == 0){
				currentIndex = g_numOfPwmLeds - 1;
			}
			else{
				currentIndex--;
			}
			g_pc.printf("Current INDEX: ");
			g_pc.printf("%d\r\n", currentIndex);
			wait_ms(150);
		}
		if(!g_buttons[1] && pwmLeds[currentIndex].getBrightness() < 1.0){
			pwmLeds[currentIndex].setBrightness(pwmLeds[currentIndex].getBrightness()+ledIncrease);
			g_pc.printf("Current BRIGHTNESS: ");
			g_pc.printf("%f\r\n", pwmLeds[currentIndex].getBrightness()*10);
			wait_ms(100);
		}
		if(!g_buttons[0] && pwmLeds[currentIndex].getBrightness() > 0.0){
			pwmLeds[currentIndex].setBrightness(pwmLeds[currentIndex].getBrightness()-ledIncrease);
			g_pc.printf("Current BRIGHTNESS: ");
			g_pc.printf("%f\r\n", pwmLeds[currentIndex].getBrightness()*10);
			wait_ms(100);
		}
	}
}


