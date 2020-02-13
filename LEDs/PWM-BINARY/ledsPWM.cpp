
#include "mbed.h"

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// Bottom LEDs on K64F-KIT
DigitalOut g_led1(PTA1);
DigitalOut g_led2(PTA2);

// Buttons on K64F-KIT
DigitalIn g_but9(PTC9);
DigitalIn g_but10(PTC10);
DigitalIn g_but11(PTC11);
DigitalIn g_but12(PTC12);

// Useful variables
static const int g_numOfPwmLeds = 8;
static float ledIncrease = 0.1;

// PwmLed class definition
struct PwmLed{
		DigitalOut led;
		float brightness;
		static const int timeUnit = 15;

		void setBrightness(float brightness){
			brightness = brightness;
		}
		void update(int timeFrame)
		{
			if (timeFrame < (timeUnit * brightness))
			{
				led = true;
			}
			else
			{
				led = false;
			}
		}
		int getTimeUnit(){
			return timeUnit;
		}
		float getBrightness(){
			return brightness;
		}
};

// Top LEDs on K64F-KIT
PwmLed pwmLeds[g_numOfPwmLeds] =
{
	{ PTC0, 0 },
	{ PTC1, 0 },
	{ PTC2, 0 },
	{ PTC3, 0 },
	{ PTC4, 0 },
	{ PTC5, 0 },
	{ PTC7, 0 },
	{ PTC8, 0 }
};

int main()
{
	// Serial line initialization
	g_pc.baud(115200);

	// Increasing led brightness from 0% to 100% by 10% repetitively
	while (1)
	{
		// Updating all leds
		for(int l_tick = 0; l_tick < pwmLeds[0].getTimeUnit(); l_tick++){
			for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++)
			{
				pwmLeds[l_currentLed].update(l_tick);
			}
		}
		// Increasing brightness by 10% or decreasing it to 0% if current brightness is 100%
		for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++)
		{
			if(pwmLeds[l_currentLed].getBrightness() == 1.0){
				pwmLeds[l_currentLed].setBrightness(0.0);
			}
			else{
				pwmLeds[l_currentLed].setBrightness(pwmLeds[l_currentLed].getBrightness()+ledIncrease);
			}
		}
	}
}

