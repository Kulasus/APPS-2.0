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
static int g_numOfPwmLeds = 8;
static float ledIncrease = 0.1;

// PwmLed class definition
class PwmLed{
	private:
		static int timeUnit = 15;
		float brightness;
		DigitalOut led;
	public:
		PwmLed (DigitalOut, float);
		void setBrigthness (float);
		void update (int);
		int getTimeUnit();
		float getBrightness();
}

// PwmLed class implementation
PwmLed::PwmLed(DigitalOut led, float brightness){
	led = led;
	brightness = brightness;
}
void PwmLed::setBrightness(float brightness){
	brightness = brightness;
}
void PwmLed::update(int timeFrame){
	if (timeFrame < (timeUnit * brightness))
	{
		led = true;
	}
	else
	{
		led; = false;
	}
}
int PwmLed::getTimeUnit(){
	return timeUnit;
}
float PwmLed::getBrightness(){
	return brightness;
}

// Top LEDs on K64F-KIT
PwmLed pwmLeds[g_numOfPwmLeds] = {
	{new PwmLed(DigitalOut(PTC0), 0)},
	{new PwmLed(DigitalOut(PTC1), 0)},
	{new PwmLed(DigitalOut(PTC2), 0)},
	{new PwmLed(DigitalOut(PTC3), 0)},
	{new PwmLed(DigitalOut(PTC4), 0)},
	{new PwmLed(DigitalOut(PTC5), 0)},
	{new PwmLed(DigitalOut(PTC7), 0)},
	{new PwmLed(DigitalOut(PTC8), 0)}
}

int main()
{
	// Serial line initialization
	g_pc.baud(115200);

	// Increasing led brightness from 0% to 100% by 10% repetitively
	while (1)
	{
		// Updating all leds
		for(int l_tick = 0; l_tick < PwmLed[0]->getTimeUnit(); l_tick++){
			for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; i++)
			{
				pwmLeds[l_currentLed]->update(l_tick);
			}
		}
		// Increasing brightness by 10% or decreasing it to 0% if current brightness is 100%
		for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; i++)
		{
			if(pwmLeds[l_currentLed]->getBrightness() == 1.0){
				pwmLeds[l_currentLed]->setBrightness(0.0);
			}
			else{
				pwmLeds[l_currentLed]->setBrightness(pwmLeds[l_currentLed]->getBrightness()+ledIncrease);
			}
		}
	{
}
