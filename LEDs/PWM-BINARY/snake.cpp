#include "mbed.h"

// Constant variables
static const int g_numOfPwmLeds = 10;
static const float ledIncrease = 0.05;
static const int g_numOfRedLeds = 10;
static const int g_numOfButtons = 4;
static const int g_numOfRgbLeds = 6;

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// All redLEDs on K64F-KIT
DigitalOut g_redLeds[g_numOfRedLeds] = {(PTA1),(PTA1),(PTC0),(PTC1),(PTC2),(PTC3),(PTC4),(PTC5),(PTC7),(PTC8)};

// All rgbLEds on K64F-KIT
DigitalOut g_rgbLeds[g_numOfRgbLeds] = {PTB9,PTB3,PTB2,PTB19,PTB18,PTB11};

// All Buttons on K64F-KIT
DigitalIn g_buttons[g_numOfButtons] = {(PTC9),(PTC10),(PTC11),(PTC12)};

// PwmLed struct definition
struct PwmLed{
		DigitalOut led;
		float brightness;
		static const int timeUnit = 15; //Number of all states

		//Just a setter
		void setBrightness(float brightness){
			this->brightness = brightness;
		}
		void update(int timeFrame) //timeFrame is one state of all states in timeUnit
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
	{ PTC0, 0 },
	{ PTC1, 0 },
	{ PTC2, 0 },
	{ PTC3, 0 },
	{ PTC4, 0 },
	{ PTC5, 0 },
	{ PTC7, 0 },
	{ PTC8, 0 },
	{ PTB3, 0.05},
	{ PTB18, 0.05}
};

void updater(){
	for(int l_tick = 0; l_tick < pwmLeds[0].getTimeUnit(); l_tick++){
		for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++)
		{
			pwmLeds[l_currentLed].update(l_tick);
		}
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

	bool stopka = false;

	int indexHlava = 2;
	int indexTelo = 1;
	int indexOcas = 0;

	while(1){
		for(int i = 0; i < g_numOfPwmLeds-2; i++){
			pwmLeds[i].setBrightness(0.0);
		}
		pwmLeds[indexHlava].setBrightness(1.0);
		pwmLeds[indexTelo].setBrightness(0.88);
		pwmLeds[indexOcas].setBrightness(0.6);

		if(!stopka){
			indexHlava += 1; indexTelo += 1; indexOcas += 1;
		}

		if(!g_buttons[0]){
			stopka = !stopka;
			wait_ms(25);
		}

		if(indexHlava == 8){
			indexHlava = 0;
		}
		if(indexTelo == 8){
			indexTelo = 0;
		}
		if(indexOcas == 8){
			indexOcas = 0;
		}
		wait_ms(50);
	}
}

