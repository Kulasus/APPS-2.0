#include "mbed.h"

// Constant variables
static const int g_numOfPwmLeds = 14;
static const float ledIncrease = 0.1;
static const int g_numOfRedLeds = 10;
static const int g_numOfButtons = 4;
// Global variables
int currentIndex = 0; //for colourpicker function

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// All redLEDs on K64F-KIT
DigitalOut g_redLeds[g_numOfRedLeds] = {(PTA1),(PTA1),(PTC0),(PTC1),(PTC2),(PTC3),(PTC4),(PTC5),(PTC7),(PTC8)};

// All Buttons on K64F-KIT
DigitalIn g_buttons[g_numOfButtons] = {(PTC9),(PTC10),(PTC11),(PTC12)};

// PwmLed struct definition
class PwmLed{
	private:
		DigitalOut *led; //Pointer to led object
		float brightness;
		static const int timeUnit = 15; //Number of all states
		int timeFrame; //One state

	public:
		PwmLed(PinName pin, float brightness)
		{
			this->led = new DigitalOut(pin);
			this->brightness = brightness;
			this->timeFrame = 0;
		}

		//Just a setter
		void setBrightness(float brightness){
			this->brightness = brightness;
		}
		void update() //timeFrame is one state of all states in timeUnit
		{
			if(this->timeFrame == timeUnit){
				this->timeFrame = 0;
			}
			if (this->timeFrame < (this->timeUnit * this->brightness))
			{
				this->led->write(1);
			}
			else
			{
				this->led->write(0);
			}
			this->timeFrame++;
		}

		//Just a getter
		int getTimeUnit(){
			return this->timeUnit;
		}
		//Just a getter
		float getBrightness(){
			return this->brightness;
		}
};

// Top leds on K64F-KIT -> creation of Pwm objects...
PwmLed pwmLeds[g_numOfPwmLeds] =
{
	{ PTC0, 0},
	{ PTC1, 0},
	{ PTC2, 0},
	{ PTC3, 0},
	{ PTC4, 0},
	{ PTC5, 0},
	{ PTC7, 0},
	{ PTC8, 0},
	{ PTB2, 0}, //B 8
	{ PTB3, 0}, //G 9
	{ PTB9, 0}, //R 10
	{ PTB11, 0}, //B 11
	{ PTB18, 0}, //G 12
	{ PTB19, 0} //R 13

};

//Calling update function on all leds in array
void updater(){
	for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++){
		pwmLeds[l_currentLed].update();
	}
}

//For fun :D
void policie(){;
	pwmLeds[8].setBrightness(1.0);
	pwmLeds[10].setBrightness(0);
	pwmLeds[11].setBrightness(0);
	pwmLeds[13].setBrightness(1.0);
	wait_ms(100);
	pwmLeds[8].setBrightness(0);
	pwmLeds[10].setBrightness(1.0);
	pwmLeds[11].setBrightness(1.0);
	pwmLeds[13].setBrightness(0);

}


void colourPicker(){

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

int main()
{
	// Serial line initialization
 	g_pc.baud(115200);


	Ticker ticker;
	ticker.attach_us(callback(updater),1000);


	while (1)
	{
		//policie();
		//colourPicker();
	}
}

