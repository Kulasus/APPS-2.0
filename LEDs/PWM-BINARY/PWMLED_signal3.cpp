#include "mbed.h"

// Constant variables
static const int g_numOfPwmLeds = 14; // All top redLeds + All rgbLeds = 14
static const int ledIncrease = 10; // 10% Increase of led brightness
static const int g_numOfRedLeds = 10; // Both top and bottom redLeds
static const int g_numOfRgbLeds = 6; // Two rgbLeds = 2*3 pins
static const int g_numOfButtons = 4;

// Global variables
int currentIndex = 0; // colourpicker function variable
int currentIndexSignal = 0; // signal function variable
int currentRedLedBrightness = 0; // signal function variable which stores current brightness of red leds
int previousRedLedBrightness = 0; // signal function variable which stores previous brightness of red leds
// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// All redLEDs on K64F-KIT
DigitalOut g_redLeds[g_numOfRedLeds] = {(PTA1),(PTA1),(PTC0),(PTC1),(PTC2),(PTC3),(PTC4),(PTC5),(PTC7),(PTC8)};

// All rgbLEDs on K644-KIT
DigitalOut g_rgbLeds[g_numOfRgbLeds] = {(PTB2),(PTB3),(PTB9),(PTB11),(PTB18),(PTB19)};

// All Buttons on K64F-KIT
DigitalIn g_buttons[g_numOfButtons] = {(PTC9),(PTC10),(PTC11),(PTC12)};

// PwmLed class definition
class PwmLed{
	private:
		DigitalOut *led; // Pointer to led object
		float brightness; // Leds brightness
		static const int timeUnit = 15; // Number of all states
		int timeFrame; // One state

	public:
		// Constructor
		PwmLed(PinName pin, int brightness)
		{
			this->led = new DigitalOut(pin); //Creation of new DigitalOut object (Led)
			this->setBrightness(brightness); //Using setter to recalculate percents to float variable
			this->timeFrame = 0;
		}

		// brightness setter, recalculates int value representing percents of brightness to float variable
		void setBrightness(int brightness){
			this->brightness = (float)brightness/100;
		}

		/* Function which is called timeUnit times. It determines if led is on or off in the current timeFrame
		This is basicaly how pwm is implemented. The program runs so quickly that you are not able to see if the led
		is on or off. Number of timeFrames in which led is on increases the brightness of led. */
		void update()
		{
			// Reset of timeFrame counter after it reaches timeUnit
			if(this->timeFrame == timeUnit){
				this->timeFrame = 0;
			}
			/* This calculation determines if led should be on or not, in dependency on brightness.
			If brightness is set higher, then the number of timeFrames in which led is on is bigger.*/
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

		// timeUnit getter
		int getTimeUnit(){
			return this->timeUnit;
		}
		// brightness getter
		int getBrightness(){
			return (int)(this->brightness*100);
		}
};

// Creation of Pwm objects.
PwmLed pwmLeds[g_numOfPwmLeds] =
{
	{ PTC0, 0}, // Top leds
	{ PTC1, 0}, // |
	{ PTC2, 0}, // |
	{ PTC3, 0}, // |
	{ PTC4, 0}, // |
	{ PTC5, 0}, // |
	{ PTC7, 0}, // |
	{ PTC8, 0}, // |
	{ PTB2, 100}, //B 8   // RGB leds
	{ PTB3, 100}, //G 9   // |
	{ PTB9, 100}, //R 10  // |
	{ PTB11, 0}, //B 11 // |
	{ PTB18, 0}, //G 12 // |
	{ PTB19, 0} //R 13  // |

};



// Function which calls update on every led in pwmLeds array
void updater(){
	for (int l_currentLed = 0; l_currentLed < g_numOfPwmLeds; l_currentLed++){
		pwmLeds[l_currentLed].update();
	}
}

// Function which simulates police beacon
void policie(){;
	pwmLeds[8].setBrightness(100);
	pwmLeds[10].setBrightness(0);
	pwmLeds[11].setBrightness(0);
	pwmLeds[13].setBrightness(100);
	wait_ms(100);
	pwmLeds[8].setBrightness(0);
	pwmLeds[10].setBrightness(100);
	pwmLeds[11].setBrightness(100);
	pwmLeds[13].setBrightness(0);
	wait_ms(100);

}

/* Function which lets you choose led with buttons, and increase/decrease its brightness with buttons.
You can also select rgb leds, and then mix their colour. */
void colourPicker(){

	// Go to next led
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

	// Go to previous led
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

	// Add brightness
	if(!g_buttons[1] && pwmLeds[currentIndex].getBrightness() < 100){
		pwmLeds[currentIndex].setBrightness(pwmLeds[currentIndex].getBrightness()+ledIncrease);
		g_pc.printf("Current BRIGHTNESS: ");
		g_pc.printf("%d\r\n", pwmLeds[currentIndex].getBrightness());
		wait_ms(100);
	}

	// Decrease brightness
	if(!g_buttons[0] && pwmLeds[currentIndex].getBrightness() > 0){
		pwmLeds[currentIndex].setBrightness(pwmLeds[currentIndex].getBrightness()-ledIncrease);
		g_pc.printf("Current BRIGHTNESS: ");
		g_pc.printf("%d\r\n", pwmLeds[currentIndex].getBrightness());
		wait_ms(100);
	}
}
/* Function which calculates and sets brightness of all top red leds based on brightness of rgb led. You can set brightness of rgb led by
holding buttons(0-2), or reset all leds by button 3. */
void signal(){
	// Increase brightness of blue led
	if(!g_buttons[0] && pwmLeds[8].getBrightness() < 100){
		pwmLeds[8].setBrightness(pwmLeds[8].getBrightness()+ledIncrease);
	}
	// Increase brightness of green led
	if(!g_buttons[1] && pwmLeds[9].getBrightness() < 100){
		pwmLeds[9].setBrightness(pwmLeds[9].getBrightness()+ledIncrease);
	}
	// Increase brightness of red led
	if(!g_buttons[2] && pwmLeds[10].getBrightness() < 100){
		pwmLeds[10].setBrightness(pwmLeds[10].getBrightness()+ledIncrease);
	}
	// Automaticaly decrease brightness of blue led
	if(g_buttons[0] && pwmLeds[8].getBrightness() > 0){
		pwmLeds[8].setBrightness(pwmLeds[8].getBrightness()-ledIncrease);
	}
	// Automaticaly decrease brightness of green led
	if(g_buttons[1] && pwmLeds[9].getBrightness() > 0){
		pwmLeds[9].setBrightness(pwmLeds[9].getBrightness()-ledIncrease);
	}
	// Automaticaly decrease brightness of red led
	if(g_buttons[2] && pwmLeds[10].getBrightness() > 0){
		pwmLeds[10].setBrightness(pwmLeds[10].getBrightness()-ledIncrease);
	}
	// Turns of all leds
	if(!g_buttons[3]){
		for(int i = 0; i < g_numOfPwmLeds - 3; i++){
			pwmLeds[i].setBrightness(0);
		}
	}
	// Storing brightness to prevent serial output overloading
	previousRedLedBrightness = currentRedLedBrightness;
	// Calculation of red leds brightness depending on sum of brightness of rgb led
	currentRedLedBrightness = (pwmLeds[8].getBrightness()+pwmLeds[9].getBrightness()+pwmLeds[10].getBrightness())/3;
	// Printing brightness to serial output
	if(currentRedLedBrightness != previousRedLedBrightness){
		g_pc.printf("Current redLEDs brightness: ");
		g_pc.printf("%d\r\n", currentRedLedBrightness);
	}
	// Setting redLeds brightness to calculated value
	for(int i = 0; i < g_numOfPwmLeds - 6; i++){
		pwmLeds[i].setBrightness(currentRedLedBrightness);
	}
	// Wait to actualy see the change
	wait_ms(50);
}

bool freeze = true;
int counter = 0;
void signal2(){

		if(g_buttons[0] && pwmLeds[8].getBrightness() < 100 && freeze) {
			pwmLeds[8].setBrightness(pwmLeds[8].getBrightness()+ledIncrease);
			pwmLeds[11].setBrightness(pwmLeds[11].getBrightness()-ledIncrease);
		}

		if(g_buttons[1] && pwmLeds[9].getBrightness() < 100 && freeze){
			pwmLeds[9].setBrightness(pwmLeds[9].getBrightness()+ledIncrease);
			pwmLeds[12].setBrightness(pwmLeds[12].getBrightness()-ledIncrease);
		}

		if(g_buttons[2] && pwmLeds[10].getBrightness() < 100 && freeze){
			pwmLeds[10].setBrightness(pwmLeds[10].getBrightness()+ledIncrease);
			pwmLeds[13].setBrightness(pwmLeds[13].getBrightness()-ledIncrease);
		}

		if(!g_buttons[0] && pwmLeds[11].getBrightness() < 100 && freeze){
			pwmLeds[11].setBrightness(pwmLeds[11].getBrightness()+ledIncrease);
			pwmLeds[8].setBrightness(pwmLeds[8].getBrightness()-ledIncrease);
		}

		if(!g_buttons[1] && pwmLeds[12].getBrightness() < 100 && freeze){
			pwmLeds[12].setBrightness(pwmLeds[12].getBrightness()+ledIncrease);
			pwmLeds[9].setBrightness(pwmLeds[9].getBrightness()-ledIncrease);
		}

		if(!g_buttons[2] && pwmLeds[13].getBrightness() < 100 && freeze){
			pwmLeds[13].setBrightness(pwmLeds[13].getBrightness()+ledIncrease);
			pwmLeds[10].setBrightness(pwmLeds[10].getBrightness()-ledIncrease);
		}

		if(!g_buttons[3])
		{
			freeze = false;
		}

		if(!g_buttons[0] || !g_buttons[1] || !g_buttons[2])
		{
			freeze = true;
		}

		previousRedLedBrightness = currentRedLedBrightness;
		currentRedLedBrightness = (pwmLeds[11].getBrightness()+pwmLeds[12].getBrightness()+pwmLeds[13].getBrightness())/30;

		// Printing brightness to serial output
		if(currentRedLedBrightness != previousRedLedBrightness){
			g_pc.printf("Current redLEDs brightness:  " );
			g_pc.printf("%d\r\n", pwmLeds[11].getBrightness());
		}
		// Setting redLeds brightness to calculated value
		for(int i = 0; i < g_numOfPwmLeds - 6; i++){
			if(i < currentRedLedBrightness)
			{
				pwmLeds[i].setBrightness(currentRedLedBrightness*10);
			}
			else
			{
				pwmLeds[i].setBrightness(0);
			}
		}
		wait_ms(130);
}

int main()
{
	// Serial line initialization
 	g_pc.baud(115200);

	// Ticker calling updater every ms
	Ticker ticker;
	ticker.attach_us(callback(updater),1000);

	// Uncomment only one!
	while (1)
	{
		//policie();
		//colourPicker();
		//signal();
		signal2();

	}
}

