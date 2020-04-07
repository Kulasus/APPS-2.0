# APPS-2.0
Repository for Architecture of computers and parallel systems course on VŠB
## Built With
[MCUXpresso IDE](https://www.nxp.com/design/software/development-software/mcuxpresso-software-and-tools/mcuxpresso-integrated-development-environment-ide:MCUXpresso-IDE) </br>
[CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
## Licence:
GNU General Public License v3.0

## Useful links:
[Petr Olivka's web page about course](http://poli.cs.vsb.cz/edu/apps) </br>
[Course syllabus](http://poli.cs.vsb.cz/edu/apps/lab/apps-syllabus.pdf) </br>
[Petr Olivka's youtube channel (Assembly language tutorials)](https://www.youtube.com/channel/UCVsJ3Mvp8HL_kFgqgSpHjlA) </br>
[Assembly language example files and references](http://poli.cs.vsb.cz/edu/apps/soj/) </br>
[Assembly language textbook](http://poli.cs.vsb.cz/edu/apps/soj/down/apps-soj-skripta.pdf) </br>
[CUDA programming](http://poli.cs.vsb.cz/edu/apps/cuda/cuda-programming.pdf) </br>
[CUDA tutorial](https://developer.nvidia.com/cuda-education-training)

## Current TODOs:
- [ ] Assembly language part 0
- [ ] Assembly language part 1

## LEDs:
Implementation of programs for leds on K64F-KIT. First part of course. Individual programs are in APPS-2.0/LEDs/PWM-BINARY/
### Serial line output: 
```shell
$ minicom -D /dev/ttyACM0
```
### Complete PwmLed class:
More information about how this works is in ledsPWM.cpp
```cpp
class PwmLed{
	private:
		DigitalOut *led;
		float brightness;
		static const int timeUnit = 15;
		int timeFrame;

	public:
		PwmLed(PinName pin, int brightness)
		{
			this->led = new DigitalOut(pin);
			this->setBrightness(brightness);
			this->timeFrame = 0;
		}
		void setBrightness(int brightness){
			this->brightness = (float)brightness/100;
		}
		void update()
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
		int getTimeUnit(){
			return this->timeUnit;
		}
		int getBrightness(){
			return (int)(this->brightness*100);
		}
};
```
### binaryDisplay.cpp
Program which displays binary value which is selected with buttons
### ledsPWM.cpp
Program implements PwmLed class. This class represents pulse wide modulation on led to create illusion of brightness.
There are also three simulations:
- policie
    - Function which simulates police beacon.
- colourPicker
    - Function which lets you select led with button and increase/decrease its brightness with buttons.
- singal
    - Function which calculates and sets brightness of top red leds based on brighntess of rgb led.
### snake.cpp
Program also implements PwmLed class. This program implements function which represent snake, which goes throught leds.
You can stop him, increase or decrease his speed.
## LCD
Implementation of programs for LCD on K64F-KIT. Second part of course. Individual programs are in APPS-2.0/LCD
### shapes.cpp
Program implements multiple classes which represents shapes (triangle, circle...) and classes for displaying text on the LCD
### Drawing character on LCD from font.h file logic
```cpp
// Draw function implementation using for cycles to iterate over values in fonts
    void draw()
    {
    	// Iterating over rows
        for(int y = 0; y < HEIGHT; y++)
        {
	    // Selecting one specific position in fonts file
            int radek_fontu = font[character][y];

	    // Iterating over characters until we reach the width of character (last value of character)
            for(int x = 0; x < WIDTH; x++)
            {
                //if(radek_fontu & (HEIGHT-WIDTH << x)) drawPixel(pos.x + x, pos.y + y);	       //LSB
            	if(radek_fontu & (HEIGHT-WIDTH << x)) drawPixel(pos.x - x + WIDTH, pos.y + y);    //MSB
            }
        }
    }
```
## Assembler
Fourth part of the course focused on programming in Assembler&C languages. Individual programs can be found in APPS-2.0/Assembler
### Cheatsheet
In this file you can find various useful functions in Assembler, basicaly all other files in this repo will somehow be subsets of this file.
### Cheatsheet 2
Extended Cheatsheet 1 
### Printing_Value_test
In this file you can find very basic function in assembler which will set value of variable to 'System funguje.' and print it to console.
### Basic_MOV_functions
In this file you can find first assigment from Assembler language. All functions are using only MOV/MOVSX instructions. There are multiple functions, some of them are using functions in C to take a use of cycles. Some of them are totaly bruteforce and wont work on different values or different array lengths.
### Basic_MOV_functions_bruteforce
This file contains only bruteforce subset of functions from Basic_MOV_functions. 
### Parameters
This file contains functions based around taking function parameters from C to assembler and returning values using RAX register.
### Parameters Realtime
This file contains other functions based around taking function parameters from C to assembler and returng values using RAX register. This functions were in realtime test.
### Divide&Conquer
This file contains functions based around division, multiplication, using stack and conditional movement 
## Authors:
[@Kulasus](https://github.com/Kulasus), [@lolray](https://github.com/lolray)

## Special thanks:
[@michalscepka](https://github.com/michalscepka)
