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
[CUDA programming](http://poli.cs.vsb.cz/edu/apps/cuda/cuda-programming.pdf) </br>
[CUDA tutorial](https://developer.nvidia.com/cuda-education-training)

## Current TODOs:
- [X] Logging
- [X] LEDs
- [X] LEDs with PWM
- [X] Test PwmLed class
- [ ] Snake 

## LEDs:
Source code for Leds is now in LEDs/PWM-BINARY/main.cpp
### Serial line output: 
$ minicom -D /dev/ttyACM0
### binaryDisplay.cpp
Program which displays binary value which is selected with buttons
### ledsPWM.cpp
Program implements PwmLed class. This class represents pulse wide modulation to create illusion of brightness.
There are also two simulations:
- policie
    - Function which simulates police beacon.
- colourPicker
    - Function which lets you select led with button and increase/decrease its brightness with buttons.
### snake.cpp
Program also implements PwmLed class. This program implements function which represent snake, which goes throught leds.
You can stop him, increase or decrease his speed. There will be also representation of health and enemies.
- [ ] Not implemented yet

## Authors:
@Kulasus, @lolray
