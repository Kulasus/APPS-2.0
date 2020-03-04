#include "mbed.h"
#include "lcd_lib.h"
#include <string>
#include "font22x36_msb.h"		// Font import
#include <vector>

#define WIDTH 22	// Font width
#define HEIGHT 36	// Font height
int rozdil = HEIGHT - WIDTH;

// Serial line for printf output
Serial pc(USBTX, USBRX);

using namespace std;

DigitalOut g_backLight(PTC3, 0);		// Display backlight

int offset = 22; //Space between font characters

// Structure which holds coordinates of one point
struct Point2D
{
    int32_t x, y;
};
// Structure which holds values of RGB of one colour
struct RGB
{
    uint8_t r, g, b;
};
// GraphElement class implementation
class GraphElement
{
public:
    // Foreground and background colour
    RGB fg_color, bg_color;
	
    GraphElement(){}

    GraphElement( RGB t_fg_color, RGB t_bg_color ) : fg_color( t_fg_color ), bg_color( t_bg_color ) {}

    // Function which draws pixel on certain position with foreground colour
    void drawPixel( int32_t t_x, int32_t t_y ) {
	    lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( fg_color )); 
    }

    // Function which draws the specific graphical element
    virtual void draw() = 0;

    // Function which creates illusion of disapearement of certail graphical element (calls swap_fg_bg_color() function)
    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }
	
private:
    // Function which swaps foreground and backgroun colour
    void swap_fg_bg_color() {
	    RGB l_tmp = fg_color;
	    fg_color = bg_color;
	    bg_color = l_tmp;
    }

    // Function which converts 24-bit RGB colour to 16-bit RGB colour
    int convert_RGB888_to_RGB565(RGB t_color)
    {
	// Union which represents 16-bit RGB colour -> Union unlike struct allocates shared memory for all of its variables. 
	// It's easier for us then to use bit operations like bit shifitng, because all this operations takes place in one shared memory.
	// If we would use struct, there would be risk of accessing different memory then we want.
        union URGB {
		struct 
		{
			int b:5; 
			int g:6; 
			int r:5;
		}; 
		short rgb565; 
	} urgb;
	
	// This calculation shifts bits of each color value to right. After that it uses bitwise AND to anulate reamining bits
	// in 24-bit part
        urgb.r = (t_color.r >> 3) & 0x1F;
        urgb.g = (t_color.g >> 2) & 0x3F;
        urgb.b = (t_color.b >> 3) & 0x1F;
	    
        return urgb.rgb565;
    }
};

// Pixel class implementation, unlike GraphElement this class have also position.
class Pixel : public GraphElement
{
public:
    // Position of pixel on LCD	
    Point2D pos;
    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : pos( t_pos ), GraphElement( t_fg_color, t_bg_color ) {}
    // Draw method implementation
    virtual void draw() {
    	drawPixel( pos.x, pos.y ); 
    }   
};
// Circle class implementation
class Circle : public GraphElement
{
public:
    // Center of circle
    Point2D center;
    // Radius of circle
    int32_t radius;

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) :
        center( t_center ), radius( t_radius ), GraphElement( t_fg, t_bg ) {};
	
    // Implementation of draw function using Bresenham Circle algorithm
    void draw()
    {
        int f = 1 - radius;
        int ddF_x = 0;
        int ddF_y = -2 * radius;
        int x = 0;
        int y = radius;

        int x0 = center.x;
        int y0 = center.y;

        drawPixel(x0, y0 + radius);
        drawPixel(x0, y0 - radius);
        drawPixel(x0 + radius, y0);
        drawPixel(x0 - radius, y0);

        while(x < y)
        {
            if(f >= 0)
            {
                y--;
                ddF_y += 2;
                f += ddF_y;
            }
            x++;
            ddF_x += 2;
            f += ddF_x + 1;
            drawPixel(x0 + x, y0 + y);
            drawPixel(x0 - x, y0 + y);
            drawPixel(x0 + x, y0 - y);
            drawPixel(x0 - x, y0 - y);
            drawPixel(x0 + y, y0 + x);
            drawPixel(x0 - y, y0 + x);
            drawPixel(x0 + y, y0 - x);
            drawPixel(x0 - y, y0 - x);
        }
    }
};

// Line class implementation
class Line : public GraphElement
{
public:
    // The first and the last point of line
    Point2D pos1, pos2;

    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) :
      pos1( t_pos1 ), pos2( t_pos2 ), GraphElement( t_fg, t_bg ) {}
    
    // Draw function implementation using Bresenham Line algorithm
    void draw()
    {
        int x0 = pos1.x;
        int y0 = pos1.y;
        int x1 = pos2.x;
        int y1 = pos2.y;

        int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;

        for(;;){
            drawPixel(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            e2 = 2*err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }
};

// Rectangle class algorithm
class Rectangle : public GraphElement
{
public:
	// Four corners of rectangle(A,B,C,D) and center point(S)
	Point2D pointA, pointB, pointC, pointD, pointS;
	// Length of first and second side
	int strana1, strana2;
	// Colours of rectangle
	RGB fg, bg;
	// 6 lines representing final rectangle. 4 sides(A,B,C,D) and 2 diagonals(U1, U2)
	Line *stranaA, *stranaB, *stranaC, *stranaD, *stranaU1, *stranaU2;
	Rectangle(Point2D pointS, int strana1, int strana2, RGB fg, RGB bg): pointS(pointS), strana1(strana1), strana2(strana2), GraphElement(fg, bg){
		this->pointA.x = pointS.x - strana1/2;
		this->pointA.y = pointS.y - strana2/2;
		this->pointC.x = pointS.x + strana1/2;
		this->pointC.y = pointS.y + strana2/2;
		this->pointB.x = pointS.x - strana1/2;
		this->pointB.y = pointS.y + strana2/2;
		this->pointD.x = pointS.x + strana1/2;
		this->pointD.y = pointS.y - strana2/2;

		this->stranaA = new Line({pointA.x, pointA.y}, {pointB.x, pointB.y}, fg, bg);
		this->stranaB = new Line({pointA.x, pointA.y}, {pointD.x, pointD.y}, fg, bg);
		this->stranaC = new Line({pointB.x, pointB.y}, {pointC.x, pointC.y}, fg, bg);
		this->stranaD = new Line({pointC.x, pointC.y}, {pointD.x, pointD.y}, fg, bg);
		this->stranaU1 = new Line({pointA.x, pointA.y}, {pointC.x, pointC.y}, fg, bg);
		this->stranaU2 = new Line({pointB.x, pointB.y}, {pointD.x, pointD.y}, fg, bg);
	}
	// Draw function implementation using drawing combination of lines
	void draw()
	{
		stranaA->draw();
		stranaB->draw();
		stranaC->draw();
		stranaD->draw();
		stranaU1->draw();
		stranaU2->draw();
	}
};

// Triangle class implementation TODO, not working right
class Triangle : public GraphElement
{
public:
    // Center of triangle
    Point2D center;    
    // Length of one side of triangle
    int strana;
    // Colours of triangle
    RGB fg, bg;
    // Height of triangle
    double vyska;
    // 
    Point2D point_c1, point_c2, point_c3, point_a1;
    Line *strana_a, *strana_b, *strana_c;

    Triangle( Point2D t_center, int t_strana, RGB t_fg, RGB t_bg ) :
        center( t_center ), strana( t_strana ), GraphElement( t_fg, t_bg )
    {
    	this->fg = t_fg;
    	this->bg = t_bg;

    	this->vyska = sqrt(pow(strana, 2) - pow(strana/ 2.0, 2));
    	this->point_c2 = { center.x, center.y + (vyska / 3) };
    	this->point_c1 = {center.x - (strana / 2), center.y + (vyska / 3)};
    	this->point_c3 = {center.x + (strana / 2), center.y + (vyska / 3)};
    	this->point_a1 = {center.x, center.y - ((vyska / 3) * 2)};

    	this->strana_a = new Line({point_c1.x, point_c1.y}, {point_a1.x, point_a1.y}, fg, bg);
    	this->strana_c = new Line({point_c1.x, point_c1.y}, {point_c3.x, point_c3.y}, fg, bg);
    	this->strana_b = new Line({point_c3.x, point_c3.y}, {point_a1.x, point_a1.y}, fg, bg);
    };

    void draw()
    {
    	strana_c->draw();
    	strana_a->draw();
    	strana_b->draw();
    }
};

// Character class implementation, represents one character from font
class Character : public GraphElement
{
public:
    // Position of character
    Point2D pos;
    // Character represented in char -> used to select right character from fonts file using ASCII value
    char character;

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) :
      pos( t_pos ), character( t_char ), GraphElement( t_fg, t_bg ) {};
      
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
                //if(radek_fontu & (1 << x)) drawPixel(pos.x + x, pos.y + y);	       //LSB
		if(radek_fontu & (1 << x)) drawPixel(pos.x - x + WIDTH, pos.y + y);    //MSB
            }
        }
    }
};

// Text class implementation, very similiar to character class implementation
class Text : public GraphElement
{
public:
    // Position of starting character
    Point2D pos;
    
    // Array of characters (string..)
    string str;
	
    // Flag which represents if text should be written horiontaly or verticaly 
    bool horizontal = false;

    Text( Point2D t_pos, string t_str, RGB t_fg, RGB t_bg, bool horizontal ) :
      pos( t_pos ), str( t_str ), GraphElement( t_fg, t_bg )
        {
            this->horizontal = horizontal;
        };
    
    // Implementation of draw function, similiar to character class draw function. Only difference here is that it 
    // iterates over more characters and adds offset between them.
    void draw()
    {
        int offs = 0;
	
	// Iterating over each character in string
        for (int i = 0; i < str.size(); i++)
        {
	    // Iterating over rows
            for(int y = 0; y < HEIGHT; y++)
            {
	        // Selecting one specific position in fonts file
		int radek_fontu = font[str[i]][y];
		
		// Iterating over characters until we reach the width of character (last value of character)
                for(int x = 0; x < WIDTH; x++)
                {
                    if(horizontal)
                    {
			//if(radek_fontu & (1 << x)) drawPixel(pos.x + x + offs, pos.y + y);           //LSB
                        if(radek_fontu & (1 << x)) drawPixel(pos.x - x + WIDTH + offs, pos.y + y);    //MSB
                    }
                    else
                    {
                        //if(radek_fontu & (1 << x)) drawPixel(pos.x + x, pos.y + y + offs);	      //LSB
			if(radek_fontu & (1 << x)) drawPixel(pos.x - x + WIDTH, pos.y + y + offs);    //MSB
                    }
                }
            }
            offs += offset;
        }
    }
};

// Creation of basic colours
RGB black = {0, 0, 0};
RGB white = {255, 255, 255};
RGB bordo = {128, 0, 32};
RGB cyan = {0, 255, 255};
RGB green = {0, 255, 0};
RGB blue = {0, 0, 255};
RGB red = {255, 0, 0};
RGB deeppink = {255, 20, 147};

int main()
{
	// Serial line initialization
	pc.baud(115200);
	// LCD initialization
 	lcd_init();
	lcd_clear();
	// CODE here
	return 0;
}

