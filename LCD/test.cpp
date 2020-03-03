// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 08/2016
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Main program for LCD module
//
// **************************************************************************

#include "mbed.h"
#include "lcd_lib.h"
#include <string>
//#include "font8x8.cpp"		//neodkomentovavat
#include "font22x36_msb.h"
#include <vector>

//#define WIDTH 8
//#define HEIGHT 8
#define WIDTH 22
#define HEIGHT 36
#define FONT font
int rozdil = HEIGHT - WIDTH;

// Serial line for printf output
Serial pc(USBTX, USBRX);

// two dimensional array with fixed size font
//extern uint8_t font8x8[256][8];		//odkomenovat u 8x8

using namespace std;

DigitalOut bl(PTC3, 0);		// backlight

int offset = 22;
int dist = 10;
bool sviti = true;
int T = 15;

// Simple graphic interface

struct Point2D
{
    int32_t x, y;
};

struct RGB
{
    uint8_t r, g, b;
};

class GraphElement
{
public:
    // foreground and background color
    RGB fg_color, bg_color;

    // constructor
    GraphElement(){}

    GraphElement( RGB t_fg_color, RGB t_bg_color ) :
        fg_color( t_fg_color ), bg_color( t_bg_color ) {}

    // ONLY ONE INTERFACE WITH LCD HARDWARE!!!
    void drawPixel( int32_t t_x, int32_t t_y ) { lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( fg_color ) ); }

    // Draw graphics element
    virtual void draw() = 0;

    // Hide graphics element
    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }
    //virtual void hideVert() { swap_fg_bg_color(); drawVert(); swap_fg_bg_color(); }
private:
    // swap foreground and backgroud colors
    void swap_fg_bg_color() { RGB l_tmp = fg_color; fg_color = bg_color; bg_color = l_tmp; }

    // conversion of 24-bit RGB color into 16-bit color format
    int convert_RGB888_to_RGB565( RGB t_color )
    {
        union URGB {struct {int b:5; int g:6; int r:5;}; short rgb565; } urgb;
        urgb.r = (t_color.r >> 3) & 0x1F;
        urgb.g = (t_color.g >> 2) & 0x3F;
        urgb.b = (t_color.b >> 3) & 0x1F;
        return urgb.rgb565;
    }
};

class Pixel : public GraphElement
{
public:
    // constructor
    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : pos( t_pos ), GraphElement( t_fg_color, t_bg_color ) {}
    // Draw method implementation
    virtual void draw() { drawPixel( pos.x, pos.y ); }
    // Position of Pixel
    Point2D pos;
};

class Circle : public GraphElement
{
public:
    // Center of circle
    Point2D center;
    // Radius of circle
    int32_t radius;

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) :
        center( t_center ), radius( t_radius ), GraphElement( t_fg, t_bg ) {};

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


class Character : public GraphElement
{
public:
    // position of character
    Point2D pos;
    // character
    char character;

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) :
      pos( t_pos ), character( t_char ), GraphElement( t_fg, t_bg ) {};

    void draw()
    {
        for(int y = 0; y < HEIGHT; y++)
        {
            //int radek_fontu = font8x8[character][y];
			int radek_fontu = font[character][y];
            for(int x = 0; x < WIDTH; x++)
            {
                //if(radek_fontu & (1 << x)) drawPixel(pos.x + x, pos.y + y);		//LSB
				if(radek_fontu & (1 << x)) drawPixel(pos.x - x + WIDTH, pos.y + y);    //MSB
            }
        }
    }
};

class Line : public GraphElement
{
public:
    // the first and the last point of line
    Point2D pos1, pos2;

    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) :
      pos1( t_pos1 ), pos2( t_pos2 ), GraphElement( t_fg, t_bg ) {}

    void draw()
    {
        int x0 = pos1.x;
        int y0 = pos1.y;
        int x1 = pos2.x;
        int y1 = pos2.y;

        int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2; //error value e_xy

        for(;;){  //loop
            drawPixel(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            e2 = 2*err;
            if (e2 >= dy) { err += dy; x0 += sx; } //e_xy+e_x > 0
            if (e2 <= dx) { err += dx; y0 += sy; } //e_xy+e_y < 0
        }
    }
};


class Text : public GraphElement
{
public:
    // position of character
    Point2D pos;
    // characters
    string str;

    bool horizontal = false;

    Text( Point2D t_pos, string t_str, RGB t_fg, RGB t_bg, bool horizontal ) :
      pos( t_pos ), str( t_str ), GraphElement( t_fg, t_bg )
        {
            this->horizontal = horizontal;
        };

    void draw()
    {
        int offs = 0;
        for (int i = 0; i < str.size(); i++)
        {
            for(int y = 0; y < HEIGHT; y++)
            {
                //int radek_fontu = font8x8[str[i]][y];
				int radek_fontu = font[str[i]][y];
                for(int x = 0; x < WIDTH + rozdil; x++)
                {
                    if(horizontal)
                    {
						//if(radek_fontu & (1 << x)) drawPixel(pos.x + x + offs, pos.y + y);           //LSB
                        if(radek_fontu & (1 << x)) drawPixel(pos.x+x+WIDTH+offs, pos.y+y);    //MSB
                    }
                    else
                    {
                        //if(radek_fontu & (1 << x)) drawPixel(pos.x + x, pos.y + y + offs);			//LSB
						if(radek_fontu & (1 << x)) drawPixel(pos.x + x, pos.y + y + offs);    //MSB
                    }
                }
            }
            offs += offset;
        }
    }
};

class Rectangle : public GraphElement
{
public:
	Point2D pointA, pointB, pointC, pointD, pointS;
	int strana1, strana2;
	RGB fg, bg;
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

class Triangle : public GraphElement
{
public:
    // Center of triangle
    Point2D center;
    // Radius of triangle
    int strana;

    RGB fg, bg;
    double vyska;
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
 	lcd_init();				// LCD initialization
	lcd_clear();			// LCD clear screen

	int l_color_red = 0xF800;
	int l_color_green = 0x07E0;
	int l_color_blue = 0x001F;
	int l_color_white = 0xFFFF;

	Point2D centerPoints;
	centerPoints.x = 160;
	centerPoints.y = 120;

	Pixel center(centerPoints, white, black);
	center.draw();
	Rectangle kokotek(centerPoints, 200,75,white,black);
	kokotek.draw();

	//    Text( Point2D t_pos, string t_str, RGB t_fg, RGB t_bg, bool horizontal ) :
	Text randomText(centerPoints,"ahoj", cyan, black, true);
	randomText.draw();

	return 0;
}

