// ***********************************************************************
//
// Demo program for subject Computer Architectures and Paralel systems
// Petr Olivka, Dept. of Computer Science, FEECS, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example for reading of captured RDS data from FM Radio. (04/2020)
//
// ***********************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define stationLength 8
#define radioTextLength 65
#define picodesLength 10

int picodesCount = 0;
char station_old[stationLength];
char station[stationLength];
char radiotext[radioTextLength];
int hours = 0;
int minutes = 0;
int picode = 0;
int picode_old = 0;
int bytes[13];
int picodes[10];

bool picodes_contains(int picode){
    for (int i = 0; i < picodesLength; i++)
    {
        if(picodes[i] == picode){
            return true;
        }
    }
    return false;
}

void print_array(int *array, int length){
    for (int i = 0; i < length; i++)
    {
        printf("%X ", array[i]);
    }
    printf("\n");
}

// class RDSReader for reading captured RDS data from FM radio
class RDSReader
{
  public:
    // constructor without argumets
    RDSReader()
    {
        m_data_file = nullptr;
        m_file_name[ 0 ] = 0;
    }

    // destructor
    ~RDSReader()
    {
        close_file();
    }

    // Open file with captured data, argument is file name
    int open_file( const char *t_file_name )
    {
        if ( !t_file_name || !t_file_name[ 0 ] )
        {
            fprintf( stderr, "No file name specified!\n" );
            return -1;
        }
        strcpy( m_file_name, t_file_name );
        return reopen_file();
    }

    // Reopen file
    int reopen_file()
    {
        close_file();

        m_data_file = fopen( m_file_name, "r" );
        if ( !m_data_file ) 
        {
            fprintf( stderr, "Unable to open data file '%s'!\n", m_file_name );
            return -1;
        }
        return 0;
    }

    // Close file
    int close_file()
    {
        if ( m_data_file ) 
        {
            fclose( m_data_file );
            m_data_file = nullptr;
            return 0;
        }
        return -1;
    }

    // Method get_rds_status returns 13 bytes read by FM radio command 
    // 0x24 - FM_RDS_STATUS.
    // Data is returned in the same order as it is specified in datasheet.
    // Argument is void to be possible use any data type.
    // 00:00:00 81 35 01 02 22 05 05 41 E8 23 49 4F 00
    // skipchars------------------data----------------
    int get_rds_status( void *t_data )
    {
        // check if emtpy
        if ( !t_data || !m_data_file ) return -1;

        char *l_data = ( char * ) t_data;
        char l_file_line[ 1024 ];

        // end of file
        if ( !fgets( l_file_line, sizeof( l_file_line ), m_data_file ) ) return 0;

        // first eight chars on line
        int l_skip_chars = 8;
        int index = 0;
        while ( 1 )
        {
            int l_tmp, l_read;
            if ( sscanf( l_file_line + l_skip_chars, "%X%n", &l_tmp, &l_read ) < 1 ) break;
            //adds byte on index in array of bytes
            bytes[index] = l_tmp;
            index++;
            * ( l_data++ ) = l_tmp;
            l_skip_chars += l_read;
        }
        return l_data - ( char * ) t_data;
    }

  protected:
    FILE *m_data_file;
    char m_file_name[ 1024 ];
};

bool change_picode(int* array){
    picode_old = picode;
    picode = (array[4] << 8) + array[5];
    if(picode_old != picode){
        for (int i = 0; i < stationLength; i++)
        {
            station[i] = '.';
        }
        for (int i = 0; i < radioTextLength; i++)
        {
            radiotext[i] = '.';
        }
        return true;
    }
    return false;
}

bool check_data(int* array){
    if((array[1] & 2) == 2 || 
    (array[12] & 3) == 3 || 
    (array[12] & 12) == 12 || 
    (array[12] & 48) == 48 || 
    (array[12] & 192) == 192 ||
    (array[1] & 1) != 1 ||
    (array[2] & 4) == 4 ||
    (array[3] & 255) == 0)
    {
        return true;
    }
    return false;
}

void do_group0A(int* array){
    switch ((array[7] & 3))
    {
    case 0:
        station_old[0] = station[0];
        station_old[1] = station[1];
        station[0] = bytes[10];
        station[1] = bytes[11];
        break;
    case 1:
        station_old[2] = station[2];
        station_old[3] = station[3];
        station[2] = bytes[10];
        station[3] = bytes[11];
        break;
    case 2:
        station_old[4] = station[4];
        station_old[5] = station[5];
        station[4] = bytes[10];
        station[5] = bytes[11];        
        break;
    case 3:
        station_old[6] = station[6];
        station_old[7] = station[7];
        station[6] = bytes[10];
        station[7] = bytes[11];
        break;
    default:
        break;
    }
}

void do_group2A(int* array){
    for (int i = 0; i < 16; i++)
    {
        if((array[7] & 15) == i){
            radiotext[i*4]=array[8];
            radiotext[i*4+1]=array[9];
            radiotext[i*4+2]=array[10];
            radiotext[i*4+3]=array[11];
        }
    }
}


void do_group4A(int* array){
    hours = 0;
    minutes = 0;
    if((array[9] & 1) == 1){
        hours+=16;
    }
    bool positive = (array[11] & 32) != 32;
    float halfs = (array[11] & 31) % 2;
    if(halfs > 0){
        minutes+=30;
        if(positive){
            hours += ((array[11] & 31) - 1) / 2;
        }
        else{
            hours -= ((array[11] & 31) - 1) / 2;
        }
    } 
    else
    {
        if(positive){
            hours += ((array[11] & 31) / 2);
        }
        else{
            hours -= ((array[11] & 31) / 2);
        }
    }
    int pom = array[10];
    minutes += (array[10] << 2) & 60;
    hours += (pom >> 4);
    minutes += array[11] >> 6 & 3;
}

void print_info(){
    if((hours<10) && (minutes<10)){
        printf( "Cas: 0%d:0%d | Vysila stanice %04X | Nazev: %s | Radiotext: %s\n" , hours, minutes, picode, station, radiotext); 
    }
    else if((hours<10) && (minutes > 10)){
        printf( "Cas: 0%d:%d | Vysila stanice %04X | Nazev: %s | Radiotext: %s\n", hours, minutes, picode, station, radiotext); 
    }
    else if((hours>10) && (minutes < 10)){
        printf( "Cas: %d:0%d | Vysila stanice %04X | Nazev: %s | Radiotext: %s\n", hours, minutes, picode, station, radiotext); 
    }
    else{
        printf( "Cas: %d:%d | Vysila stanice %04X | Nazev: %s | Radiotext: %s|n", hours, minutes, picode, station, radiotext); 
    }
}
void print_info_short(){
    printf("PICODE: %04X | STATION: %s\n",picode, station);
}

bool check_station(){
    for (int i = 0; i < stationLength; i++)
    {
        if(station[i] != station_old[i]){
            return true;
        }
    }
    return false;
}
bool check_station2(){
    for (int i = 0; i < stationLength; i++)
    {
        if(station[i] == '.'){
            return false;
        }
    }
    return true;
}

#define FM_RDS_STATUS_ANS_LEN   13

int main( int t_argn, char **t_argc )
{
    if ( t_argn < 2 )
    {
        printf( "Usage: rds.dat\n" );
        exit( 1 );
    }

    RDSReader rds;
    rds.open_file( t_argc[ 1 ] );
    void *l_ptr = malloc(FM_RDS_STATUS_ANS_LEN);
    bool picode_changed = false;

    //printf("Start...\n");
    while ( rds.get_rds_status( l_ptr ) == FM_RDS_STATUS_ANS_LEN)
    {
        //usleep(2000);

        //Overeni platnosti dat
        if(check_data(bytes)){
            continue;
        }

        //Zmeni PI code
        picode_changed = change_picode(bytes);
        if(picode_changed){
            while (rds.get_rds_status( l_ptr ) == FM_RDS_STATUS_ANS_LEN)
            {
                //Detekuje skupinu 0A a zpracuje ji
                if((bytes[6] & 248) == 0){
                    do_group0A(bytes);
                }

                //Detekuje skupinu 2A a zpracuje ji
                else if((bytes[6] & 248) == 32){
                    do_group2A(bytes);
                }

                //Detekuje skupinu 4A a zpracuje ji
                else if((bytes[6] & 248) == 64){
                    do_group4A(bytes);
                }
                // Vypise picode a stanici jen pokud dany picode uz nebyl
                if(station[stationLength-1] != '.' && check_station() && check_station2()){
                    if(!picodes_contains(picode)){
                        picodes[picodesCount] = picode;
                        picodesCount++;
                        print_info_short();
                        break;
                    }
                }
            }
        }
    }

    //printf("\nEnd...\n");

    rds.close_file();
}
