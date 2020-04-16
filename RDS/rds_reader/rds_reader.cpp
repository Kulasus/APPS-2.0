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

char station[8];
char radiotext[65];
int hours = 0;
int minutes = 0;
int bytes[13];

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

    printf("Start...\n");
    void *l_ptr = malloc(FM_RDS_STATUS_ANS_LEN);
    int groups_count = 0;
    int good_groups_count = 0;
    int bad_groups_count = 0;
    int b_groups_count = 0;
    int a_groups_count = 0;
    int group0a_count = 0;
    int group2a_count = 0;
    int group4a_count = 0;
    int group0b_count = 0;
    while ( rds.get_rds_status( l_ptr ) == FM_RDS_STATUS_ANS_LEN)
    {
        usleep(2000);
        //print_array(bytes, 13);
        printf("\n station: %s ",station);
        printf("| radiotext: %s ",radiotext);
        if(minutes < 10){
            printf("| time: %d:0%d",hours,minutes);
        }
        else
        {
            printf("| time: %d:%d",hours,minutes);
        }
        groups_count++;
        if((bytes[1] & 2) == 2 || (bytes[12] & 3) == 3 || (bytes[12] & 12) == 12 || (bytes[12] & 48) == 12 || (bytes[12] & 192) == 12){
            bad_groups_count++;
        }
        else{
            good_groups_count++;
        }
        // Group 0A logic STATION
        if((bytes[6] & 248) == 0){
            group0a_count++;
            if((bytes[7] & 3) == 0){
                station[0] = bytes[10];
                station[1] = bytes[11];
            }
            else if((bytes[7] & 3) == 1){
                station[2] = bytes[10];
                station[3] = bytes[11];
            }
            else if((bytes[7] & 3) == 2){
                station[4] = bytes[10];
                station[5] = bytes[11];
            }
            else if((bytes[7] & 3) == 3){
                station[6] = bytes[10];
                station[7] = bytes[11];
            }
        }
        // Group 2A logic RADIOTEXT
        else if((bytes[6] & 248) == 32){
            group2a_count++;
            int index = 0;
            for (int i = 0; i < 16; i++)
            {
                if((bytes[7] & 15) == i){
                    radiotext[index]=bytes[8];
                    radiotext[index+1]=bytes[9];
                    radiotext[index+2]=bytes[10];
                    radiotext[index+3]=bytes[11];
                }
                index+=4;
            }
        }
        // Group 4A logic TIME
        else if((bytes[6] & 248) == 64){
            group4a_count++;
            hours = 0;
            minutes = 0;
            if(bytes[9] & 1 == 1){
                hours+=16;
            }
            bool positive = bytes[11] & 32 != 32;
            float halfs = (bytes[11] & 31) % 2;
            if(halfs > 0){
                minutes+=30;
                hours += ((bytes[11] & 31) - 1) / 2;
            } 
            else
            {
                hours += (bytes[11] & 31) / 2;   
            }
            int pom = bytes[10];
            minutes += (bytes[10] << 2) & 60;
            hours += (pom >> 4);
            minutes += bytes[11] >> 6 & 3;
        }
    }
    printf("\nEnd...\n");
    printf("Total groups received: %d\n", groups_count);
    printf("good groups: %d\n", good_groups_count);
    printf("bad groups: %d\n", bad_groups_count);
    printf("0a groups: %d\n", group0a_count);   
    printf("2a groups: %d\n", group2a_count);  
    printf("4a groups: %d\n", group4a_count);  

    rds.close_file();
}
