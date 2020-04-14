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

int bytes[13];

void print_array(int *array, int length){
    for (int i = 0; i < length; i++)
    {
        printf("%d ", array[i]);
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
    int count = 0;
    while ( rds.get_rds_status( l_ptr ) > 0 )
    {
        count++;
        printf("Reading...\n");
        print_array(bytes, 13);
    }
    printf("End...\n");
    printf("Lines: %d\n", count);

    rds.close_file();
}
