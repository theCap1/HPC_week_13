#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "rpcmem.h"
#include "pd_status_notification.h"
#include "dsp_capabilities_utils.h"
#include "mini_hexa.h"

/**
 * Register status notifcations of the DSP process.
 * Details: User guide under ipc/rpc.html, chapter "Status notifications of DSP User process".
 **/
int fastrpc_notif_dsp( void                      * i_context,
                       int                         i_domain,
                       int                         i_session,
                       remote_rpc_status_flags_t   i_status ){
    int l_err = AEE_SUCCESS;

    if( i_status == FASTRPC_USER_PD_UP ) {
      printf( "  dsp user process is up\n");
    }
    else if( i_status == FASTRPC_USER_PD_EXIT ) {
      printf("  dsp user process exited\n");
    }
    else if( i_status == FASTRPC_USER_PD_FORCE_KILL ) {
      printf("  dsp user process forcefully kill\n");
    }
    else if( i_status == FASTRPC_USER_PD_EXCEPTION ) {
      printf("  exeception occurred in dsp user process\n");
    }
    else if( i_status == FASTRPC_DSP_SSR ) {
      printf("  subsystem restart of the dsp running user process\n");
    }
    else {
      l_err = AEE_EBADITEM;
    }
    return l_err;
}

int main() {
  // disable buffering of output
  setbuf( stdout,
          NULL );

  printf( "*************************\n" );
  printf( "* welcome to mini_hexa! *\n" );
  printf( "*************************\n" );

  // error code
  int l_err = 0;

  // run on cDSP
  int l_domain_id = 3;

  // utils/examples/dsp_capabilities_utils.h
  bool l_valid_domain_id = is_valid_domain_id( l_domain_id,
                                               0 );
  assert( l_valid_domain_id );

  // check that we may run unsigned using unsigned user protection domain
  // utils/examples/dsp_capabilities_utils.h
  bool l_unsignedpd_supported = is_unsignedpd_supported( l_domain_id );
  assert( l_unsignedpd_supported );

  // get domain struct from domain value
  // utils/examples/dsp_capabilities_utils.h
  domain * l_domain = get_domain( l_domain_id );
  assert( l_domain != NULL );
  printf( "running on domain: %d\n", l_domain_id );

  // set remote session parameters
  // incs/remote.h
  struct remote_rpc_control_unsigned_module l_data;
  l_data.domain = l_domain_id;
  l_data.enable = 1;

  l_err = remote_session_control( DSPRPC_CONTROL_UNSIGNED_MODULE, // request id
                                  (void *) &l_data,               // address of structure
                                  sizeof( l_data ) );             // length of data
  assert( l_err == AEE_SUCCESS );

  // assemble uniform resource identifier
  int l_mini_hexa_uri_domain_len = strlen( mini_hexa_URI ) + MAX_DOMAIN_URI_SIZE;

  char * l_mini_hexa_uri_domain = (char *) malloc( l_mini_hexa_uri_domain_len );
  assert( l_mini_hexa_uri_domain != NULL );

  l_err = snprintf( l_mini_hexa_uri_domain,
                    l_mini_hexa_uri_domain_len,
                    "%s%s",
                    mini_hexa_URI,
                    l_domain->uri );
  assert( l_err >= 0 );

  printf( "mini_hexa_URI_domain: %s\n",
           l_mini_hexa_uri_domain );

  // enable status notifications from client protection domain
  // utils/examples/pd_status_notification.h
  l_err = request_status_notifications_enable( l_domain_id,
                                               (void*) 0x12345678,
                                               fastrpc_notif_dsp );
  assert( l_err == AEE_SUCCESS );

  /*
   * Microbenchmarks
   */
  printf( "running micro\n" );

  // open handle in the mini_hexa domain
  remote_handle64 l_handle_micro = -1;
  l_err = mini_hexa_open( l_mini_hexa_uri_domain,
                          &l_handle_micro );
  assert( l_err == AEE_SUCCESS );

  // call device function
  printf("  calling device function mini_hexa_micro\n");
  l_err = mini_hexa_micro( l_handle_micro );
  assert( l_err == AEE_SUCCESS );

  // close handle
  l_err = mini_hexa_close( l_handle_micro);
  assert( l_err == AEE_SUCCESS );

  printf( "finished micro\n" );

  /*
   * sgemm
   */
  printf( "running sgemm\n" );

  // open handle
  remote_handle64 l_handle_sgemm = -1;
  l_err = mini_hexa_open( l_mini_hexa_uri_domain,
                          &l_handle_sgemm );
  assert( l_err == AEE_SUCCESS );

  // call device function
  printf("  calling mini_hexa_sgemm on the DSP\n");
  l_err = mini_hexa_sgemm( l_handle_sgemm );
  assert( l_err == AEE_SUCCESS );

  // close handle
  l_err = mini_hexa_close( l_handle_sgemm );
  assert( l_err == AEE_SUCCESS );

  printf( "finished sgemm\n" );

  // release memory of mini_hexa domain
  if( l_mini_hexa_uri_domain ) {
    free( l_mini_hexa_uri_domain );
  }

  printf( "******************************\n" );
  printf( "* finished running mini_hexa *\n" );
  printf( "******************************\n" );

  return EXIT_SUCCESS;
}
