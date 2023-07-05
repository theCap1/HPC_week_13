#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "mini_hexa.h"

#include <stddef.h>
#include <sys/time.h>
#include "kernels/gemm_ref.h"
#include <qurt.h>
#include <stdint.h>

int32_t simple_function_asm( int32_t i_value );

int32_t micro_hvx_qf32( int32_t i_n_repetitions );

void micro_hvx_qf32_wrapped( void * args ) {
  // TODO: call microkernel
  qurt_thread_exit( QURT_EOK );
}

void gemm_asm_cdsp_192_4_128( float const * i_a,
                              float const * i_b,
                              float       * io_c );

int mini_hexa_open(const char*uri, remote_handle64* handle) {
  * handle = 0;
  return 0;
}

/**
 * @param handle, the value returned by open
 * @retval, 0 for success, should always succeed
 */
int mini_hexa_close( remote_handle64 handle ) {
   if( handle ) {
      free( (void*) handle );
   }
   return 0;
}

double get_walltime( void ) {
  struct timeval l_tp;
  gettimeofday( &l_tp, NULL );
  double l_time = (double) ( l_tp.tv_sec + l_tp.tv_usec/1000000.0 );

  return l_time;
}

/**
 * Runs a set of microbenchmarks and reports performance using FARF. 
 **/
int mini_hexa_micro( remote_handle64 i_h ) {
  FARF( ALWAYS, "=============== running mini_hexa_micro ===============" );

  int l_value = simple_function_asm( 5 );
  FARF( ALWAYS, "  simple_function_asm( 5 ): %i", l_value);

  int64_t l_n_repetitions = 0;
  int64_t l_n_flops = 0;
  double l_tp0 = 0;
  double l_tp1 = 0;
  double l_dur = 0;
  double l_gflops = 0;

  /*
   * micro_hvx_qf32
   */
  l_n_repetitions = 100000000;
  l_tp0 = get_walltime();
  l_n_flops = micro_hvx_qf32( l_n_repetitions );
  l_tp1 = get_walltime();

  l_dur = l_tp1 - l_tp0;

  FARF( ALWAYS, "  duration micro_hvx_qf32: %f\n", l_dur );
  l_n_flops *= l_n_repetitions;
  l_gflops = l_n_flops * 1.0E-9;
  l_gflops = l_gflops / l_dur;
  FARF( ALWAYS, "  GFLOPS micro_hvx_qf32: %f\n", l_gflops );

  // threaded execution
  int64_t l_n_threads = 4;

  qurt_thread_t l_tid[l_n_threads];
  int l_thread_exit_status[l_n_threads];
  qurt_thread_attr_t l_attr[l_n_threads];
  void * l_thread_stack_addr[l_n_threads];
  uint64_t l_stack_size = 1024*4;

  for( int64_t l_td = 0; l_td < l_n_threads; l_td++ ) {
    l_thread_stack_addr[l_td] = malloc(l_stack_size);
    assert( l_thread_stack_addr[l_td] != NULL );
  
    qurt_thread_attr_init( l_attr+l_td );
    qurt_thread_attr_set_name( l_attr+l_td,
                               (char *) "my_thread_name" );
    qurt_thread_attr_set_stack_addr( l_attr+l_td,
                                     l_thread_stack_addr[l_td]);
    qurt_thread_attr_set_stack_size( l_attr+l_td,
                                     l_stack_size );
    qurt_thread_attr_set_priority( l_attr+l_td,
                                   QURT_THREAD_ATTR_PRIORITY_DEFAULT/2 );
  }

  l_tp0 = get_walltime();

  for( int64_t l_td = 0; l_td < l_n_threads; l_td++ ) {
    FARF( ALWAYS, "  spawning thread %i", l_td );
    int l_retcode = qurt_thread_create( l_tid+l_td,
                                        l_attr+l_td,
                                        micro_hvx_qf32_wrapped,
                                        (void *) &l_n_repetitions );
    assert( l_retcode == QURT_EOK );
  }
  FARF( ALWAYS, "  finished spawning %i threads", l_n_threads );

  for( int64_t l_td = 0; l_td < l_n_threads; l_td++ ) {
    FARF( ALWAYS,   "  waiting for thread %i", l_td );
    int l_status = qurt_thread_join( l_tid[l_td],
                                     l_thread_exit_status+l_td );
    assert(( l_status==QURT_EOK) || (l_status==QURT_ENOTHREAD));

    free( l_thread_stack_addr[l_td] );
  }

  l_tp1 = get_walltime();
  l_dur = l_tp1 - l_tp0;

  FARF( ALWAYS, "  duration micro_hvx_qf32 (threaded): %f\n", l_dur );
  l_n_repetitions = 100000000; // assumed in wrapper
  l_n_flops = 15*64;
  l_n_flops *= l_n_repetitions;
  l_n_flops *= l_n_threads;
  l_gflops = l_n_flops * 1.0E-9;
  l_gflops = l_gflops / l_dur;
  FARF( ALWAYS, "  GFLOPS micro_hvx_qf32 (threaded): %f\n", l_gflops );

  FARF( ALWAYS, "=============== finished mini_hexa_micro ===============" );

  return 0;
}

/**
 * Single precision (qfloat accumulator) matrix-matrix multiplication.
 **/
int mini_hexa_sgemm( remote_handle64 i_h ) {
  FARF( ALWAYS, "=============== running mini_hexa_sgemm ===============" );

  // allocate memory
  int l_size = 128*128;
  float * l_a = malloc( l_size * sizeof(float) );
  float * l_b = malloc( l_size * sizeof(float) );
  float * l_c = malloc( l_size * sizeof(float) );
  float * l_c_ref = malloc( l_size * sizeof(float) );

  double l_tp0 = 0;
  double l_tp1 = 0;
  double l_dur = 0;
  double l_gflops = 0;
  int64_t l_n_repetitions = 0;
  double l_max_err = 0;

  // init data
  srand48( time(NULL) );
  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_a[l_id] = (float) drand48();
  }
  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_b[l_id] = (float) drand48();
  }
  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_c[l_id] = (float) drand48();
  }
  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_c_ref[l_id] = l_c[l_id];
  }

  /*
   * CDSP: 192, 4, 1
   */
  // TODO: finish implementation


  /*
   * CDSP: 192, 4, 128
   */

  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_c[l_id] = (float) drand48();
  }
  for( int64_t l_id = 0; l_id < l_size; l_id++ ) {
    l_c_ref[l_id] = l_c[l_id];
  }

  l_n_repetitions = 100000;


  FARF( ALWAYS, "testing gemm_asm_cdsp_192_4_128 kernel" );

  // run reference implementation
  gemm_ref_mnk( l_a,
                l_b,
                l_c_ref,
                192,
                4,
                128,
                192,
                128,
                192 );

  // run assembly kernel
  FARF( ALWAYS, "  calling gemm_asm_cdsp_192_4_128" );

  gemm_asm_cdsp_192_4_128( l_a,
                           l_b,
                           l_c );

  l_max_err = 0;
  for( int64_t l_m = 0; l_m < 192; l_m++ ) {
    for( int64_t l_n = 0; l_n < 4; l_n++ ) {
      double l_err = l_c[l_n*192+l_m] - l_c_ref[l_n*192+l_m];
      l_err = l_err < 0 ? -l_err : l_err;
      l_max_err = l_max_err > l_err ? l_max_err : l_err;
    }
  }

  FARF( ALWAYS, "  max error: %f", l_max_err );

  // time asm kernel
  l_tp0 = get_walltime();
  for( int64_t l_re = 0; l_re < l_n_repetitions; l_re++ ) {
    gemm_asm_cdsp_192_4_128( l_a,
                             l_b,
                             l_c );
  }
  l_tp1 = get_walltime();

  l_dur = l_tp1 - l_tp0;

  FARF( ALWAYS, "  duration gemm_asm_cdsp_192_4_128: %f seconds\n", l_dur );

  l_gflops  = l_n_repetitions;
  l_gflops *= 192 * 4 * 128 * 2;
  l_gflops *= 1.0E-9;
  l_gflops /= l_dur;
  FARF( ALWAYS, "  GFLOPS gemm_asm_cdsp_192_4_128: %f\n", l_gflops );

  free( l_a );
  free( l_b );
  free( l_c );
  free( l_c_ref );

  FARF( ALWAYS, "=============== finished mini_hexa_sgemm ===============" );


  return 0;
}
