/* test_front_block.c at 48 kHz / fft 1024 (hop 512): the front feeding the Align-ULCNet block at N = 1 and N = 2 with
 * the test catalog. The scene delay is the 16 kHz scene's 480 samples scaled to 48 kHz. */

#define BLK_SR 48000
#define BLK_FFT 1024
#define BLK_PN 2
#define BLK_BLOCKS 240
#define BLK_DELAY 1440

#include "test_front_block.c"
