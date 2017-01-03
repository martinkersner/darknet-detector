// Martin Kersner, 2016/12/28
// 2016/12/28

#include "darknet/src/network.h"
#include "darknet/src/detection_layer.h"
#include "darknet/src/region_layer.h"
#include "darknet/src/cost_layer.h"
#include "darknet/src/utils.h"
#include "darknet/src/parser.h"
#include "darknet/src/box.h"
#include "darknet/src/image.h"
#include "darknet/src/option_list.h"
#include <sys/time.h>

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#define FRAMES 3

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float demo_thresh = 0;
static int demo_index = 0;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;

static float *predictions[FRAMES];
static image images[FRAMES];
static float *avg;
