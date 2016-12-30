#ifndef IMAGE_H_
#define IMAGE_H_

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "darknet/src/image.h"

image ipl_to_image2(IplImage*);
image get_image_from_stream2(CvCapture*);

#endif // IMAGE_H_
