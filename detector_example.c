// Martin Kersner, 2016/12/28
// 2016/12/28

#include "detector_example.h"
#include "detector_image.h"

void *fetch_in_thread(void *ptr)
{
    in = get_image_from_stream2(cap);
    if(!in.data){
        error("Stream closed.");
    }
    in_s = resize_image(in, net.w, net.h);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
    demo_index = (demo_index + 1)%FRAMES;

    draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

    return 0;
}

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char **argv) {
  int frame_skip = 0;
  char **names = NULL;
  int classes = 21;
  float thresh = 0.4;

  image **alphabet = load_alphabet();
  int delay = frame_skip;
  demo_names = names;
  demo_alphabet = alphabet;
  demo_classes = classes;
  demo_thresh = thresh;

  char *filename   = "/home/martin/DeepLearning/BeaverDam/annotator/static/videos/8.mp4";
  char *weightfile = "/home/martin/github/darknet/yolo.weights";
  char *cfgfile    = "/home/martin/github/darknet/cfg/yolo.cfg";
  char *prefix     = "";

  int cam_index = 0;

  net = parse_network_cfg(cfgfile);

  if (weightfile) {
    load_weights(&net, weightfile);
  }

  set_batch_network(&net, 1);
  srand(2222222);

  if (filename) {
    printf("video file: %s\n", filename);
    cap = cvCaptureFromFile(filename);
  } else {
    cap = cvCaptureFromCAM(cam_index);
  }

  if(!cap) error("Couldn't connect to webcam.\n");

  layer l = net.layers[net.n-1];
  int j;

  avg = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

  boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

  pthread_t fetch_thread;
  pthread_t detect_thread;

  fetch_in_thread(0);
  det = in;
  det_s = in_s;

  fetch_in_thread(0);
  detect_in_thread(0);
  disp = det;
  det = in;
  det_s = in_s;

  for(j = 0; j < FRAMES/2; ++j){
    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;
  }

  int count = 0;
  if(!prefix){
    cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
    cvMoveWindow("Demo", 0, 0);
    cvResizeWindow("Demo", 1352, 1013);
  }

  double before = get_wall_time();

  while(1){
    ++count;
    if(1){
      if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
      if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

      if(!prefix){
        show_image(disp, "Demo");
        int c = cvWaitKey(1);
        if (c == 10){
          if(frame_skip == 0) frame_skip = 60;
          else if(frame_skip == 4) frame_skip = 0;
          else if(frame_skip == 60) frame_skip = 4;   
          else frame_skip = 0;
        }
      }else{
        char buff[256];
        sprintf(buff, "%s_%08d", prefix, count);
        save_image(disp, buff);
      }

      pthread_join(fetch_thread, 0);
      pthread_join(detect_thread, 0);

      if(delay == 0){
        free_image(disp);
        disp  = det;
      }
      det   = in;
      det_s = in_s;
    }else {
      fetch_in_thread(0);
      det   = in;
      det_s = in_s;
      detect_in_thread(0);
      if(delay == 0) {
        free_image(disp);
        disp = det;
      }
      show_image(disp, "Demo");
      cvWaitKey(1);
    }
    --delay;
    if(delay < 0){
      delay = frame_skip;

      double after = get_wall_time();
      float curr = 1./(after - before);
      fps = curr;
      before = after;
    }
  }

  return 0;
}
