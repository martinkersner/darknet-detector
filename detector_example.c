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

image **load_alphabet2()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "darknet/data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

int main(int argc, char **argv) {
  char *filename   = argv[1]; // video file name
  char *weightfile = argv[2]; // *.weights
  char *cfgfile    = argv[3]; // *.cfg
  char *datafile   = argv[4]; // *.data

  int frame_skip = 0;
  int delay = frame_skip;

  list *options = read_data_cfg(datafile);

  demo_names    = get_labels(option_find_str(options, "names", "darknet/data/names.list"));
  demo_alphabet = load_alphabet2();
  demo_classes  = option_find_int(options, "classes", 20);
  demo_thresh   = 0.4;

  char *prefix  = NULL;
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

  layer l = net.layers[net.n-1]; // last layer
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

  for (j = 0; j < (FRAMES/2)+1; ++j) {
    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;
  }

  int count = 0;
  if (!prefix) {
    cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
    cvMoveWindow("Demo", 0, 0);
    cvResizeWindow("Demo", 1352, 1013);
  }

  double before = get_wall_time();

  while (1) {
    ++count;
    if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
    if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

    if (!prefix) {
      show_image(disp, "Demo");
      cvWaitKey(1);
    }
    else {
      char buff[256];
      sprintf(buff, "%s_%08d", prefix, count);
      save_image(disp, buff);
    }

    pthread_join(fetch_thread, 0);
    pthread_join(detect_thread, 0);

    if (delay == 0) {
      free_image(disp);
      disp = det;
    }

    det   = in;
    det_s = in_s;

    --delay;
    if (delay < 0) {
      delay = frame_skip;

      double after = get_wall_time();
      float curr = 1./(after - before);
      fps = curr;
      before = after;
    }
  }

  return 0;
}
