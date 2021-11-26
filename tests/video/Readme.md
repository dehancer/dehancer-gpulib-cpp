### Forward/Reverse playback OpneCV

###
```cpp
int main (int argc, char* argv[])
{
  cv::VideoCapture cap(argv[1]);

  double frame_rate = cap.get(CV_CAP_PROP_FPS);

  // Calculate number of msec per keyframe.
  // (msec/sec / frames/sec = msec/keyframe)
  double frame_msec = 1000 / frame_rate;

  // Seek to the end of the video.
  cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);

  // Get video length (because we're at the end).
  double video_time = cap.get(CV_CAP_PROP_POS_MSEC);

  cv::Mat keyframe;
  cv::namedWindow("window");

  while (video_time > 0)
  {
    // Decrease video time by number of msec in one keyframe
    // and seek to the new time.
    video_time -= frame_msec;
    cap.set(CV_CAP_PROP_POS_MSEC, video_time);

    // Grab the keyframe and display it.
    cap >> keyframe;
    cv::imshow("window", keyframe);

    // Necessary for opencv's event loop to work.
    // Wait for the length of one keyframe before
    // continuing the loop. Exit if the user presses
    // any key. If you want the video to play faster
    // or slower, adjust the parameter accordingly.    
    if (cv::waitKey(frame_msec) >= 0)
      break;
  }
}
```