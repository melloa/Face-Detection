#ifndef CONFIG_H
#define CONFIG_H

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <numeric>
#include <iostream>

typedef struct {
       cv::Point2f p1;
       cv::Point2f p2;
       float score;
       cv::Point2f dP1;
       cv::Point2f dP2;
} box;

typedef struct {
       cv::Point2f LE;
       cv::Point2f RE;
       cv::Point2f N;
       cv::Point2f LM;
       cv::Point2f RM;
} landmark;

#endif
