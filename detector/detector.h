#ifndef DETECTOR_H
#define DETECTOR_H

//#include <fstream>

#include "../config.h"
#include "../net/bnet.h"
#include "../net/pnet.h"
#include "../net/rnet.h"
#include "../net/onet.h"
#include "image.h"

using namespace std;
using namespace caffe;

using std::string;

class Detector {
  public:
        Detector(  const string& pnet_model_file,
                   const string& pnet_trained_file,
                   const string& rnet_model_file,
                   const string& rnet_trained_file,
                   const string& onet_model_file,
                   const string& onet_trained_file,
                   const string& image_name);
  
        // Main Function
        const cv::Mat Detect(Image& img);
  
  private:

        // General Functions      
        vector<int> nms                 (std::vector <box>total_boxes, 
                                         float threshold, 
                                         bool  type); // 0 = "Union", 1 = "min"
        vector<box> generateBoundingBox (std::vector< std::vector <float>> data,
                                         std::vector<int> shape_map,
                                         float scale,
                                         float threshold);
        void padBoundingBox             (int imgHeight, int imgWidth);
        void writeOutputImage           (const cv::Mat& image);

        // Net Wrapper Functions
        void pnetWrapper (Image& img);        
        void rnetWrapper (Image& img);        
        void onetWrapper (Image& img); 
        
        // Debug functions    
        void printCurrentOutputs (const char* folder_name, const cv::Mat& image);
        void rnetCreateInputTest ();    
        void onetCreateInputTest ();
        
  private:
        PNet pnet;
        RNet rnet;
        ONet onet;
       
        // Definitions 
        int minSize;
        float factor;

        // Threshold to consider points as potential
        // candidates. 1 for each NN.
        float thresholds[3];
        float nms_thresholds[3];
};

#endif
