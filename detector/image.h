#ifndef IMAGE_H
#define IMAGE_H

#include "../config.h"

using namespace std;
using namespace caffe;

using std::string;

class Image {
  public:
        Image(string fn, cv::Mat img)
        {
            filename = fn;
            image = cv::imread(fn, -1);
        }

        cv::Mat get() 
        { 
            return image; 
        }

        bool empty() { return img.empty(); }

        void print() 
        {
            cout << "Image - " << filename << endl;
                    
            for(int i = 0; i < bounding_boxes.size(); i++)
            {
                cout << "\tBounding Box:" << endl
                    << "\t\t" << bounding_boxes[i].P1 << endl
                    << "\t\t" << bounding_boxes[i].P2 << endl
                    << "\t\t" << bounding_boxes[i].Score << endl
                    << "\t\t" << bounding_boxes[i].dP1 << endl
                    << "\t\t" << bounding_boxes[i].dP1 << endl;
            }
                    
            for(int i = 0; i < landmarks.size(); i++)
            {
                cout << "\tLandmarks:" << endl
                    << "\t\t" << landmarks[i].LE << endl
                    << "\t\t" << landmarks[i].RE << endl
                    << "\t\t" << landmarks[i].N << endl
                    << "\t\t" << landmarks[i].LM << endl
                    << "\t\t" << landmarks[i].RM << endl;
            }
        }

        vector<box> bounding_boxes;
        vector<landmark> landmarks;
  
  private:
        cv::Mat image;
        string filename;
        
  private:
};

#endif