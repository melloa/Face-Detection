#include <ctime>
#include "config.h"
#include "detector/detector.h"
#include "net/bnet.h"
#include "net/pnet.h"
#include "net/rnet.h"
#include "net/onet.h"

using namespace caffe;
using std::string;

int main(int argc, char** argv) {
        if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
              << " img.jpg" << std::endl;
        return 1;
        }

        ::google::InitGoogleLogging(argv[0]);

        string pnet_model_file   = "model/det1.prototxt";
        string pnet_trained_file = "model/det1.caffemodel";
        string rnet_model_file   = "model/det2.prototxt";
        string rnet_trained_file = "model/det2.caffemodel";
        string onet_model_file   = "model/det3.prototxt";
        string onet_trained_file = "model/det3.caffemodel";
        
        string file = argv[1];

        Detector detector(pnet_model_file, 
                        pnet_trained_file, 
                        rnet_model_file,
                        rnet_trained_file,
                        onet_model_file,
                        onet_trained_file,
                        file);

        std::cout << "---------- Face Detector for "
                  << file << " ----------" << std::endl;

        Image img(file);
        CHECK(!img.empty()) << "Unable to decode image " << file;

        // Detect function
        clock_t begin = clock();
        cv::Mat outputImage = detector.Detect(img);

        clock_t end = clock();
        // Print Output
        cout << "Execution time was: " << double(end-begin) / CLOCKS_PER_SEC << endl;
        
        if(outputImage.empty())
        {
              cout << "Failed ..." << endl;
              return -1;
        }

        
        stringstream ss;
        ss << "outputs/" << file ;
        string commS = ss.str();
        // remove input part
        string in = "inputs/";
        string::size_type i = commS.find(in);
        if (i!= std::string::npos) commS.erase(i,in.length());
        const char* comm = commS.c_str();
        cout << "writing " << comm << endl;
        cv::imwrite(comm, outputImage);
        return 0;
}
