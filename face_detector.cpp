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

        cv::Mat img = cv::imread(file, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;

        cv::Mat Matfloat;
        cv::Mat processed_frame;
        img.convertTo(Matfloat, CV_32FC3);
	
        cv::Mat Normalized;
	Normalized = Matfloat;
//	cv::normalize(Matfloat, Normalized, -1, 1, cv::NORM_MINMAX, -1);

        if (Normalized.channels() == 3 || Normalized.channels() == 4 )
          cv::cvtColor(Normalized, Normalized, cv::COLOR_BGR2RGB);
        else if (Normalized.channels() == 1)
          cv::cvtColor(Normalized, Normalized, cv::COLOR_GRAY2RGB);

        processed_frame = Normalized.t();

        // Detect function
        clock_t begin = clock();
        cv::Mat outputImage = detector.Detect(img);
        outputImage = detector.Detect(Matfloat);
        outputImage = detector.Detect(Normalized);
        outputImage = detector.Detect(processed_frame);
		

        clock_t end = clock();
        // Print Output
        cout << "Execution time was: " << double(end-begin) / CLOCKS_PER_SEC << endl;
        
        if(outputImage.empty())
        {
              cout << "Failed ..." << endl;
              return 1;
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
}
