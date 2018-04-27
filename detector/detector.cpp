#include "detector.h"

// Function to return indices of sorted array
vector<int> ordered(vector<float> values) {
    std::vector<int> indices(values.size());
    std::size_t n(0);
    std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });

    std::sort(
        std::begin(indices), std::end(indices),
        [&](size_t a, size_t b) { return values[a] < values[b]; }
    );
    return indices;
}

/*
        GENERAL FUNCTIONS
        *****************
*/
Detector::Detector(const string& pnet_model_file,
                   const string& pnet_trained_file,
                   const string& rnet_model_file,
                   const string& rnet_trained_file,
                   const string& onet_model_file,
                   const string& onet_trained_file,
                   const string& image_name) 
                   :    pnet(pnet_model_file, 
                        pnet_trained_file),
                        rnet(rnet_model_file, 
                        rnet_trained_file),
                        onet(onet_model_file, 
                        onet_trained_file) {
                           
        // Definitions 
        minSize = 20;
         factor = 0.709;

        // Threshold to consider points as potential
        // candidates. 3 for the 3 nets.
        thresholds[0] = 0.6;
        thresholds[1] = 0.6;
        thresholds[2] = 0.8;
        
        // Threshold to merge candidates
        nms_thresholds[0] = 0.8;
        nms_thresholds[1] = 0.7;
        nms_thresholds[2] = 0.3;
        
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif

}

const cv::Mat Detector::Detect(Image& img) {

        // First Stage
        cout << "Running PNET" << endl;
        pnetWrapper(img);
        img.print();
        
        // Second Stage
        if (img.bounding_boxes.size() > 0){
                cout << "Running RNET" << endl;
                rnetWrapper(img);
                img.print();
        }
        
        // Third Stage
        if (img.bounding_boxes.size() > 0){
                cout << "Running ONET" << endl;
                onetWrapper(img);
                img.print();
        }
               
        // Write final output to global variables
        cout << "Creating Output" << endl;
        writeOutputImage(img);
        img.print();
        
        return img.processed_image;
}

vector<int> Detector::nms (std::vector <box>total_boxes, 
                           float threshold, 
                           bool  type){

        vector <int> pick;
        // cout << "NMS Recieves total boxes of " << total_boxes.size() << endl;
        
        if (total_boxes.size() == 0){
                return pick;
        }
        
        vector <float> x1  (total_boxes.size());
        vector <float> y1  (total_boxes.size());
        vector <float> x2  (total_boxes.size());
        vector <float> y2  (total_boxes.size());
        vector <float> s   (total_boxes.size());
        vector <float> area(total_boxes.size());

        // Initialize vectors
        for (unsigned int i = 0; i < total_boxes.size(); i++){
                  x1[i] = total_boxes[i].p1.x;
                  y1[i] = total_boxes[i].p1.y;
                  x2[i] = total_boxes[i].p2.x;
                  y2[i] = total_boxes[i].p2.y;
                   s[i] = total_boxes[i].score;
                area[i] = ((float)x2[i]-(float)x1[i]) * ((float)y2[i]-(float)y1[i]);
        }

        // Sort s and create indexes
        vector <int> I = ordered(s);
        // for (int i = 0; i < I.size(); i++){
                // cout << I[i] << " " << total_boxes[I[i]].score << endl;//" " << s[i] << endl;
        // }
        while (I.size() > 0){
                
                // To store new Indexes
                vector <int> Inew;

                int i = I[I.size() - 1];
                pick.push_back(i);
                
                for (unsigned int j = 0; j < I.size()-1; j++){
                        float   xx1 = max(x1[i],  x1[I[j]]);
                        float   yy1 = max(y1[i],  y1[I[j]]);
                        float   xx2 = min(x2[i],  x2[I[j]]);
                        float   yy2 = min(y2[i],  y2[I[j]]);
                        float     w = max(  0.0f, (xx2-xx1));
                        float     h = max(  0.0f, (yy2-yy1));
                        float inter = w * h;
                        float   out;
                        if (type == false){ // Union
                                out = inter/(area[i] + area[I[j]] - inter);
                        } else { // Min
                                out = inter/min(area[i], area[I[j]]);
                        }
                        // Add index to Inew if under threshold
                        if (out <= threshold){
                               Inew.push_back(I[j]); 
                        }
                } 
                // Copy new I into I
                I.swap(Inew);
                Inew.clear();
        }
        // pick.clear();
        // for (int j = 0; j < total_boxes.size(); j++){
                // pick.push_back(j);
        // }
        // cout << "NMS Chosen boxes are " << pick.size() << endl;
        return pick;
}

vector<box> Detector::generateBoundingBox(std::vector< std::vector <float>> data,
                                          std::vector<int> shape_map,
                                          float scale,
                                          float threshold){

        int stride   = 2;
        int cellsize = 12;
        
        // cout << "Generate bounding box output. Receiving shape of: " 
        // << shape_map[0] << " "
        // << shape_map[1] << " "
        // << shape_map[2] << " "
        // << shape_map[3] << endl;
        // cout << "Scale: " << scale << endl;
        // cout << "Threshold: "  << threshold << endl;
        
        vector<box> temp_boxes;
        for (int y = 0; y < shape_map[2]; y++){
                for (int x = 0; x < shape_map[3]; x++){
                        // We need to access the second array.
                        if (data[1][(shape_map[2] + y) * shape_map[3] + x] >= threshold){
                                box temp_box;
                                
                                // Points for Bounding Boxes
                                cv::Point p1(floor((stride*x+1)/scale),
                                             floor((stride*y+1)/scale));
                                cv::Point p2(floor((stride*x+cellsize-1+1)/scale),
                                             floor((stride*y+cellsize-1+1)/scale));
                                             
                                temp_box.p1 = p1;
                                temp_box.p2 = p2;
                                
                                //.score
                                temp_box.score = data[1][(shape_map[2] + y) * shape_map[3] + x];
                                
                                // Reg (dx1,dy1,dx2,dy2)
                                cv::Point dp1 (data[0][(0*shape_map[2] + y) * shape_map[3] + x],
                                               data[0][(1*shape_map[2] + y) * shape_map[3] + x]);
                                cv::Point dp2 (data[0][(2*shape_map[2] + y) * shape_map[3] + x],
                                               data[0][(3*shape_map[2] + y) * shape_map[3] + x]);
                                
                                temp_box.p1 = dp1;
                                temp_box.p2 = dp2;
                                
                                // Add box to bounding boxes
                                temp_boxes.push_back(temp_box);
                        }
                }
        }
        //cout << "Generated a total of " << temp_boxes.size() << " boxes" << endl;
        return temp_boxes;
}

void Detector::printCurrentOutputs(const char* folder_name, Image& image) {
         
        // Generate cropped images from the main image        
        for (unsigned int i = 0; i < image.bounding_boxes.size(); i++) {
                cv::Rect rect =  cv::Rect(image.bounding_boxes[i].p1.x,
                                          image.bounding_boxes[i].p1.y, 
                                          image.bounding_boxes[i].p2.x - image.bounding_boxes[i].p1.x,  //width
                                          image.bounding_boxes[i].p2.y - image.bounding_boxes[i].p1.y); //height
                cv::Mat crop = cv::Mat(image.get(), rect).clone();
                
                
                int minl = min (image.get().rows, image.get().cols);
        
                // Used so the thickness of the marks is based on the size
                // of the image
                int thickness = ceil((float) minl / 270.0);
        
                if (folder_name == "ONET"){
                        cv::circle(crop, 
                                image.landmarks[i].LE-image.bounding_boxes[i].p1,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(crop, 
                                image.landmarks[i].RE-image.bounding_boxes[i].p1,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(crop, 
                                image.landmarks[i].N-image.bounding_boxes[i].p1,
                                thickness,
                                cv::Scalar(0, 255, 0),
                                -1);
                        cv::circle(crop, 
                                image.landmarks[i].LM-image.bounding_boxes[i].p1,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                        cv::circle(crop, 
                                image.landmarks[i].RM-image.bounding_boxes[i].p1,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                        
                }
                
                // Save the image
                stringstream ss;

                string name;// = "Res_";
                string type = ".jpg";

                ss << folder_name << "/" << name << image.bounding_boxes[i].score << type;

                string filename = ss.str();
                ss.str("");

                cv::imwrite(filename, crop);

        }
}

void Detector::padBoundingBox(Image img, int imgHeight, int imgWidth){
        
        for (unsigned int j = 0; j < img.bounding_boxes.size(); j++){
                if (img.bounding_boxes[j].p2.x >= imgWidth){ //.p2.x > w
                        // shift box
                        img.bounding_boxes[j].p1.x -= img.bounding_boxes[j].p2.x - imgWidth;
                        img.bounding_boxes[j].p2.x = imgWidth - 1;
                }
                
                if (img.bounding_boxes[j].p2.y >= imgHeight){ //.p2.y > h
                        // shift box
                        img.bounding_boxes[j].p1.y -= img.bounding_boxes[j].p2.y - imgHeight;
                        img.bounding_boxes[j].p2.y = imgHeight - 1;
                }
                
                if (img.bounding_boxes[j].p1.x < 0){
                        // shift box
                        img.bounding_boxes[j].p2.x -= img.bounding_boxes[j].p1.x;
                        img.bounding_boxes[j].p1.x = 0;
                }
                
                if (img.bounding_boxes[j].p1.y < 0){
                        // shift box
                        img.bounding_boxes[j].p2.y -= img.bounding_boxes[j].p1.y;
                        img.bounding_boxes[j].p1.y = 0;
                }
        }
}

void Detector::writeOutputImage(Image& img) {
 
        img.get().copyTo(img.processed_image);
        
        int minl = min (img.get().rows, img.get().cols);
        
        // Used so the thickness of the marks is based on the size
        // of the image
        int thickness = ceil((float) minl / 270.0);
        
        for (unsigned int i = 0; i < img.bounding_boxes.size(); i++) {
                cv::rectangle(img.processed_image, 
                        img.bounding_boxes[i].p1, 
                        img.bounding_boxes[i].p2, 
                        cv::Scalar(255, 255, 255),
                        thickness);
        }
        for (unsigned int i = 0; i < img.landmarks.size(); i++) {
                cv::circle(img.processed_image, 
                        img.landmarks[i].LE,
                        thickness,
                        cv::Scalar(255, 0, 0),
                        -1);
                cv::circle(img.processed_image, 
                        img.landmarks[i].RE,
                        thickness,
                        cv::Scalar(255, 0, 0),
                        -1);
                cv::circle(img.processed_image, 
                        img.landmarks[i].N,
                        thickness,
                        cv::Scalar(0, 255, 0),
                        -1);
                cv::circle(img.processed_image, 
                        img.landmarks[i].LM,
                        thickness,
                        cv::Scalar(0, 0, 255),
                        -1);
                cv::circle(img.processed_image, 
                        img.landmarks[i].RM,
                        thickness,
                        cv::Scalar(0, 0, 255),
                        -1);
        }
}

void Detector::pnetWrapper(Image& img)
{        
            // Preprocess Input image (Convert to Float, Normalize, change channels, transpose)
        cv::Mat Matfloat;
        img.frame.convertTo(Matfloat, CV_32FC3);

        cv::Mat Normalized;
        cv::normalize(Matfloat, Normalized, -1, 1, cv::NORM_MINMAX, -1);

        if (Normalized.channels() == 3 || Normalized.channels() == 4 )
        cv::cvtColor(Normalized, Normalized, cv::COLOR_BGR2RGB);
        else if (Normalized.channels() == 1)
        cv::cvtColor(Normalized, Normalized, cv::COLOR_GRAY2RGB);

        img.processed_frame = Normalized.t();
        /*
                Initialize INPUTS
        */
        int factor_count = 0;        
        float minl = min (img.get().rows, img.get().cols);
        float m = 12.0 / (float) minSize;

        // Fixme: For performance
        // Further scale images to process image through NN efficiently 
        // (When images are really big!!)
        if (minl >= 1080) m = m * 1080 / (minl * 1.7);
        
        minl = minl*m;
        
        // Create Scale Pyramid
        std::vector<float> scales;
        
        while (minl >= 12){
                scales.push_back(m*pow(factor,factor_count));
                minl *= factor;
                factor_count++;
        }
        
        for (unsigned int j = 0; j < scales.size(); j++){
                // Create Scale Images
                float scale = scales[j];
                
                cv::Size pnet_input_geometry (ceil(img.processed_image.cols*scale), 
                                              ceil(img.processed_image.rows*scale));
                pnet.SetInputGeometry(pnet_input_geometry);
                
                // Resize the Image
                std::vector <cv::Mat> img_data;
                
                cv::Mat resized;
                cv::resize(img.processed_image, resized, pnet_input_geometry);
                
                img_data.push_back(resized);
                
                // Pnet Input Setup
                pnet.FeedInput(img_data);
                
                // Pnet Forward data
                pnet.Forward();
              
                std::vector<int> shape;
                std::vector<int>* shape_ptr = &shape;
                std::vector< std::vector <float>> output_data;
                std::vector< std::vector <float>>* output_data_ptr = &output_data;
                
                pnet.RetrieveOutput(*shape_ptr, *output_data_ptr);
                
                // Generate Bounding Box based on output from net
                vector<box> temp_boxes = generateBoundingBox(output_data,
                                                             shape,
                                                             scale,
                                                             thresholds[0]);
                // Run NMS on boxes
                vector<int> pick = nms (temp_boxes, nms_thresholds[0], 0);
                
                // Select chosen boxes, update img.bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(temp_boxes[pick[j]]);
                }
                img.bounding_boxes.insert(img.bounding_boxes.end(), chosen_boxes.begin(), chosen_boxes.end()); 
        }
        
        if (img.bounding_boxes.size() > 0){
                vector<int> pick = nms (img.bounding_boxes, nms_thresholds[1], 0);
                // Select chosen boxes, update img.bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(img.bounding_boxes[pick[j]]);
                }
                
                img.bounding_boxes.swap(chosen_boxes);
                
                vector<box> correct_box(img.bounding_boxes.size());
                for (unsigned int j = 0; j < img.bounding_boxes.size(); j++){
                        float regw = img.bounding_boxes[j].p2.x-img.bounding_boxes[j].p1.x;
                        float regh = img.bounding_boxes[j].p2.y-img.bounding_boxes[j].p1.y;
                        correct_box[j].p1.x = img.bounding_boxes[j].p1.x + img.bounding_boxes[j].dP1.x*regw;
                        correct_box[j].p1.y = img.bounding_boxes[j].p1.y + img.bounding_boxes[j].dP1.y*regh;
                        correct_box[j].p2.x = img.bounding_boxes[j].p2.x + img.bounding_boxes[j].dP2.x*regw;
                        correct_box[j].p2.y = img.bounding_boxes[j].p2.y + img.bounding_boxes[j].dP2.y*regh;
                        correct_box[j].score = img.bounding_boxes[j].score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].p2.y - correct_box[j].p1.y;
                        float w = correct_box[j].p2.x - correct_box[j].p1.x;
                        float l = max(w, h);
                        
                        correct_box[j].p1.x += w*0.5 - l*0.5;
                        correct_box[j].p1.y += h*0.5 - l*0.5;
                        correct_box[j].p2.x = correct_box[j].p1.x + l;
                        correct_box[j].p2.y = correct_box[j].p1.y + l;
                        
                        // Fix value to int
                        correct_box[j].p1.x = floor(correct_box[j].p1.x);
                        correct_box[j].p1.y = floor(correct_box[j].p1.y);
                        correct_box[j].p2.x = floor(correct_box[j].p2.x);
                        correct_box[j].p2.y = floor(correct_box[j].p2.y);
                }
                
                img.bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img, img.processed_image.rows, img.processed_image.cols);
                
        }
}

void Detector::rnetWrapper(Image& img){
        
        cv::Size rnet_input_geometry(24, 24);
        
        rnet.SetInputGeometry(rnet_input_geometry);
        
        // Vector of cropped images
        vector<cv::Mat> cropBoxes;

        // Generate cropped images from the main image        
        for (unsigned int i = 0; i < img.bounding_boxes.size(); i++) {
                
                cv::Rect rect =  cv::Rect(img.bounding_boxes[i].p1.x,
                                          img.bounding_boxes[i].p1.y, 
                                          img.bounding_boxes[i].p2.x - img.bounding_boxes[i].p1.x,  //width
                                          img.bounding_boxes[i].p2.y - img.bounding_boxes[i].p1.y); //height
        
                cv::Mat crop = cv::Mat(img.processed_image, rect).clone();
               
                // Resize the cropped Image
                cv::Mat img_data;
                cv::resize(crop, img_data, rnet_input_geometry);
                
                cropBoxes.push_back(img_data);
        }

        // Rnet Input Setup
        rnet.FeedInput(cropBoxes);
        
        // Rnet Forward data
        rnet.Forward();
      
        std::vector<int> shape;
        std::vector<int>* shape_ptr = &shape;
        std::vector< std::vector <float>> output_data;
        std::vector< std::vector <float>>* output_data_ptr = &output_data;
        
        rnet.RetrieveOutput(*shape_ptr, *output_data_ptr);
        
        // Filter Boxes that are over threshold and collect mv output values as well
        vector<box> chosen_boxes;
        for (int j = 0; j < shape[0]; j++){ // same as num boxes
                if (output_data[0][j*2+1] > thresholds[1]){
                        
                        // Saving mv output data in boxes extra information
                        img.bounding_boxes[j].p1.x = output_data[1][j*4+0];
                        img.bounding_boxes[j].p1.y = output_data[1][j*4+1];
                        img.bounding_boxes[j].p2.x = output_data[1][j*4+2];
                        img.bounding_boxes[j].p2.y = output_data[1][j*4+3];              
                        img.bounding_boxes[j].score = output_data[0][j*2+1];
                        chosen_boxes.push_back(img.bounding_boxes[j]);
                }
        }
        img.bounding_boxes.swap(chosen_boxes);
               
        if (img.bounding_boxes.size() > 0){
                vector<int> pick = nms (img.bounding_boxes, nms_thresholds[1], 0);
                // Select chosen boxes, update img.bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(img.bounding_boxes[pick[j]]);
                }
                
                img.bounding_boxes.swap(chosen_boxes);
                                        
                vector<box> correct_box(img.bounding_boxes.size());
                for (unsigned int j = 0; j < img.bounding_boxes.size(); j++){
                        
                        // Apply BBREG
                        float regw = img.bounding_boxes[j].p2.x-img.bounding_boxes[j].p1.x;
                        float regh = img.bounding_boxes[j].p2.y-img.bounding_boxes[j].p1.y;
                        correct_box[j].p1.x = img.bounding_boxes[j].p1.x + img.bounding_boxes[j].dP1.x*regw;
                        correct_box[j].p1.y = img.bounding_boxes[j].p1.y + img.bounding_boxes[j].dP1.y*regh;
                        correct_box[j].p2.x = img.bounding_boxes[j].p2.x + img.bounding_boxes[j].dP2.x*regw;
                        correct_box[j].p2.y = img.bounding_boxes[j].p2.y + img.bounding_boxes[j].dP2.y*regh;
                        correct_box[j].score = img.bounding_boxes[j].score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].p2.y - correct_box[j].p1.y;
                        float w = correct_box[j].p2.x - correct_box[j].p1.x;
                        float l = max(w, h);
                        
                        correct_box[j].p1.x += w*0.5 - l*0.5;
                        correct_box[j].p1.y += h*0.5 - l*0.5;
                        correct_box[j].p2.x = correct_box[j].p1.x + l;
                        correct_box[j].p2.y = correct_box[j].p1.y + l;
                        
                        // Fix value to int
                        correct_box[j].p1.x = floor(correct_box[j].p1.x);
                        correct_box[j].p1.y = floor(correct_box[j].p1.y);
                        correct_box[j].p2.x = floor(correct_box[j].p2.x);
                        correct_box[j].p2.y = floor(correct_box[j].p2.y);
                }
                
                img.bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img, img.processed_image.rows, img.processed_image.cols);
                
                // Test
                // cout << "Total bounding boxes passing " << img.bounding_boxes.size() << endl;
        }
}

void Detector::onetWrapper(Image& img){
        
        cv::Size onet_input_geometry(48, 48);
        
        onet.SetInputGeometry(onet_input_geometry);
                vector<cv::Mat> cropBoxes;

        // Generate cropped images from the main image
        for (unsigned int i = 0; i < img.bounding_boxes.size(); i++) {

                cv::Rect rect =  cv::Rect(img.bounding_boxes[i].p1.x,
                                                img.bounding_boxes[i].p1.y,
                                                img.bounding_boxes[i].p2.x - img.bounding_boxes[i].p1.x, //width
                                                img.bounding_boxes[i].p2.y - img.bounding_boxes[i].p1.y) //height
                                                & cv::Rect(0, 0, img.get().cols, img.get().rows);

                cv::Mat crop = cv::Mat(img.get(), rect).clone();

                // Resize the cropped Image
                cv::Mat img_data;

                cv::resize(crop, img_data, onet_input_geometry);

                cropBoxes.push_back(img_data);

                img_data.release();
        }
        onet.FeedInput(cropBoxes);
        // Onet Forwar d data
        onet.Forward();
      
        std::vector<int> shape;
        std::vector<int>* shape_ptr = &shape;
        std::vector< std::vector <float>> output_data;
        std::vector< std::vector <float>>* output_data_ptr = &output_data;
        
        onet.RetrieveOutput(*shape_ptr, *output_data_ptr);
        
        // Print Image data!//
        Blob* input_layer = onet.GetNet()->input_blobs()[0];
        const std::vector<int> shape2 = input_layer->shape();
                
        // Filter Boxes that are over threshold and collect mv output values as well
        vector<box> chosen_boxes;
        for (int j = 0; j < shape[0]; j++){ // same as num boxes
                if (output_data[2][j*2+1] > thresholds[2]){
                        
                        // Saving mv output data in boxes extra information
                        img.bounding_boxes[j].p1.x = output_data[0][j*4+0];
                        img.bounding_boxes[j].p1.y = output_data[0][j*4+1];
                        img.bounding_boxes[j].p2.x = output_data[0][j*4+2];
                        img.bounding_boxes[j].p2.y = output_data[0][j*4+3];              
                        img.bounding_boxes[j].score = output_data[2][j*2+1];
                        chosen_boxes.push_back(img.bounding_boxes[j]);
                        
                        // Create Points for box
                        landmark points;
                        
                        float w = img.bounding_boxes[j].p2.x - img.bounding_boxes[j].p1.x;
                        float h = img.bounding_boxes[j].p2.y - img.bounding_boxes[j].p1.y;

                        points.LE.x = w*output_data[1][j*10+0] + img.bounding_boxes[j].p1.x;
                        points.RE.x = w*output_data[1][j*10+1] + img.bounding_boxes[j].p1.x;
                        points.N.x  = w*output_data[1][j*10+2] + img.bounding_boxes[j].p1.x;
                        points.LM.x = w*output_data[1][j*10+3] + img.bounding_boxes[j].p1.x;
                        points.RM.x = w*output_data[1][j*10+4] + img.bounding_boxes[j].p1.x;
                        
                        points.LE.y = h*output_data[1][j*10+5] + img.bounding_boxes[j].p1.y;
                        points.RE.y = h*output_data[1][j*10+6] + img.bounding_boxes[j].p1.y;
                        points.N.y  = h*output_data[1][j*10+7] + img.bounding_boxes[j].p1.y;
                        points.LM.y = h*output_data[1][j*10+8] + img.bounding_boxes[j].p1.y;
                        points.RM.y = h*output_data[1][j*10+9] + img.bounding_boxes[j].p1.y;
                        
                        img.landmarks.push_back(points);
                }
        }
        img.bounding_boxes.swap(chosen_boxes);
               
        if (img.bounding_boxes.size() > 0){
                vector<int> pick = nms (img.bounding_boxes, nms_thresholds[2], 1); // Min
                // Select chosen boxes, update img.bounding_boxes vector
                vector<box> chosen_boxes;
                vector<landmark> chosen_points;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(img.bounding_boxes[pick[j]]);
                        chosen_points.push_back(img.landmarks[pick[j]]);
                }
                
                img.bounding_boxes.swap(chosen_boxes);
                img.landmarks.swap(chosen_points);
                                        
                vector<box> correct_box(img.bounding_boxes.size());
                for (unsigned int j = 0; j < img.bounding_boxes.size(); j++){
                        
                        // Apply BBREG
                
                        float regw = img.bounding_boxes[j].p2.x-img.bounding_boxes[j].p1.x;
                        float regh = img.bounding_boxes[j].p2.y-img.bounding_boxes[j].p1.y;
                        correct_box[j].p1.x = img.bounding_boxes[j].p1.x + img.bounding_boxes[j].dP1.x*regw;
                        correct_box[j].p1.y = img.bounding_boxes[j].p1.y + img.bounding_boxes[j].dP1.y*regh;
                        correct_box[j].p2.x = img.bounding_boxes[j].p2.x + img.bounding_boxes[j].dP2.x*regw;
                        correct_box[j].p2.y = img.bounding_boxes[j].p2.y + img.bounding_boxes[j].dP2.y*regh;
                        correct_box[j].score = img.bounding_boxes[j].score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].p2.y - correct_box[j].p1.y;
                        float w = correct_box[j].p2.x - correct_box[j].p1.x;
                        float l = max(w, h);
                        
                        correct_box[j].p1.x += w*0.5 - l*0.5;
                        correct_box[j].p1.y += h*0.5 - l*0.5;
                        correct_box[j].p2.x = correct_box[j].p1.x + l;
                        correct_box[j].p2.y = correct_box[j].p1.y + l;
                        
                        // Fix value to int
                        correct_box[j].p1.x = floor(correct_box[j].p1.x);
                        correct_box[j].p1.y = floor(correct_box[j].p1.y);
                        correct_box[j].p2.x = floor(correct_box[j].p2.x);
                        correct_box[j].p2.y = floor(correct_box[j].p2.y);
                }
                
                img.bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img, img.get().rows, img.get().cols);
                
                // Test
                // cout << "Total bounding boxes passing " << img.bounding_boxes.size() << endl;
        }
}
