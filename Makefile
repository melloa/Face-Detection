GCC := g++
RM = rm -f

CaffeLocation = /opt/caffe
CaffeLIB = -L$(CaffeLocation)/build/lib
CaffeINC = -I$(CaffeLocation)/include/
BlasLIB = -I/usr/local/cuda-9.0/targets/x86_64-linux/include/
OpenCVLib = -L/workspace/Face-Detection/opencv-3.4.1/build/modules/
NetLocation = /workspace/Face-Detection/net
NetLIB = -L$(NetLocation)

GccFLAGS =  -pthread -std=c++11 -O3 $(BlasLIB) $(OpenCVLib)
GccLibs = $(CaffeLIB) $(CaffeINC) $(NetLIB) $(OpenCVLib)

GccLinkFLAGS = -lprotobuf -lglog `pkg-config opencv --cflags --libs` -lboost_system -lcaffe-nv -lnet

debug: GccFLAGS += -DDEBUG -g -Wall
debug: all

# The build target executable:
TARGET = face_detector

all: build

build: $(TARGET)

$(TARGET): $(TARGET).cpp detector/detector.cpp net/libnet.so #matlab/onetInput
	$(GCC) $(GccLibs) -Wl,-rpath=$(NetLocation) $< detector/detector.cpp -o $@ $(GccFLAGS) $(GccLinkFLAGS)

# Create Shared library for net objects
net/libnet.so: net/bnet.o net/pnet.o net/rnet.o net/onet.o
	$(GCC) $(CaffeINC) $(GccFLAGS) -shared $< net/pnet.o net/rnet.o net/onet.o -o $@ 

net/bnet.o: net/bnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/pnet.o: net/pnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/rnet.o: net/rnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/onet.o: net/onet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

matlab/onetInput: matlab/onetInput.cpp
	matlab -nodisplay -nosplash -nodesktop -r "mex -v -client engine matlab/onetInput.cpp;exit;"
	mv onetInput matlab/

no-libnet: $(TARGET).cpp detector/detector.cpp 
	$(GCC) $(GccLibs) -Wl,-rpath=$(NetLocation) $< detector/detector.cpp -o $@ $(GccFLAGS) $(GccLinkFLAGS)
			
clean:
	$(RM) $(TARGET) *.o net/*.so net/*.o *.tar* *.core* matlab/onetInput
        
run:
	./$(TARGET) test1.jpg
