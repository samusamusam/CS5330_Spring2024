# OSX compiler
CC = clang++

# Dwarf compiler
#CC = g++

CXX = $(CC)

# OSX include paths
CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/local/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF

# Dwarf include paths
#CFLAGS = -I. # opencv includes are in /usr/include

CXXFLAGS = $(CFLAGS)

# OSX Library paths
LDFLAGS = -L/opt/local/lib/opencv4/ -L/opt/local/lib

# Dwarf Library paths
#LDFLAGS = -L/usr/lib64 # opencv libraries are here

# OSX libraries
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

# Dwarf libraries
#LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc 

BINDIR = .

dnnex: dnnexample.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~ 
