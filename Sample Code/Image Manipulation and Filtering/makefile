# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OSX include paths for MacPorts
CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/local/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to wwhere your openCV libraries are stored
# these settings work for macports
LDFLAGS = -L/opt/local/lib/opencv4/ -L/opt/local/lib  # opencv libraries are here

# opencv libraries and MacOS frameworks
# these settings work for macOS and macports
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect 


# Location to write the executables
BINDIR = .

# rules for making various executables
img: imgMod.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

tut1: tut1.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

tut2: tut2.o gauss.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~ 
