# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build"

# Include any dependencies generated for this target.
include CMakeFiles/Tutorial2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Tutorial2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Tutorial2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Tutorial2.dir/flags.make

CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o: CMakeFiles/Tutorial2.dir/flags.make
CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 1/Examples/Tutorial2.cpp
CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o: CMakeFiles/Tutorial2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o -MF CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o.d -o CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Tutorial2.cpp"

CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Tutorial2.cpp" > CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.i

CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Tutorial2.cpp" -o CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.s

CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o: CMakeFiles/Tutorial2.dir/flags.make
CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 1/Examples/Gaussian.cpp
CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o: CMakeFiles/Tutorial2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o -MF CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o.d -o CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Gaussian.cpp"

CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Gaussian.cpp" > CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.i

CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/Examples/Gaussian.cpp" -o CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.s

# Object files for target Tutorial2
Tutorial2_OBJECTS = \
"CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o" \
"CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o"

# External object files for target Tutorial2
Tutorial2_EXTERNAL_OBJECTS =

Tutorial2: CMakeFiles/Tutorial2.dir/Examples/Tutorial2.cpp.o
Tutorial2: CMakeFiles/Tutorial2.dir/Examples/Gaussian.cpp.o
Tutorial2: CMakeFiles/Tutorial2.dir/build.make
Tutorial2: /opt/homebrew/lib/libopencv_gapi.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_stitching.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_alphamat.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_aruco.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_bgsegm.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_bioinspired.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_ccalib.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_dnn_objdetect.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_dnn_superres.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_dpm.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_face.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_freetype.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_fuzzy.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_hfs.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_img_hash.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_intensity_transform.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_line_descriptor.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_mcc.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_quality.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_rapid.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_reg.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_rgbd.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_saliency.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_sfm.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_stereo.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_structured_light.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_superres.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_surface_matching.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_tracking.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_videostab.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_viz.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_wechat_qrcode.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_xfeatures2d.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_xobjdetect.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_xphoto.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_shape.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_highgui.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_datasets.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_plot.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_text.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_ml.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_phase_unwrapping.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_optflow.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_ximgproc.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_video.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_videoio.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_imgcodecs.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_objdetect.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_calib3d.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_dnn.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_features2d.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_flann.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_photo.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_imgproc.4.8.1.dylib
Tutorial2: /opt/homebrew/lib/libopencv_core.4.8.1.dylib
Tutorial2: CMakeFiles/Tutorial2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Tutorial2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Tutorial2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Tutorial2.dir/build: Tutorial2
.PHONY : CMakeFiles/Tutorial2.dir/build

CMakeFiles/Tutorial2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Tutorial2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Tutorial2.dir/clean

CMakeFiles/Tutorial2.dir/depend:
	cd "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/build/CMakeFiles/Tutorial2.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/Tutorial2.dir/depend

