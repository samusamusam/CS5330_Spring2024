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
CMAKE_SOURCE_DIR = "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build"

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/main.cpp
CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/main.cpp"

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/main.cpp" > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/main.cpp" -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/threshold.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/threshold.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/threshold.cpp
CMakeFiles/main.dir/threshold.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/threshold.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/threshold.cpp.o -MF CMakeFiles/main.dir/threshold.cpp.o.d -o CMakeFiles/main.dir/threshold.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/threshold.cpp"

CMakeFiles/main.dir/threshold.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/threshold.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/threshold.cpp" > CMakeFiles/main.dir/threshold.cpp.i

CMakeFiles/main.dir/threshold.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/threshold.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/threshold.cpp" -o CMakeFiles/main.dir/threshold.cpp.s

CMakeFiles/main.dir/cleanup.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/cleanup.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/cleanup.cpp
CMakeFiles/main.dir/cleanup.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/cleanup.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/cleanup.cpp.o -MF CMakeFiles/main.dir/cleanup.cpp.o.d -o CMakeFiles/main.dir/cleanup.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/cleanup.cpp"

CMakeFiles/main.dir/cleanup.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/cleanup.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/cleanup.cpp" > CMakeFiles/main.dir/cleanup.cpp.i

CMakeFiles/main.dir/cleanup.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/cleanup.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/cleanup.cpp" -o CMakeFiles/main.dir/cleanup.cpp.s

CMakeFiles/main.dir/segment.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/segment.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/segment.cpp
CMakeFiles/main.dir/segment.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/segment.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/segment.cpp.o -MF CMakeFiles/main.dir/segment.cpp.o.d -o CMakeFiles/main.dir/segment.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/segment.cpp"

CMakeFiles/main.dir/segment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/segment.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/segment.cpp" > CMakeFiles/main.dir/segment.cpp.i

CMakeFiles/main.dir/segment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/segment.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/segment.cpp" -o CMakeFiles/main.dir/segment.cpp.s

CMakeFiles/main.dir/computeFeatures.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/computeFeatures.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/computeFeatures.cpp
CMakeFiles/main.dir/computeFeatures.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/computeFeatures.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/computeFeatures.cpp.o -MF CMakeFiles/main.dir/computeFeatures.cpp.o.d -o CMakeFiles/main.dir/computeFeatures.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/computeFeatures.cpp"

CMakeFiles/main.dir/computeFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/computeFeatures.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/computeFeatures.cpp" > CMakeFiles/main.dir/computeFeatures.cpp.i

CMakeFiles/main.dir/computeFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/computeFeatures.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/computeFeatures.cpp" -o CMakeFiles/main.dir/computeFeatures.cpp.s

CMakeFiles/main.dir/csvReadWrite.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/csvReadWrite.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/csvReadWrite.cpp
CMakeFiles/main.dir/csvReadWrite.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/main.dir/csvReadWrite.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/csvReadWrite.cpp.o -MF CMakeFiles/main.dir/csvReadWrite.cpp.o.d -o CMakeFiles/main.dir/csvReadWrite.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/csvReadWrite.cpp"

CMakeFiles/main.dir/csvReadWrite.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/csvReadWrite.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/csvReadWrite.cpp" > CMakeFiles/main.dir/csvReadWrite.cpp.i

CMakeFiles/main.dir/csvReadWrite.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/csvReadWrite.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/csvReadWrite.cpp" -o CMakeFiles/main.dir/csvReadWrite.cpp.s

CMakeFiles/main.dir/distance.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/distance.cpp.o: /Users/sam/Documents/Northeastern\ Khoury/CS5330\ Pattern\ Recognition\ &\ Computer\ Vision/Projects/Project\ 3/distance.cpp
CMakeFiles/main.dir/distance.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/main.dir/distance.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/distance.cpp.o -MF CMakeFiles/main.dir/distance.cpp.o.d -o CMakeFiles/main.dir/distance.cpp.o -c "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/distance.cpp"

CMakeFiles/main.dir/distance.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/distance.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/distance.cpp" > CMakeFiles/main.dir/distance.cpp.i

CMakeFiles/main.dir/distance.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/distance.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/distance.cpp" -o CMakeFiles/main.dir/distance.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/threshold.cpp.o" \
"CMakeFiles/main.dir/cleanup.cpp.o" \
"CMakeFiles/main.dir/segment.cpp.o" \
"CMakeFiles/main.dir/computeFeatures.cpp.o" \
"CMakeFiles/main.dir/csvReadWrite.cpp.o" \
"CMakeFiles/main.dir/distance.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/threshold.cpp.o
main: CMakeFiles/main.dir/cleanup.cpp.o
main: CMakeFiles/main.dir/segment.cpp.o
main: CMakeFiles/main.dir/computeFeatures.cpp.o
main: CMakeFiles/main.dir/csvReadWrite.cpp.o
main: CMakeFiles/main.dir/distance.cpp.o
main: CMakeFiles/main.dir/build.make
main: /opt/homebrew/lib/libopencv_gapi.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_stitching.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_alphamat.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_aruco.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_bgsegm.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_bioinspired.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_ccalib.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_dnn_objdetect.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_dnn_superres.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_dpm.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_face.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_freetype.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_fuzzy.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_hfs.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_img_hash.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_intensity_transform.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_line_descriptor.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_mcc.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_quality.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_rapid.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_reg.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_rgbd.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_saliency.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_sfm.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_stereo.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_structured_light.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_superres.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_surface_matching.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_tracking.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_videostab.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_viz.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_wechat_qrcode.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_xfeatures2d.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_xobjdetect.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_xphoto.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_shape.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_highgui.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_datasets.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_plot.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_text.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_ml.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_phase_unwrapping.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_optflow.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_ximgproc.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_video.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_videoio.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_imgcodecs.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_objdetect.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_calib3d.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_dnn.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_features2d.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_flann.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_photo.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_imgproc.4.8.1.dylib
main: /opt/homebrew/lib/libopencv_core.4.8.1.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build" "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 3/build/CMakeFiles/main.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend
