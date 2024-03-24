Samuel Lee
Github URL: https://github.com/samusamusam/CS5330_Spring2024
Video Link: https://drive.google.com/file/d/1x46YLrpiistNxpExy-5QiC7zT87VrYAg/view?usp=drive_link
OS: Mac
IDE: VS Code
Instructions to Run Program
- build the entire project
- I included my calibration data .yml file, so no need to re-calibrate if you don't want to.
- ./calibrate - program to capture images of the target to calibrate
	- type 's' to save images
	- type 'c' to calibrate images and create corresponding matrices and data
	- type 'q' to quit
- ./AR - program to show 3D axes and 3D object in 2D image plane in live video feed for single target
	- type 'q' to quit
- ./AR2 - program to show 3D axes and 3D object in 2D image plane in live video feed for multiple targets
	- type 'q' to quit
- ./feature - program to test feature detection with Harris corners
	- type 'q' to quit
- ./generateAR - program to generate augmented reality image/video with user input
	- follow terminal instructions
Instructions to Test Extensions
- run the program ./AR2 and ./generateAR
- for AR2, have two chessboards appear in the frame to see corresponding objects
- for generateAR, make sure to follow terminal instructions and input the correct image file or video file with the chessboard target showing
Travel Days: N/A