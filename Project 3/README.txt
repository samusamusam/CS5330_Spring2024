Name: Samuel Lee and Anjith Prakash Chathan Kandy
URLs to Video(s): https://drive.google.com/file/d/1AQq8gdA_31ZiAcrtJ1z6WdMZaS2YgxZT/view
OS: Mac (Sam) and Windows (Anjith)
IDE: VS Code
Instructions to Run Program
- build the program and run it (according to OS and method of running OpenCV applications)
- once program is running, there should be multiple windows showing different version of the object recognition program
- windows "Live Video" should show you the original unedited video stream
- windows "Thresholded" should show you the thresholded binary image video stream
- windows "Cleaned" should show you the morphological filters placed on the binary image video stream
- windows "Coloured" should show you the regions colored in different colors in the video stream
- windows "Features" should show you the regions bounded by a box, with axis of least moments and percentage filled shown, as well as the most similar object classification label
- to enter training mode, type "T"; this should highlight only the most central object shown in the video stream
- in training mode, type "N" to capture the feature vector of the highlighted region
- to exit training mode, type "T" again
- to exit the program, type "q' at any time
Testing Extensions
- check out our CSV of training data on 10 objects
- in the "Features" window you can see multiple objects recognized if you have multiple objects in the frame
- if there is an unknown object, the label will say "UNKNOWN"
- check our code for cosine distance; we didn't implement it in our main.cpp due to wanting to stick with scaled Euclidean as provided in the assignment
Travel Days: 1 day