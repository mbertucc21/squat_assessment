# squat_assessment
Developed program to take in input images and determine if the subject (human) is completing a squat and provides feedback (please check out the 'outputs' folder).  Utilized OpenCV, NumPy and Matplotlib libraries.

Tested on random images obtained from google as well as test images of myself (which I chose not to upload).  Please note you will need to download the pre-trained pose estimation model to have this program work.  See details below.

SPECIAL NOTE:  THANK-YOU TO MR. SATYA MALLICK OF 'LEARNOPENCV' FOR THE SKELTON CODE AND ACCESS TO THE
PRE-TRAINED CNN MODELS UTILIZED TO DETECT THE KEY BODY FEATURES.  PLEASE CHECK OUT HIS GITHUB PROFILE:
https://github.com/spmallick.  THIS PARTICULAR CODE UTILIZES THE 'OPENPOSE' REPO FOUND HIS 'LEARNOPENCV'
DIRECTORY.  PLEASE VISIT https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
TO SEE HOW TO DOWNLOAD THE PRE-TRAINED MODEL NEEDED TO RUN THIS PROGRAM.

My general comments:

The Below Code will take in an input image (shot with an iphone) and resized to about 640 h x 480 w
(however, when developing this program, I did not constantly feed in 640 x 480 data).

I Created the below program to play around with the OpenPose Pre-Trained Model.  Most of the calculations below
are based on the x and y coordinates and % of their locations; however, to improve this code, it would have
been better to reshape so that all input images are 640 h x 480 w and used % of the input frame to calculate the
squat errors (hips, knees, ankles, back, etc.).

Right now, the below code places a higher 'y value' weight on the points lower in the body (i.e. ankles, knees)
as these points are higher on the y-scale (y increases as you go down).  As an example a ankle point at y=700
has a 10% of 70, whereas a hip point at y=500 has a 10% of 50, thereby making the weights of each point uneven).
As mentioned above, it would have been better to reshape the entire input image so that they are all consistent
(680 x 480) and used percentages of the frame rather than percentages of the point (i.e. 10% y-difference in a
680 x 480 frame would be a value of 68 across the whole image, rather than just at a specific point).

Other improvements can be made, especially if I was able to use a video input rather than just images and get
X, Y, and Z coordinates.  Just using image inputs for now as I am running this on my CPU and do not have a
GPU to run this on (running videos looks like a slide show on my computer).  In addition, X, Y, Z coordinates
would greatly help improve the accuracy of comments made by this squatting program, as I am currently only
using X and Y coordinates to make comments, and thus, the subject must be facing a specific position
(about 45 degrees towards the camera) in order to make accurate comments.

Will look at the tf-OpenPose Estimator in the future, as I see it is possible to map out X, Y and Z coordinates
for the that specific Pose Estimator; however, would greatly help if i had a GPU and fed in video inputs.
Conclusion:  This OpenPose Pre-Trained Model is very versatile and many measurements can be made from the
pose coordinates given.  I'm very surprised on how accurate this model runs as well, as I would expect something
like this to give me many more errors (i.e. locating incorrect points).   I can see why this model  won the
COCO keypoints challenge in 2016!

Lines of Code: ~400
