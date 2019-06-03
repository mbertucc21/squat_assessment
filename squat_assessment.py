"""
SPECIAL NOTE:  THANK-YOU TO MR. SATYA MALLICK OF 'LEARNOPENCV' FOR THE SKELTON CODE AND ACCESS TO THE
PRE-TRAINED CNN MODELS UTILIZED TO DETECT THE KEY BODY FEATURES.  PLEASE CHECK OUT HIS GITHUB PROFILE:
https://github.com/spmallick.  THIS PARTICULAR CODE UTILIZES THE 'OPENPOSE' REPO FOUND HIS 'LEARNOPENCV'
DIRECTORY.  PLEASE VISIT https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
TO SEE HOW TO DOWNLOAD THE PRE-TRAINED MODEL NEEDED TO RUN THIS PROGRAM.

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

"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

##############################################################
# --- STEP 1: DOWNLOAD MODEL WEIGHTS --- (.caffeemodel files)
##############################################################

# https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

########################################
# --- STEP 2: LOAD THE NETWORK ---
########################################

# Choose either MPI (15 points) or COCO (18 points) output format
MODE = "MPI"  # Use MPI, less points, don't care about eyes at this point

# These will all be set in the if statement below
protoFile = None
weightsFile = None
nPoints = None
POSE_PAIRS = None
squat_pose_pairs = None

# if MODE is "COCO":
#     protoFile = "pose/coco/pose_deploy_linevec.prototxt"
#     weightsFile = "pose/coco/pose_iter_440000.caffemodel"
#
#     # FOR ALL POINTS
#     nPoints = 18
#     POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],
#                    [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

# elif MODE is "MPI" :  # Remove if state below and use this elif if un-commented

if MODE is 'MPI':
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

    # FOR ALL POINTS
    nPoints = 15  # 14 body points + 1 for background
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                  [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

    # FOR SQUAT POINTS
    # n_squat_points = 7  # 6 body points + 1 for background
    # squat_pose_pairs = [[1, 14], [14, 8], [14, 11], [8, 9], [11, 12], [9, 10], [12, 13]]


###################################################
# --- STEP 3: READ IMAGE AND PREPARE THE INPUT ---
###################################################

# Read in the image
img = "me_0001"  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< INPUT IMG HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

frame = cv2.imread("./inputs/" + img + ".jpg")  # Change to .jpeg or .png if necessary
print(frame.shape)  # height, width, color channels
print('\n')

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

# Copy of frame for different exercises/outputs to be shown on the frame
squat_frame = np.copy(frame)

# Load the files into memory using below
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()

# Specify the input image dimensions ???  (Not sure how this works, but these numbers seem to work well)
# inWidth and inHeight is used below for blobFromImage, but according to OpenCV documentation, the parameter
# where they are being placed is in 'size' which is the 'spatial size for output image'
inWidth = 368
inHeight = 368

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
# For blobFromImage, see: https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
# sqapRB --> flag which indicates that swap first and last channels in 3-channel image is necessary.
# crop --> 	flag which indicates whether image will be cropped after resize or not

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)


#######################################################
# --- STEP 4: MAKE PREDICTIONS AND PARSE KEYPOINTS ---
#######################################################

# The forward method (below) for the DNN class in OpenCV makes a forward pass through the network which is just
# another way of saying it is making a prediction. Note that the Output is a 4D matrix:
# - image ID (in case you pass more than one image)
# - index of a keypoint.  Produces confidence maps and parts affinity maps which are all concatenated.
#   (COCO produces 57 parts --> 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps
#   Similarly for MPI, it produces (44 points).  We will be using only the first few points which correspond
#   to key points
# - The third dimension is the height of the output map and the fourth dimension is the width of the output map

output = net.forward()
print(output.shape)  # 4 dimensions (image, points, height of output map, width)
print('\n')

print("time taken by network : {:.3f}".format(time.time() - t))
print('\n')

# We check whether each keypoint is present in the image or not.  We get the location of the keypoint by
# finding the maxima of the confidence map of that keypoint.  We also use a threshold to reduce false detections

H = output.shape[2]  # 46
W = output.shape[3]  # 46

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    # print('FOR POINT {}:'.format(i))
    # print('x = {}'.format(x))
    # print('y = {}'.format(y))
    # print('============================')

    if prob > threshold:
        cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

#######################################################################################################################
# ---------------------------- STEP 5: MY CODE: Determine Exercise and Make Comments ----------------------------------
#######################################################################################################################
location_list = ['00 - Head: ',
                 '01 - Neck: ',
                 '02 - R Shoulder: ',
                 '03 - R Elbow: ',
                 '04 - R Wrist: ',
                 '05 - L Shoulder: ',
                 '06 - L Elbow: ',
                 '07 - L Wrist: ',
                 '08 - R Hip: ',
                 '09 - R Knee: ',
                 '10 - R Ankle: ',
                 '11 - L Hip: ',
                 '12 - L Knee: ',
                 '13 - L Ankle: ',
                 '14 - Chest: ',
                 '15 - Background: ']

for i in range(nPoints):
    print(location_list[i] + str(points[i]))

print('\n')

head = points[0]
neck = points[1]
right_shoulder = points[2]
left_shoulder = points[5]
right_hip = points[8]
left_hip = points[11]
right_knee = points[9]
left_knee = points[12]
right_ankle = points[10]
left_ankle = points[13]
chest = points[14]

# X coordinate index at [0] and Y coordinate index at [1]
x = 0
y = 1

#########################################
# EVEN POSE
#########################################
even_pose = False

y_even_hips = (left_hip[y] * 0.9, right_hip[y], left_hip[y] * 1.1)
y_even_knees = (left_knee[y] * 0.9, right_knee[y], left_knee[y] * 1.1)
y_even_ankles = (left_ankle[y] * 0.9, right_ankle[y], left_ankle[y] * 1.1)

# Check --> Are Hips Even (y-dir)? (10% margin)
print('Checking to see if hips are even...')
print('Hip Coordinates (with 10% margin): ' + str(y_even_hips))
if (left_hip[y] * 0.9) <= right_hip[y] <= (left_hip[y] * 1.1):
    print('***CHECK COMPLETE: Even Hips', '\n')

    # Check --> Are Knees Even (y-dir)? (10% margin)
    print('Checking to see if knees are even...')
    print('Knee Coordinates (with 10% margin): ' + str(y_even_knees))
    if (left_knee[y] * 0.9) <= right_knee[y] <= (left_knee[y] * 1.1):
        print('***CHECK COMPLETE: Even Knees', '\n')

        # Check --> Are Ankles Even (y-dir)? (10% margin)
        print('Checking to see if ankles are even...')
        print('Ankle Coordinates (with 10% margin): ' + str(y_even_ankles))
        if (left_ankle[y] * 0.9) <= right_ankle[y] <= (left_ankle[y] * 1.1):
            print('***CHECK COMPLETE: Even Ankles', '\n')

            # If hips, knees and ankles are even, then even_pose = True
            even_pose = True

if not even_pose:
    print('Non Even Pose')
if even_pose:
    print('Even Pose')

print('\n')

#########################################
# SQUATTING FORM
#########################################
squatting = False

left_hips_from_knees = (left_hip[y] * 0.80, left_knee[y], left_hip[y] * 1.32)
right_hips_from_knees = (right_hip[y] * 0.80, right_knee[y], right_hip[y] * 1.32)

tucked_r_ankles_facing_left = (right_hip[x] * 0.9, right_ankle[x], right_knee[x] * 1.1)
tucked_l_ankles_facing_left = (left_hip[x] * 0.9, left_ankle[x], left_knee[x] * 1.1)

tucked_r_ankles_facing_right = (right_hip[x] * 1.1, right_ankle[x], right_knee[x] * 0.9)
tucked_l_ankles_facing_right = (left_hip[x] * 1.1, left_ankle[x], left_knee[x] * 0.9)

# if even_pose = True (above)
if even_pose:

    # Check --> Are 20% < Hips < 32% from knees (y dir)
    print('Checking hips from knees...')
    print('Left hip from knees (y-dir): ' + str(left_hips_from_knees))
    print('Right hip from knees (y-dir): ' + str(right_hips_from_knees))
    if (left_hip[y] * 0.80) <= left_knee[y] <= (left_hip[y] * 1.32) and \
            (right_hip[y] * 0.80) <= right_knee[y] <= (right_hip[y] * 1.32):
        print('***CHECK COMPLETE: Hips +/- 20% from knees', '\n')

        # Check --> Are Ankles Tucked? (x dir)
        print('Checking ankle tuck...')
        print('If subject is facing left, left ankle tuck (x-dir): ' + str(tucked_r_ankles_facing_left))
        print('If subject is facing left, right ankle tuck (x-dir): ' + str(tucked_l_ankles_facing_left))
        print('If subject is facing right, right ankle tuck (x-dir): ' + str(tucked_r_ankles_facing_right))
        print('If subject is facing right, left ankle tuck (x-dir): ' + str(tucked_l_ankles_facing_right))

        # Subject Facing Left
        if 0.9 * right_hip[x] <= right_ankle[x] <= right_knee[x] * 1.1 and \
                0.9 * left_hip[x] <= left_ankle[x] <= left_knee[x] * 1.1:
            print('***CHECK COMPLETE: Ankles tucked between knees and hips', '\n')
            squatting = True

        # Subject Facing Right
        if 1.1 * right_hip[x] >= right_ankle[x] >= right_knee[x] * 0.9 and \
                1.1 * left_hip[x] >= left_ankle[x] >= left_knee[x] * 0.9:
            print('***CHECK COMPLETE: Ankles tucked between knees and hips', '\n')
            squatting = True

    elif left_knee[y] < (left_hip[y] * 0.80):
        print('Your hips are too far below your knees')
    else:
        print('You are not bending your hips low enough')


print('\n')

if not squatting:
    print('Non Squatting Form')

if squatting:
    print('#' * 20)
    print('SUBJECT IS SQUATTING')
    print('#' * 20)

print('\n')

#########################################
# IMPROVEMENT COMMENTS
#########################################

if even_pose and squatting:

    # Using Hip/Knee Measurements from above, see if hips need to be lowered:
    Low_Hips = False
    left_hips_from_knees2 = (left_knee[y], left_hip[y] * 1.23)
    right_hips_from_knees2 = (right_knee[y], right_hip[y] * 1.23)

    # Check --> Are Hips low enough (y dir)
    print('Checking if you should lower hips...')
    print(left_hips_from_knees2)
    print(right_hips_from_knees2)

    if left_knee[y] <= (left_hip[y] * 1.23) and right_knee[y] <= (right_hip[y] * 1.23):
        print('***CHECK COMPLETE: Hips in good position', '\n')
        Low_Hips = True
    else:
        print("Move your hips a little lower", '\n')

    # Checking to see slope from hips to chest
    Posture = False
    hips_x = (left_hip[x] + right_hip[x]) / 2
    hips_y = (left_hip[y] + right_hip[y]) / 2
    # NOTE: y increases as you go down, so switch Y2 and Y1 and X2 and X1
    hip_to_chest_slope = abs((chest[y] - hips_y)/(chest[x] - hips_x))

    # Check --> Is Back Straight(x and y dir)?
    print('Checking posture...')
    print('Hips to chest slope is: ' + str(hip_to_chest_slope))

    if 1.0 < hip_to_chest_slope < 4.0:
        print('***CHECK COMPLETE: Chest/Hips in good position', '\n')
        Posture = True
    elif hip_to_chest_slope < 1.4:
        print('You are leaned too far forward, pull your chest backwards', '\n')
    else:
        print('You are leaned too far backwards, push your chest forward', '\n')

    # Check --> Are Shoulders lined up with ankles? (x dir)
    Shoulders = False
    l_shoulders_ankles = (left_ankle[x] * 0.80, left_shoulder[x], left_ankle[x] * 1.20)
    r_shoulders_ankles = (right_ankle[x] * 0.80, right_shoulder[x], right_ankle[x] * 1.20)

    print('Checking Shoulders and Ankles...')
    print('Left shoulder +/- 20% from ankles (x-dir): ' + str(l_shoulders_ankles))
    print('Right shoulder +/- 20% from ankles (x-dir): ' + str(r_shoulders_ankles))

    if left_ankle[x] * 0.80 <= left_shoulder[x] <= left_ankle[x] * 1.20 and \
            right_ankle[x] * 0.80 <= right_shoulder[x] <= right_ankle[x] * 1.20:
        print('***CHECK COMPLETE: Shoulders in good position relative to ankles', '\n')
        Shoulders = True

    elif left_ankle[x] * 0.80 <= right_shoulder[x] <= left_ankle[x] * 1.20 and \
            right_ankle[x] * 0.80 <= left_shoulder[x] <= right_ankle[x] * 1.20:
        print('Image Detection may be reversed, please re-take photo')

    else:
        print('Line up your shoulders with your ankles')

    # Give Comments
    if Shoulders and Posture and Low_Hips:
        print('\n')
        print('Great Squatting Form!!!')

    if not Shoulders or not Posture or not Low_Hips:
        print('\n')
        print('Adjust your squatting Form using the comments above')


###############################
# --- STEP 6: Show Outputs ---
###############################

# Since we know the indices of the points before-hand, we can draw the skelton when we have the keypoints
# by just joining the pairs.  This is done using the code given below (for loop).

# --- SQUAT SKELETON ---
# squat_pose_pairs = [[1, 14], [14, 8], [14, 11], [8, 9], [11, 12], [9, 10], [12, 13]]
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:

        cv2.line(squat_frame, points[partA], points[partB], (0, 255, 0), 2)
        cv2.circle(squat_frame, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

# cv2.imshow('Output-Skeleton', frame)
# cv2.imshow('Output-Keypoints', squat_frame)

cv2.imwrite('./outputs/' + img + '-Keypoints.jpg', frame)
cv2.imwrite('./outputs/' + img + '-Squat_Form.jpg', squat_frame)

# print('\n')
# print("Total time taken : {:.3f}".format(time.time() - t))
# print('\n')

plt.imshow(frame)  # Key Points
plt.show()

plt.imshow(squat_frame)  # Squat Skeleton Points
plt.show()

cv2.waitKey(0)
