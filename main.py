import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

corner_points = []
chessboard = (6, 9)
automatic_image_dir = "images/automatic_training"
manual_image_dir = "images/manual_training"
testing_image_dir = "images/testing"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DEBUG = False


def order_points(pts):
    """
    Orders the points of a quadrilateral in a specific order.

    Args:
        pts (numpy.ndarray): Array of points representing the quadrilateral vertices.

    Returns:
        numpy.ndarray: Array of points ordered in the specified order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_corners(img, points, chessboard_size):
    """
       Finds the corners of a chessboard on an image.

       Args:
           img (numpy.ndarray): The input image.
           points (list): List of corner points of the chessboard.
           chessboard_size (tuple): Tuple containing the number of inner corners per row and column.

       Returns:
           tuple: A tuple containing the image with lines and circles drawn on the corners and an array of all detected
           points.
       """
    tl, tr, br, bl = np.array(points)
    print(tl, tr, br, bl)
    all_points = []

    for j in range(chessboard_size[1]):
        # Interpolate vertical line points
        start_vert = tl + (tr - tl) * (j / (chessboard_size[1]-1))
        end_vert = bl + (br - bl) * ((chessboard_size[1]-1 - j) / (chessboard_size[1]-1))
        cv2.line(img, tuple(start_vert.astype(int)), tuple(end_vert.astype(int)), (0, 255, 0), 2)
        for p in np.linspace(start_vert, end_vert, chessboard_size[0]):
            all_points.append([p])
            cv2.circle(img, tuple(p.astype(int)), 2, (0, 0, 255), 3)

    all_points_np = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

    return img, all_points_np


def automatic_corner_detection(img, criteria, chessboard_size, showboard=False):
    """
    Automatically detects corners of a chessboard pattern on an image.

    Args:
        img (numpy.ndarray): The input image.
        criteria: Termination criteria for the iterative corner refinement process.
        chessboard_size (tuple): Tuple containing the number of inner corners per row and column.
        showboard (bool, optional): Whether to display the detected chessboard corners on the image.

    Returns:
        tuple: A tuple containing a boolean indicating successful detection and an array of detected corner points.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if showboard:
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("output", int(img.shape[1] / 3), int(img.shape[0] / 3))
            cv2.imshow('output', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return ret, corners


def automatic_corner_detection_live(img, criteria, chessboard_size):
    """
    Automatically detects corners of a chessboard pattern on a live image feed.

    Args:
        img (numpy.ndarray): The input image.
        criteria: Termination criteria for the iterative corner refinement process.
        chessboard_size (tuple): Tuple containing the number of inner corners per row and column.

    Returns:
        tuple: A tuple containing a boolean indicating successful detection and an array of detected corner points.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

    return ret, corners


def undistort(gray_test_img, mtx, dist):
    """
       Undistorts a grayscale image using camera calibration parameters.

       Args:
           gray_test_img (numpy.ndarray): Grayscale input image.
           mtx (numpy.ndarray): Camera matrix.
           dist (numpy.ndarray): Distortion coefficients.

       Returns:
           tuple: A tuple containing the new camera matrix and the undistorted color image.
       """
    h, w = gray_test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(gray_test_img, mtx, dist, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return newcameramtx, dst


def mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
      Computes the mean reprojection error of a camera calibration.

      Args:
          objpoints (list): List of 3D object points.
          imgpoints (list): List of 2D image points.
          mtx (numpy.ndarray): Camera matrix.
          dist (numpy.ndarray): Distortion coefficients.
          rvecs (list): List of rotation vectors.
          tvecs (list): List of translation vectors.

      Returns:
          None
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))


def online_phase(test_img, objp, mtx, dist):
    """
    Performs the online phase of camera calibration by detecting corners in a test image and visualizing 3D objects.

    Args:
        test_img (numpy.ndarray): The test image.
        objp (list): 3D object points.
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.

    Returns:
        None
    """
    ret, corners = automatic_corner_detection(test_img, criteria, chessboard)
    if not ret:
        print('Cant detect the corners of the test picture')

    ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

    # Define 3D coordinates for drawing axes
    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    # Define 3D coordinates for a cube
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])

    # Project the 3D points onto the 2D plane
    imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)

    # Draw the axes and cube
    dst_with_objects = draw(test_img, imgpts_axis, imgpts_cube)
    cv2.imshow('Output with 3D Objects', dst_with_objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def online_phase_live(frame, objp, mtx, dist):
    """
    Performs the online phase of camera calibration by detecting corners in a test image and visualizing 3D objects.

    Args:
        test_img (numpy.ndarray): The test image.
        objp (list): 3D object points.
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.

    Returns:
        None
    """
    ret, corners = automatic_corner_detection_live(frame, criteria, chessboard)

    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # Define 3D coordinates for a cube
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])
    if ret:
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)
        frame_with_objects = draw(frame, imgpts_axis, imgpts_cube)

        # Display the frame with objects drawn
        cv2.imshow('Webcam - Press Q to Quit', frame_with_objects)
    else:
        # Display the original frame if corner detection fails
        cv2.imshow('Webcam - Press Q to Quit', frame)


def draw(img, imgpts_axis, imgpts_cube):
    """
       Draws 3D axes and a cube on the input image.

       Args:
           img (numpy.ndarray): Input image.
           imgpts_axis (numpy.ndarray): Image points of the 3D axes.
           imgpts_cube (numpy.ndarray): Image points of the cube.

       Returns:
           numpy.ndarray: Image with 3D axes and cube drawn.
    """
    # Draw axes lines
    origin = tuple(imgpts_axis[0].ravel().astype(int))
    for pt in imgpts_axis[1:]:
        img = cv2.line(img, origin, tuple(pt.ravel().astype(int)), (255, 0, 0), 5)
    # Draw cube
    imgpts = np.int32(imgpts_cube).reshape(-1, 2)
    # Draw bottom
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    # Draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # Draw top
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def draw_circle(event, x, y, flags, param):
    """
       Draws a circle on the image when the left mouse button is clicked.

       Args:
           event (int): Type of mouse event.
           x (int): x-coordinate of the mouse position.
           y (int): y-coordinate of the mouse position.
           flags (int): Additional flags.
           param: Additional parameters.

       Returns:
           None
    """
    # Draw axes lines
    global corner_points
    # If the left mouse button was clicked, record the position and draw a circle there
    if event == cv2.EVENT_LBUTTONDOWN:
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)


def Run1():
    """
       Runs the calibration process.

       Returns:
           mtx (numpy.ndarray): Camera matrix.
           dist (numpy.ndarray): Distortion coefficients.
           objp (numpy.ndarray): World coordinates for 3D points.
    """
    objpoints = []
    imgpoints = []

    global img
    global corner_points
    corner_points = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    for filename in os.listdir(automatic_image_dir):
        path = os.path.join(automatic_image_dir, filename)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print("Automatic corner detection was succesfull for image " + path)
            objpoints.append(objp)
            imgpoints.append(corners)

    for filename in os.listdir(manual_image_dir):
        path = os.path.join(manual_image_dir, filename)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print("Automatic corner detection was succesfull for image " + path)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("FAIL to automatically detect corners for image " + path)
            cv2.namedWindow('Image')
            cv2.imshow('Image', img)
            cv2.setMouseCallback('Image', draw_circle)

            # Wait until the user has clicked four points
            while len(corner_points) < 4:
                cv2.waitKey(1)

            _, points = find_corners(img, corner_points, chessboard_size=chessboard)
            img = cv2.imread(path)
            img_with_chessboard = cv2.drawChessboardCorners(img, chessboard, points, True)
            corner_points = []

            # Show the result
            cv2.imshow('Chessboard', img_with_chessboard)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            objpoints.append(objp)
            imgpoints.append(points)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist, objp


def Run2():
    """
       Runs the calibration process with 10 images.

       Returns:
           mtx (numpy.ndarray): Camera matrix.
           dist (numpy.ndarray): Distortion coefficients.
           objp (numpy.ndarray): World coordinates for 3D points.
    """
    objpoints = []
    imgpoints = []
    global img
    global corner_points
    corner_points = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    images = os.listdir(automatic_image_dir)  # List all files in the directory
    images.sort()  # Sort the files to ensure consistent order

    # Use only the first 10 images
    for image_name in images[:10]:
        path = os.path.join(automatic_image_dir, image_name)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print(f"Automatic corner detection was successful for image {path}")
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"FAIL to detect corners for image {path}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist, objp


def Run3():
    """
       Runs the calibration process with 5 images.

       Returns:
           mtx (numpy.ndarray): Camera matrix.
           dist (numpy.ndarray): Distortion coefficients.
           objp (numpy.ndarray): World coordinates for 3D points.
    """
    objpoints = []
    imgpoints = []
    global img
    global corner_points
    corner_points = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    images = os.listdir(automatic_image_dir)  # List all files in the directory
    images.sort()  # Sort the files to ensure consistent order

    # Use only the first 5 images
    for image_name in images[:5]:
        path = os.path.join(automatic_image_dir, image_name)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print(f"Automatic corner detection was successful for image {path}")
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"FAIL to detect corners for image {path}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist, objp


def RunEnhanced():
    """
        Runs an enhanced calibration process.It is not used in this assignment.

        Returns:
            mtx (numpy.ndarray): Camera matrix.
            dist (numpy.ndarray): Distortion coefficients.
            objp (numpy.ndarray): World coordinates for 3D points.
    """
    objpoints = []
    imgpoints = []
    global img
    global corner_points
    corner_points = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    for i in range(10, 11):
        path = os.path.join(automatic_image_dir, f'IMG-20240212-WA0033.jpg')
        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Histogram Equalization
        gray = cv2.equalizeHist(gray)

        # Step 2: Gaussian Blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 3: Sobel Edge Detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = cv2.convertScaleAbs(sobel_combined)

        # Create a mask where edges are white and the rest is black
        edge_mask = cv2.threshold(sobel_combined,100, 255, cv2.THRESH_BINARY)[1]
        edges_colored = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Chessboard', edges_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        overlayed_img = cv2.addWeighted(img, 0.9, edges_colored, 0.1,0)

        cv2.imshow('Chessboard', overlayed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Detect chessboard corners
        ret, corners = automatic_corner_detection(overlayed_img, criteria, chessboard,showboard=False )
        if ret:
            if DEBUG:
                print("Automatic corner detection was succesfull for image " + path)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("FAIL to detect corners for image " + path)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist, objp


def RunCameraLocation():
    """
        Runs camera location estimation based on chessboard calibration images.

        Returns:
            mtx (numpy.ndarray): Camera matrix.
            dist (numpy.ndarray): Distortion coefficients.
            objp (numpy.ndarray): World coordinates for 3D points.
        """
    objpoints = []
    imgpoints = []
    global img
    global corner_points
    corner_points = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    images = os.listdir(automatic_image_dir)  # List all files in the directory
    images.sort()  # Sort the files to ensure consistent order

    # Use only the first 10 images
    for image_name in images[:10]:
        path = os.path.join(automatic_image_dir, image_name)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print(f"Automatic corner detection was successful for image {path}")
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"FAIL to detect corners for image {path}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    rotation_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Chessboard position (assuming it's at the origin)
    chessboard_position = (0, 0, 0)
    ax.scatter(*chessboard_position, c='red', marker='o', s=100, label='Chessboard')

    for idx, tvec in enumerate(tvecs):
        # Camera positions are the negation of the translation vectors
        cam_position = -tvec.reshape(3)
        ax.scatter(*cam_position, c='blue', label=f'Image {idx + 1}' if idx == 0 else "_nolegend_")

        # Annotate the point with the image index
        ax.text(*cam_position, f'{idx + 1}', color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Camera Positions Relative to Chessboard')
    ax.legend()
    plt.show()
    return mtx, dist, objp


def RunQualityDetection():
    """
    Runs quality detection on calibration of images and rejecting based on projection error threshold.

    Returns:
        mtx (numpy.ndarray): New Camera matrix.
        dist (numpy.ndarray): New Distortion coefficients.
        objp (numpy.ndarray):  New World coordinates for 3D points.
    """
    objpoints = []
    imgpoints = []
    accepted_indices = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    images = os.listdir(automatic_image_dir)  # List all files in the directory
    images.sort()  # Sort the files to ensure consistent order

    # Use only the first 20 images
    for image_name in images[:20]:
        path = os.path.join(automatic_image_dir, image_name)
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print(f"Automatic corner detection was successful for image {path}")
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"FAIL to detect corners for image {path}")

    # Calibrate the camera using the collected points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate and check the reprojection error for each image
    max_acceptable_error = 0.075  # Define the maximum acceptable reprojection error
    for i, (rvec, tvec, objp, imgp) in enumerate(zip(rvecs, tvecs, objpoints, imgpoints)):
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        if error <= max_acceptable_error:
            accepted_indices.append(i)  # Accept the image
            print(f"Image {i + 1} accepted with reprojection error: {error}")
        else:
            print(f"Image {i + 1} rejected due to high reprojection error: {error}")

    # Filter objpoints and imgpoints to include only those from accepted images
    filtered_objpoints = [objpoints[i] for i in accepted_indices]
    filtered_imgpoints = [imgpoints[i] for i in accepted_indices]

    # Calculate mean error using only the accepted images
    mean_error = np.mean([cv2.norm(imgpoints[i], cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)[0],
                                   cv2.NORM_L2) / len(imgpoints[i]) for i in accepted_indices])
    print(f"Total mean error after rejection: {mean_error}")
    # Recalibrate the camera using only the accepted images
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(filtered_objpoints, filtered_imgpoints, img.shape[1::-1], None,
                                                       None)

    return mtx, dist, objp


def live_camera(mtx, dist, objp):
    """
    Displays live camera feed and performs camera location estimation.

    Args:
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.
        objp (numpy.ndarray): World coordinates for 3D points.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video capture.")
        return
    while True:
        ret, frame = cap.read()
        processed_frame = frame.copy()
        if not ret:
            print("Failed to grab frame")
            break

        online_phase_live(processed_frame, objp, mtx, dist)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    test_img = cv2.imread(os.path.join(testing_image_dir, sorted(os.listdir(testing_image_dir))[0]))
    print("Run 1: Offline Phase")
    mtx, dist, objp = Run1()
    print(mtx)
    print("Run 1: Online Phase")
    online_phase(test_img, objp, mtx, dist)
    test_img = cv2.imread(os.path.join(testing_image_dir, sorted(os.listdir(testing_image_dir))[0]))
    print("\nRun 2: Offline Phase")
    mtx, dist, objp = Run2()
    print(mtx)
    print("Run 2: Online Phase")
    online_phase(test_img, objp, mtx,dist)
    test_img = cv2.imread(os.path.join(testing_image_dir, sorted(os.listdir(testing_image_dir))[0]))
    print("\nRun 3: Offline Phase")
    mtx, dist, objp = Run3()
    print(mtx)
    print("Run 3: Online Phase")
    online_phase(test_img, objp, mtx, dist)

    # CHOICES
    live_camera(mtx, dist, objp)
    print("\nRun Training With Camera Location")
    mtx, dist, objp = RunCameraLocation()
    print("\nRun Without Low-Quality Images")
    mtx, dist, objp = RunQualityDetection()
    test_img = cv2.imread(os.path.join(testing_image_dir, sorted(os.listdir(testing_image_dir))[0]))
    print("Run Without Low-Quality Images: Online Phase")
    online_phase(test_img, objp, mtx, dist)


if __name__ == "__main__":
    main()
