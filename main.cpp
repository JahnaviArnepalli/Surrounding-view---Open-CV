#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "camcalib.h"
#include "image_stitching.h"

int main(int argc, char** argv) {

    //calibrate_camera();
    std::vector<cv::String> newFileNames;
    cv::glob("../labimg_stitch/img*.jpg", newFileNames, false);
    cv::FileStorage fsLoad("calibration_parameters.yml", cv::FileStorage::READ);
    cv::Matx33f K;
    cv::Vec<float, 5> k;
    cv::Size frameSize(2048, 1536);
    fsLoad["camera_matrix"] >> K;
    fsLoad["distortion_coefficients"] >> k;

    fsLoad.release();  // Close the file
    // Undistort and display a different set of images using the initial parameters
    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY);

    // Show lens corrected images
   // Directory to save undistorted images in the local directory
    std::string undistortedFolderPath = "../for_stitch/";
    
    // Undistort and save the images
    for (std::size_t i = 0; i < newFileNames.size(); i++) {
        std::cout << std::string(newFileNames[i]) << std::endl;

        cv::Mat img = cv::imread(newFileNames[i], cv::IMREAD_COLOR);

        // Undistort the image using the initial calibration parameters
        cv::Mat imgUndistorted;
        cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);
        
        // Save the undistorted image to the folder in the local directory
        std::string undistortedImagePath = undistortedFolderPath + "undistorted_image_" + std::to_string(i) + ".jpg";
        // Check if the imwrite operation is successful
        if (cv::imwrite(undistortedImagePath, imgUndistorted)) {
            std::cout << "Image saved successfully: " << undistortedImagePath << std::endl;
        }
        else {
            std::cerr << "Error saving image: " << undistortedImagePath << std::endl;
        }
        cv::imwrite(undistortedImagePath, imgUndistorted);

        // Display
        cv::Size newSize(720, 540);  // Adjust the size as needed
        cv::resize(imgUndistorted, imgUndistorted, newSize);
        cv::imshow("undistorted image", imgUndistorted);
        cv::waitKey(1000);
    }

    stitch_images();

    return 0;
}