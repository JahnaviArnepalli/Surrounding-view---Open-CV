#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>
#include "image_stitching.h"

void stitch_images() {

    std::vector<std::string> image_paths;
    cv::glob("../for_stitch/*.jpg", image_paths);

    cv::Size newSize(720, 540);  // Adjust the size as needed
    std::vector<cv::Mat> images;

    for (const auto& image_path : image_paths) {
        cv::Mat img = cv::imread(image_path);
        images.push_back(img);
        cv::resize(img, img, newSize);
        cv::imshow("Image", img);
        cv::waitKey(0);
    }
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Stitcher::Status status;
    cv::Mat stitched_img;

    status = stitcher->stitch(images, stitched_img);

    if (status == cv::Stitcher::Status::OK) {
        cv::imwrite("stitchedOutput.png", stitched_img);
        cv::imshow("Stitched Img", stitched_img);
        cv::waitKey(0);

        cv::Mat bordered_img;
        cv::copyMakeBorder(stitched_img, bordered_img, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        cv::Mat gray;
        cv::cvtColor(stitched_img, gray, cv::COLOR_BGR2GRAY);
        cv::Mat thresh_img;
        cv::threshold(gray, thresh_img, 0, 255, cv::THRESH_BINARY);

        cv::imshow("Threshold Image", thresh_img);
        cv::waitKey(0);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh_img.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> areaOI = *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
            });
        cv::Mat mask = cv::Mat::zeros(thresh_img.size(), CV_8UC1);
        cv::Rect boundingRect = cv::boundingRect(areaOI);
        cv::rectangle(mask, boundingRect.tl(), boundingRect.br(), 255, -1);

        cv::Mat minRectangle = mask.clone();
        cv::Mat sub = mask.clone();

        int maxIterations = 100;  // Set a reasonable maximum number of iterations
        int iterationCount = 0;

        while (cv::countNonZero(sub) > 0 && iterationCount < maxIterations) {
            cv::erode(minRectangle, minRectangle, cv::Mat());
            cv::subtract(minRectangle, thresh_img, sub);
            iterationCount++;
        }

        cv::findContours(minRectangle.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        areaOI = *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
            });

        cv::imshow("minRectangle Image", minRectangle);
        cv::waitKey(0);

        boundingRect = cv::boundingRect(areaOI);
        stitched_img = stitched_img(boundingRect);

        cv::imwrite("stitchedOutputProcessed.png", stitched_img);
        cv::resize(stitched_img, stitched_img, newSize);
        cv::imshow("Stitched Image Processed", stitched_img);
        cv::waitKey(0);
    }
    else {
        std::cout << "Images could not be stitched!" << std::endl;
        std::cout << "Likely not enough keypoints being detected!" << std::endl;
    }
}
