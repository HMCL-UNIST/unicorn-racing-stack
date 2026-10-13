#ifndef POLYGON_FILTER_H
#define POLYGON_FILTER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class PolygonFilter {
public:
    PolygonFilter();
    ~PolygonFilter();

    // Load map info from a YAML file path (image file name, resolution, origin, thresholds, ...)
    bool loadMapFromYAML(const std::string& yamlPath, const std::string& imagePath);

    // Set the erosion kernel size and update the contours
    void setErosionKernelSize(int pix);

    // Convert pixel coordinates to global coordinates (global = origin + pixel * resolution)
    cv::Point2d pixelToGlobal(const cv::Point& pixelPoint);

    // Check whether the given global coordinate lies inside the external contour
    bool isPointInside(const cv::Point2d& globalPoint);

    // Visualize the contour result (for debugging)
    void visualizeContours();

private:
    // Process the image, then update erosion and contours
    void updateContours();

    cv::Mat image;             // image given in the YAML (grayscale)
    cv::Mat erodedImage;       // image after erosion
    cv::Mat kernel;            // kernel used for erosion
    cv::Size kernelSize;       // kernel size

    // Contour info
    std::vector<std::vector<cv::Point>> externalContours; // external contours
    std::vector<std::vector<cv::Point>> internalContours; // internal contours

    // Map parameters (read from the YAML)
    double resolution;         // resolution (real length per pixel)
    cv::Point2d origin;        // global origin corresponding to pixel (0,0)
    double free_thresh;
    double occupied_thresh;
    int negate;                // whether the occupancy map is inverted (0 or 1)
    bool debug;
};

#endif // POLYGON_FILTER_H
