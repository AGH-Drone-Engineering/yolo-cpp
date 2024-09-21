#ifndef YOLOCPP_H
#define YOLOCPP_H

#include <vector>
#include <string>
#include <memory>

#include <opencv2/core.hpp>

class YOLOCPP
{
public:
    struct Detection
    {
        int class_id;
        std::string class_name;
        float confidence;
        cv::Scalar color;
        cv::Rect box;
    };

    YOLOCPP(std::string model_path, cv::Size input_shape, std::vector<std::string> classes);

    ~YOLOCPP();

    std::vector<Detection> detect(const cv::Mat &input);

    float confidence_threshold = 0.25;
    float score_threshold = 0.45;
    float nms_threshold = 0.50;
    bool letterbox = true;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

#endif // YOLOCPP_H
