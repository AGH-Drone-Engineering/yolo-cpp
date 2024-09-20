#ifndef INFERENCE_H
#define INFERENCE_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id;
    std::string className;
    float confidence;
    cv::Scalar color;
    cv::Rect box;
};

class Inference
{
public:
    Inference(std::string onnxModelPath, cv::Size modelInputShape, std::vector<std::string> classes);
    std::vector<Detection> runInference(const cv::Mat &input);

    float modelConfidenceThreshold = 0.25;
    float modelScoreThreshold = 0.45;
    float modelNMSThreshold = 0.50;
    bool letterBoxForSquare = true;

private:
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    const std::string modelPath;
    const std::vector<std::string> classes;
    const cv::Size2f modelShape;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
