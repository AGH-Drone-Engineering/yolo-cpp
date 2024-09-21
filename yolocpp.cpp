#include "yolocpp.h"

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

static cv::Mat format_to_square(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

class YOLOCPP::Impl
{
public:
    Impl(std::string model_path, cv::Size input_shape, std::vector<std::string> classes)
        : _model_path(std::move(model_path))
        , _classes(std::move(classes))
        , _input_shape(std::move(input_shape))
    {
        _net = cv::dnn::readNetFromONNX(_model_path);
        std::cout << "\nRunning on CPU" << std::endl;
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<Detection> detect(
        const cv::Mat &input,
        float confidence_threshold,
        float score_threshold,
        float nms_threshold,
        bool letterbox)
    {
        cv::Mat model_input = input;
        if (letterbox && _input_shape.width == _input_shape.height)
            model_input = format_to_square(model_input);

        cv::Mat blob;
        cv::dnn::blobFromImage(model_input, blob, 1.0/255.0, _input_shape, cv::Scalar(), true, false);
        _net.setInput(blob);

        std::vector<cv::Mat> outputs;
        _net.forward(outputs, _net.getUnconnectedOutLayersNames());

        int rows = outputs[0].size[1];
        int dimensions = outputs[0].size[2];

        bool yolov8 = false;
        // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
        // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
        if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
        {
            yolov8 = true;
            rows = outputs[0].size[2];
            dimensions = outputs[0].size[1];

            outputs[0] = outputs[0].reshape(1, dimensions);
            cv::transpose(outputs[0], outputs[0]);
        }
        float *data = (float *)outputs[0].data;

        float x_factor = model_input.cols / _input_shape.width;
        float y_factor = model_input.rows / _input_shape.height;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < rows; ++i)
        {
            if (yolov8)
            {
                float *classes_scores = data+4;

                cv::Mat scores(1, _classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > score_threshold)
                {
                    confidences.push_back(max_class_score);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            else // yolov5
            {
                float confidence = data[4];

                if (confidence >= confidence_threshold)
                {
                    float *classes_scores = data+5;

                    cv::Mat scores(1, _classes.size(), CV_32FC1, classes_scores);
                    cv::Point class_id;
                    double max_class_score;

                    minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                    if (max_class_score > score_threshold)
                    {
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id.x);

                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];

                        int left = int((x - 0.5 * w) * x_factor);
                        int top = int((y - 0.5 * h) * y_factor);

                        int width = int(w * x_factor);
                        int height = int(h * y_factor);

                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }

            data += dimensions;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, nms_result);

        std::vector<Detection> detections{};
        for (unsigned long i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];

            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                                    dis(gen),
                                    dis(gen));

            result.className = _classes[result.class_id];
            result.box = boxes[idx];

            detections.push_back(result);
        }

        return detections;
    }

private:
    const std::string _model_path;
    const std::vector<std::string> _classes;
    const cv::Size _input_shape;

    cv::dnn::Net _net;
};

YOLOCPP::YOLOCPP(std::string model_path, cv::Size input_shape, std::vector<std::string> classes)
    : _impl(std::make_unique<Impl>(std::move(model_path), std::move(input_shape), std::move(classes)))
{
}

YOLOCPP::~YOLOCPP() = default;

std::vector<YOLOCPP::Detection> YOLOCPP::detect(const cv::Mat &input)
{
    return _impl->detect(input, confidence_threshold, score_threshold, nms_threshold, letterbox);
}
