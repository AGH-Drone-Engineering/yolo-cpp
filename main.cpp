#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "yolocpp.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    vector<string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    YOLOCPP inf("yolov8n.onnx", 640, 640, std::move(classes));

    const string imagePath = "image.jpeg";
    Mat frame = imread(imagePath);

    auto start = chrono::high_resolution_clock::now();
    int cnt = 10;
    for (int i = 0; i < cnt; i++)
    {
        inf.detect(frame.data, frame.cols, frame.rows, frame.channels());
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double>(end - start).count() / cnt;
    cout << "Latency: " << duration * 1000 << " ms" << endl;
    cout << "FPS: " << 1 / duration << endl;

    vector<YOLOCPP::Detection> output = inf.detect(frame.data, frame.cols, frame.rows, frame.channels());

    int detections = output.size();
    cout << "Number of detections:" << detections << endl;

    for (int i = 0; i < detections; ++i)
    {
        YOLOCPP::Detection detection = output[i];

        Rect box(detection.x, detection.y, detection.w, detection.h);
        Scalar color(255, 0, 0);

        rectangle(frame, box, color, 2);

        string classString = detection.class_name + ' ' + to_string(detection.confidence).substr(0, 4);
        Size textSize = getTextSize(classString, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        rectangle(frame, textBox, color, cv::FILLED);
        putText(frame, classString, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }

    cv::imshow("Inference", frame);
    cv::waitKey(0);
}
