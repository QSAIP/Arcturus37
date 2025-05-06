# ifndef ARCTRUS37_EYE_HPP
    /**
     * @file Eye.hpp
     * @author Ramtin Kosari (ramtinkosari@gmail.com)
     * @brief Arcturus37 Eye Module
     * @date 2025-05-06
     */
    # define ARCTRUS37_EYE_HPP
    //-- Include FStream
    # ifndef _GLIBCXX_FSTREAM
        #include <fstream>
    # endif // _GLIBCXX_FSTREAM
    //-- Include Vector
    # ifndef _GLIBCXX_VECTOR
        #include <vector>
    # endif // _GLIBCXX_VECTOR
    //-- Include String
    # ifndef _GLIBCXX_STRING
        #include <string>
    # endif // _GLIBCXX_STRING
    //-- Include Random
    # ifndef _GLIBCXX_RANDOM
        #include <random>
    # endif // _GLIBCXX_RANDOM
    //-- Include OpenCV
    # ifndef OPENCV_ALL_HPP
        # include <opencv2/opencv.hpp>
    # endif // OPENCV_ALL_HPP
    //-- Include OpenCV Image Processing
    # ifndef OPENCV_IMGPROC_HPP
        # include <opencv2/imgproc.hpp>
    # endif // OPENCV_IMGPROC_HPP
    //-- Include OpenCV DNN
    # ifndef OPENCV_DNN_HPP
        # include <opencv2/dnn.hpp> 
    # endif // OPENCV_DNN_HPP
    //-- Include Getopt
    # ifndef _GETOPT_H
        # include <getopt.h>
    # endif // _GETOPT_H
    //-- Include Socket
    # ifndef _SYS_SOCKET_H
        # include <sys/socket.h>
    # endif // _SYS_SOCKET_H
    //-- Include Netinet
    # ifndef _ARPA_INET_H
        # include <arpa/inet.h>
    # endif // _ARPA_INET_H
    //-- Include Unistd
    # ifndef _UNISTD_H
        # include <unistd.h>
    # endif // _UNISTD_H
    //-- BBox Color
    # define ARC37_BBOX_COLOR cv::Scalar(0, 255, 255)
    //-- Model Path
    # define ARC37_MODEL_PATH "/home/qb/Files/Libraries/Ultralytics/ultralytics/yolo11s.onnx"
    //-- Run on CPU(0) or GPU(1)
    # define ARC37_RUN_ON_GPU 1
    //-- Confidence Threshold
    # define ARC37_CONFIDENCE_THRESHOLD 0.73
    //-- Default Detection Class
    # define ARC37_DEFAULT_DETECTION_CLASS "person"
    //-- View Field Sections
    # define ARC37_VIEW_FIELD_SECTIONS 9
    //-- Update Delay
    # define ARC37_UPDATE_DELAY 60
    //-- Share Data on Network
    # define ARC37_SHARE_DATA 0
    //-- Arcturus37 Destination IP
    # define ARC37_DESTINATION_IP "192.168.0.104"
    //-- Arcturus37 Destination Port
    # define ARC37_DESTINATION_PORT 12345
    //-- Check if Project is Running by RKACPB
    # define RKACPB
    # ifdef RKACPB
        //-- Include RKLogger
        # ifndef RKLOGGER
            # include "RKLogger.hpp"
        # endif // RKLOGGER
    # endif // RKACPB
    /**
     * @brief Detection Struct
     * @struct Detection
     * @param className Name of the Detected Class
     * @param color Color of the Detected Class
     * @param confidence Confidence of the Detection
     * @param class_id ID of the Detected Class
     * @param box Bounding Box of the Detected Class
     * @details This struct is used to store the information of the detected object.
     */
    struct Detection {
        std::string className   {};
        cv::Scalar color        {};
        float confidence        {0.0};
        int class_id            {0};
        cv::Rect box            {};
    };
    /**
     * @brief Inference Class
     * @class Inference
     * @details This class is used to perform inference on the input image.
     */
    class Inference {
        private:
            /**
             * @brief Deep Neural Network Object
             */
            cv::dnn::Net net;
            /**
             * @brief Confidence Threshold for the Model
             */
            float modelConfidenceThreshold {0.25};
            /**
             * @brief Model Score Threshold for the Model
             */
            float modelScoreThreshold      {0.45};
            /**
             * @brief Non-Maximum Suppression Threshold for the Model
             */
            float modelNMSThreshold        {0.50};
            /**
             * @brief Model Path
             */
            std::string modelPath{};
            /**
             * @brief Classes Path
             */
            std::string classesPath{};
            /**
             * @brief Classes Vector
             */
            std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
            /**
             * @brief Model Input Shape
             */
            cv::Size2f modelShape   {};
            /**
             * @brief Cuda Flag
             */
            bool cudaEnabled        {};
            /**
             * @brief Letter Box
             */
            bool letterBoxForSquare = true;
            /**
             * @brief Method to Load Classes from File
             */
            void loadClassesFromFile();
            /**
             * @brief Method to Load ONNX Network
             */
            void loadOnnxNetwork();
            /**
             * @brief Method to Format Image to Square
             * @param source Source Image
             * @param scale Scale Factor
             * @param pad_x X Padding
             * @param pad_y Y Padding
             * @return cv::Mat 
             */
            cv::Mat formatToSquare(
                const cv::Mat   &   source,
                float           *   scale,
                int             *   pad_x,
                int             *   pad_y
            );
        public:
            /**
             * @brief Focus on the Object
             */
            std::string focusOn, lastFocusOn;
            /**
             * @brief Last Region of Interest
             */
            int lastRegion = -1;
            /**
             * @brief Threshold for Debouncing
             * @details Minimum number of frames the region should stay the same before sending
             */
            int debounceThreshold = 2;
            /**
             * @brief Stable Frame Count
             * @details Number of frames the region has been stable
             */
            int stableFrameCount = 0;
            /**
             * @brief Stable Region
             * @details Region of interest that is stable
             */
            int stableRegion = -1;
            /**
             * @brief Construct a New Inference Object
             * @param onnxModelPath Path to the ONNX Model
             * @param modelInputShape Input Shape of the Model
             * @param classesTxtFile Path to the Classes Text File
             * @param runWithCuda Run the Model with CUDA
             */
            Inference(
                const std::string   &   onnxModelPath,
                const cv::Size      &   modelInputShape = {640, 640},
                const std::string   &   classesTxtFile = "",
                const bool          &   runWithCuda = true
            );
            /**
             * @brief Method to Run Inference on Input Matrix
             * @param input 
             * @return std::vector<Detection> 
             */
            std::vector<Detection> runInference(
                const cv::Mat   &   input
            );
    };
# endif // ARCTRUS37_EYE_HPP