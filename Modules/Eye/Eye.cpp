# ifndef ARCTRUS37_EYE_HPP
    # include "Eye.hpp"
    /**
     * @brief Method to Load Classes from File
     */
    void Inference::loadClassesFromFile() {
        std::ifstream inputFile(classesPath);
        if (inputFile.is_open()) {
            std::string classLine;
            while (std::getline(inputFile, classLine))
                classes.push_back(classLine);
            inputFile.close();
        }
    }
    /**
     * @brief Method to Load ONNX Network
     */
    void Inference::loadOnnxNetwork() {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (cudaEnabled) {
            // logger(ARC37_LABEL INFO "Running on CUDA");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            // logger(ARC37_LABEL INFO "Running on CPU");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }
    /**
     * @brief Construct a New Inference Object
     * @param onnxModelPath Path to the ONNX Model
     * @param modelInputShape Input Shape of the Model
     * @param classesTxtFile Path to the Classes Text File
     * @param runWithCuda Run the Model with CUDA
     */
    Inference::Inference(
        const std::string   &   onnxModelPath,
        const cv::Size      &   modelInputShape,
        const std::string   &   classesTxtFile,
        const bool          &   runWithCuda
    ) {
        modelPath = onnxModelPath;
        modelShape = modelInputShape;
        classesPath = classesTxtFile;
        cudaEnabled = runWithCuda;
        loadOnnxNetwork();
        // loadClassesFromFile();   -- Classes are Hardcoded
        focusOn = ARC37_DEFAULT_DETECTION_CLASS;
        lastFocusOn = ARC37_DEFAULT_DETECTION_CLASS;
    }
    /**
     * @brief Method to Format Image to Square
     * @param source Source Image
     * @param scale Scale Factor
     * @param pad_x X Padding
     * @param pad_y Y Padding
     * @return cv::Mat 
     */
    cv::Mat Inference::formatToSquare(
        const cv::Mat   &   source,
        float           *   scale,
        int             *   pad_x,
        int             *   pad_y
    ){
        int col = source.cols;
        int row = source.rows;
        int m_inputWidth = modelShape.width;
        int m_inputHeight = modelShape.height;
        *scale = std::min(m_inputWidth / (float)col, m_inputHeight / (float)row);
        int resized_w = col * *scale;
        int resized_h = row * *scale;
        *pad_x = (m_inputWidth - resized_w) / 2;
        *pad_y = (m_inputHeight - resized_h) / 2;
        cv::Mat resized;
        cv::resize(source, resized, cv::Size(resized_w, resized_h));
        cv::Mat result = cv::Mat::zeros(m_inputHeight, m_inputWidth, source.type());
        resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));
        resized.release();
        return result;
    }
    /**
     * @brief Method to Run Inference on Input Matrix
     * @param input 
     * @return std::vector<Detection> 
     */
    std::vector<Detection> Inference::runInference(
        const cv::Mat   &   input
    ) {
        cv::Mat modelInput = input;
        int pad_x, pad_y;
        float scale;
        if (letterBoxForSquare && modelShape.width == modelShape.height) {
            modelInput = formatToSquare(
                modelInput,
                &   scale,
                &   pad_x,
                &   pad_y
            );
        }
        cv::Mat blob;
        cv::dnn::blobFromImage(
            modelInput,
            blob,
            1.0 / 255.0,
            modelShape,
            cv::Scalar(),
            true,
            false
        );
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        int rows = outputs[0].size[1];
        int dimensions = outputs[0].size[2];
        //-- Handle YOLOv8 by Checking if the shape[2] is More than shape[1]
        bool yolov8 = false;
        if (dimensions > rows) {
            yolov8 = true;
            rows = outputs[0].size[2];
            dimensions = outputs[0].size[1];
            outputs[0] = outputs[0].reshape(1, dimensions);
            cv::transpose(outputs[0], outputs[0]);
        }
        float *data = (float *)outputs[0].data;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        for (int i = 0; i < rows; ++i) {
            //-- Handle Models
            if (yolov8) {
                float *classes_scores = data+4;
                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > modelScoreThreshold) {
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w - pad_x) / scale);
                    int top = int((y - 0.5 * h - pad_y) / scale);
                    int width = int(w / scale);
                    int height = int(h / scale);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            } else {
                float confidence = data[4];
                if (confidence >= modelConfidenceThreshold) {
                    float *classes_scores = data+5;
                    cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                    cv::Point class_id;
                    double max_class_score;
                    minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                    if (max_class_score > modelScoreThreshold) {
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id.x);
                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];
                        int left = int((x - 0.5 * w - pad_x) / scale);
                        int top = int((y - 0.5 * h - pad_y) / scale);
                        int width = int(w / scale);
                        int height = int(h / scale);
                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }
            data += dimensions;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            modelScoreThreshold,
            modelNMSThreshold,
            nms_result
        );
        std::vector<Detection> detections{};
        for (unsigned long i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = ARC37_BBOX_COLOR;
            result.className = classes[result.class_id];
            result.box = boxes[idx];
            detections.push_back(result);
        }
        return detections;
    }
    /**
     * @brief Main Function
     */
    int main() {
        // logger(ARC37_LABEL INFO "Using OpenCV Version : " CYAN, CV_VERSION, RESET);
        //-- Create Ascii Image Object
        AsciiImage asciiImage;
        //-- Create Inference Object
        Inference inf(
            ARC37_MODEL_PATH,
            cv::Size(640, 640),
            "classes.txt",
            ARC37_RUN_ON_GPU
        );
        //-- Create Video Capture Object
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            // logger(ARC37_LABEL FAILURE "Unable to Open Camera");
            return -1;
        } else {
            // logger(ARC37_LABEL SUCCESS "Camera Opened Successfully");
        }
        //-- Configure Camera (Not Work on All Cameras)
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        //-- Set IP Address and Port
        const char  *   ipAddress = ARC37_DESTINATION_IP;
        const int       port = ARC37_DESTINATION_PORT;
        //-- Create Socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            // logger(ARC37_LABEL FAILURE "Unable to Create Socket");
            return -1;
        } else {
            // logger(ARC37_LABEL SUCCESS "Socket Created Successfully");
        }
        //-- Handle Network
        if (ARC37_SHARE_DATA == 1) {
            //-- Create Server Address
            struct sockaddr_in serverAddr;
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(port);
            serverAddr.sin_addr.s_addr = inet_addr(ipAddress);
            //-- Connect to Server
            if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
                // logger(ARC37_LABEL FAILURE "Unable to Connect to Server on Port ", CYAN, port, RESET);
                close(sock);
                return -1;
            }
        }
        //-- Run Communication Loop in a Separate Thread
        std::thread communicationThread(&Inference::communicationLoop, &inf);
        //-- Definitions
        int numberOfPeople = 0;
        int numberOfPhones = 0;
        //-- Inference Loop
        while (true) {
            //-- Clear Objects List
            inf.objectsList.clear();
            //-- Handle Variables
            numberOfPeople = 0;
            //-- Read Frame
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                // logger(ARC37_LABEL FAILURE "Unable to Capture Frame");
                break;
            }
            //-- Detect Objects
            std::vector<Detection> output = inf.runInference(frame);
            int detections = output.size();
            //-- Process each Detection
            for (int i = 0; i < detections; ++i) {
                Detection detection = output[i];
                cv::Scalar color = cv::Scalar(100, 100, 100);
                bool drawCenter = true;
                bool drawBox = false;
                //-- Check Treshold
                if (detection.confidence < ARC37_CONFIDENCE_THRESHOLD) {
                    continue;
                }
                //-- Check Class ID
                if (detection.className == "person") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Person " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(0, 255, 255);
                    drawBox = true;
                    numberOfPeople++;
                } else if (detection.className == "car") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Car " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(0, 0, 255);
                } else if (detection.className == "cat") {
                    color = cv::Scalar(0, 255, 0);
                } else if (detection.className == "dog") {
                    color = cv::Scalar(0, 255, 0);
                } else if (detection.className == "backpack") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Backpack " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(0, 0, 255);
                } else if (detection.className == "bottle") {
                    color = cv::Scalar(0, 0, 255);
                    drawBox = true;
                } else if (detection.className == "chair") {
                    color = cv::Scalar(0, 0, 255);
                } else if (detection.className == "laptop") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Laptop " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(255, 255, 0);
                } else if (detection.className == "mouse") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Mouse " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(255, 255, 0);
                } else if (detection.className == "remote") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Remote " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(255, 255, 0);
                } else if (detection.className == "keyboard") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Keyboard " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(255, 255, 0);
                } else if (detection.className == "cell phone") {
                    // logger(ARC37_LABEL INFO "Detected" LIGHT_CYAN " Cell Phone " RESET "with Confidence ", CYAN, detection.confidence, RESET);
                    color = cv::Scalar(255, 255, 0);
                    drawBox = true;
                    numberOfPhones++;
                } else if (detection.className == "clock") {
                    color = cv::Scalar(0, 0, 255);
                } else {
                    // logger(ARC37_LABEL INFO "Detected", CYAN, detection.className, RESET);
                }
                //-- Draw Detection Area
                cv::Rect box = detection.box;
                //-- Draw Rectangle
                if (drawBox) {
                    cv::rectangle(frame, box, color, 2);
                }
                //-- Draw Center Point
                if (drawCenter) {
                    cv::circle(
                        frame,
                        cv::Point(box.x + box.width / 2, box.y + box.height / 2),
                        5,
                        color,
                        -1
                    );
                }
                //-- Populate Objects List
                inf.objectsList.push_back(detection.className);
                //-- Check if Focus has Changed
                //-- Handle Focus
                if (detection.className == inf.focusOn) {
                    //-- Handle Region
                    int centerX = box.x + box.width / 2;
                    float compressedX = 0.5f * (1.0f + tanh(3.0f * (
                        (static_cast<float>(centerX) / frame.cols) - 0.5f
                    )));
                    int region = static_cast<int>(compressedX * ARC37_VIEW_FIELD_SECTIONS);
                    region = std::max(0, std::min(region, ARC37_VIEW_FIELD_SECTIONS - 1));
                    region = region - 1;
                    //-- Check if Region has Changed
                    if (region != inf.lastRegion) {
                        inf.stableRegion = region;
                        inf.stableFrameCount = 0;
                        inf.lastRegion = region;
                    }
                    //-- Only Send if the Region has Stayed the Same for 'debounceThreshold' Frames
                    if (inf.stableRegion == region) {
                        inf.stableFrameCount++;
                    } else {
                        inf.stableFrameCount = 0;
                    }
                    //-- Send Data if Stable
                    if (inf.stableFrameCount >= inf.debounceThreshold) {
                        std::string regionStr = std::to_string(region);
                        if (ARC37_SHARE_DATA == 1) {
                            // // logger(ARC37_LABEL INFO "Sending Region ", CYAN, regionStr, RESET);
                            send(sock, regionStr.c_str(), regionStr.length(), 0);
                        } else {
                            // // logger(ARC37_LABEL INFO "Region ", CYAN, regionStr, RESET);
                        }
                        inf.stableFrameCount = 0;
                    }
                    //-- Show Focus Position
                    // if (inf.focusOn == "person") {
                    //     // logger(ARC37_LABEL INFO "Focus on ", CYAN, detection.className, RESET " at X : ", centerX, " , Y : ", box.y, DARK_CYAN " (", numberOfPeople, ")" RESET);
                    // } else if (inf.focusOn == "cell phone") {
                    //     // logger(ARC37_LABEL INFO "Focus on ", CYAN, detection.className, RESET " at X : ", centerX, " , Y : ", box.y, DARK_CYAN " (", numberOfPhones, ")" RESET);
                    // } else {
                    // }
                }
            }

            // idea : check color of detected objects so whenever i ask what is it looking at send color
            //        code to llm and shows nearest color to that RGB value
            asciiImage.convert(frame, NORMAL);
            // cv::imshow("Arcturus37 - Eye", frame);
            if (cv::waitKey(ARC37_UPDATE_DELAY) == 27)
            break;
        }
        communicationThread.join();
        cap.release();
    }
    /**
     * @brief Method to Send and Receive Data from LLM
     * @return std::string 
     */
    std::string Inference::communicationLoop() {
        // const char* myIP = "127.0.0.1";
        // const char* serverIP = "127.0.0.1";
        // int port = 8080;

        // while (true) {
        //     int sock = socket(AF_INET, SOCK_STREAM, 0);
        //     if (sock < 0) { perror("[Sender1] Socket failed"); continue; }

        //     sockaddr_in localAddr{}, serverAddr{};
        //     localAddr.sin_family = AF_INET;
        //     localAddr.sin_addr.s_addr = inet_addr(myIP);
        //     localAddr.sin_port = htons(0);
        //     bind(sock, (sockaddr*)&localAddr, sizeof(localAddr));

        //     serverAddr.sin_family = AF_INET;
        //     serverAddr.sin_port = htons(port);
        //     serverAddr.sin_addr.s_addr = inet_addr(serverIP);

        //     if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        //         perror("[Sender1] Connect failed");
        //         close(sock);
        //         std::this_thread::sleep_for(std::chrono::seconds(1));
        //         continue;
        //     }

        //     const char* message = "Ping from sender1 (LAN)";
        //     send(sock, message, strlen(message), 0);

        //     char buffer[1024] = {0};
        //     int bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0);
        //     if (bytesReceived > 0) {
        //         buffer[bytesReceived] = '\0';
        //         std::cout << "[Sender1] Got reply: " << buffer << std::endl;
        //     }

        //     close(sock);
        //     std::this_thread::sleep_for(std::chrono::seconds(1));
        // }
        const char* serverIP = "127.0.0.1";
        int port = 8787;
        while (true) {
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            // if (sock < 0) { perror("[Sender2] Socket failed"); continue; }
            sockaddr_in serverAddr{};
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(port);
            serverAddr.sin_addr.s_addr = inet_addr(serverIP);
            if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
                // perror("[Sender2] Connect failed");
                close(sock);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            std::string message = "I See ";
            //-- Generate Message
            for (std::string &object : objectsList) {
                message += object + " and ";
                // // logger(ARC37_LABEL INFO "Sending Object ", CYAN, message, RESET);
            }
            //-- Convert to C-String and Send
            send(sock, message.c_str(), strlen(message.c_str()), 0);
            close(sock);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
# endif // ARCTRUS37_EYE_HPP