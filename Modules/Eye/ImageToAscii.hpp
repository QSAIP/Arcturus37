# ifndef __RKLTOOLS_IMAGE_TO_ASCII
    /**
     * @file imageToAscii.hpp
     * @author Ramtin Kosari (ramtinkosari@gmail.com)
     * @brief Convert Image to Colorful Ascii Characters
     * @version 0.1
     * @date 2023-06-02
     */
    # define __RKLTOOLS_IMAGE_TO_ASCII
    //-- Include Needed Libraries
    # include <chrono>
    # include <string>
    # include <thread>
    # include <vector>
    # include <unistd.h>
    # include <iostream>
    # include <sys/ioctl.h>
    # include <opencv4/opencv2/highgui.hpp>
    # include <opencv4/opencv2/imgproc.hpp>
    //-- Check Operating System for Console Escape Character
    # if defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__) || defined(__MACH__) || defined(macintosh) || defined(Macintosh)
        # define IS_UNIX true
    # else
        # define IS_UNIX false
        std::string escChar = "\1xb[";
    # endif // Check Operating System for Console Escape Character
    /**
     * @brief Image Valid Color Channels
     */
    enum ValidChannels {
        /**
         * @brief RGB Channel
         * @note - About 3 Times Slower than Grayscale
         * @note - More RAM Usage
         */
        NORMAL,
        /**
         * @brief Grayscale Channel
         * @note - About 3 Times Faster than RGB
         * @note - Less RAM Usage
         */
        FAST
    };
    /**
     * @brief Class Definition of Colorful Character
     */
    class Character {
        protected :
            /**
             * @brief Pixel Character
             */
            std::string pixel;
            /**
             * @brief Structure to Store RGB's Values
             */
            struct Color {
                /**
                 * @brief RGB's Red Value
                 */
                int red;
                /**
                 * @brief RGB's Green Value
                 */
                int green;
                /**
                 * @brief RGB's Blue Value
                 */
                int blue;
                /**
                 * @brief Grayscale Value
                 */
                int grayscale;
            };
            /**
             * @brief Structure to Access RGB's Values
             */
            Color color;
        public:
            /**
             * @brief Character Constructor
             */
            Character() noexcept;
            /**
             * @brief Method to Change Color of Character
             * @param Red RGB's Red Value
             * @param Green RGB's Green Value
             * @param Blue RGB's Blue Value
             */
            void setRGBColor(int &Red, int &Green, int &Blue) noexcept;
    };
    /**
     * @brief Class Definitions of Image to Ascii Methodes
     */
    class AsciiImage : public Character {
        private:
            /**
             * @brief Ascii Image Character
             */
            Character aic;
            /**
             * @brief Rows of Terminal
             */
            int rows;
            /**
             * @brief Columns of Terminal
             */
            int cols;
            /**
             * @brief Object to Access Termina;'s Size
             */
            struct winsize terminalWindow;
            /**
             * @brief Cursor Blinking State
             */
            bool blinkingState;
        public:
            /**
             * @brief AsciiImage Constructor
             */
            AsciiImage() noexcept;
            /**
             * @brief Method to Get Size of Terminal
             */
            void getTermianlSize() noexcept;
            /**
             * @brief Method to Convert OpenCV's Matrix to Colorful Background Character
             * @param input Input OpenCV's Matrix
             * @param mode RGB or Grayscale Mode
             * @note Default Mode is Set to RGB, If You Want Faster Experiencing, Pass
             * 'GRAYSCALE' as Mode to The Method and for That, You Must Pass Grayscale
             * Matrix not RGB, Otherwise Output Display is 3 Times Bigger
             */
            void convert(cv::Mat &input, const int &mode);
            /**
             * @brief Method to Change Cursor Blinking State on Terminal Window
             */
            void changeBlinkingState();
            // to Do : Add Method to Use Vectors for Not Using OpenCV
    };
# endif // __RKLTOOLS_IMAGE_TO_ASCII