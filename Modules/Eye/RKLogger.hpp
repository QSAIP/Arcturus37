# ifndef RKLOGGER
    /**
     * @file RKLogger.hpp
     * @author Ramtin Kosari (ramtinkosari@gmail.com)
     * @brief Ramtin Kosari Logger Header Library
     * @def LOGGER
     * @date 2024-12-14
     */
    # define RKLOGGER
    /**
     * @def SHOW_LOGS
     * @brief Show Logs
     * @details This Macro Defines if Logs Should be Shown
     */
    # define SHOW_LOGS true
    /**
     * @def ARC37_LABEL
     * @brief Arc37 Label
     * @details This Macro Defines Arc37 Log Label
     */
    # define ARC37_LABEL "\033[38;2;0;173;173m[ARC37]\033[0m "
    /**
     * @def WARNING
     * @brief Warning Message
     * @details This Macro Defines Warning Message Log Label
     */
    # define WARNING "\033[38;2;255;255;0m[WARNING]\033[0m "
    /**
     * @def SUCCESS
     * @brief Success Message
     * @details This Macro Defines Success Message Log Label
     */
    # define SUCCESS "\033[38;2;0;255;0m[SUCCESS]\033[0m "
    /**
     * @def ERROR
     * @brief Error Message
     * @details This Macro Defines Error Message Log Label
     */
    # define FAILURE "\033[38;2;255;0;0m[FAILURE]\033[0m "
    /**
     * @def INFO
     * @brief Information Message
     * @details This Macro Defines Information Message Log Label
     */
    # define INFO "\033[38;2;0;255;255m[INFO]\033[0m "
    /**
     * @def LOG
     * @brief Log Message
     * @details This Macro Defines Log Message Log Label
     */
    # define LOG "\033[38;2;170;170;170m[LOG]\033[0m "
    /**
     * @def RESET
     * @brief Reset Message
     * @details This Macro Resets Log Message Format
     */
    # define RESET "\033[0m"
    /**
     * @def RED
     * @brief Red Message
     * @details This Macro Defines Red Color for Log Messages
     */
    # define RED "\033[38;2;255;20;40m"
    /**
     * @def GREEN
     * @brief Green Message
     * @details This Macro Defines Green Color for Log Messages
     */
    # define GREEN "\033[38;2;0;255;0m"
    /**
     * @def BLUE
     * @brief Blue Message
     * @details This Macro Defines Blue Color for Log Messages
     */
    # define BLUE "\033[38;2;0;0;255m"
    /**
     * @def CYAN
     * @brief Cyan Message
     * @details This Macro Defines Cyan Color for Log Messages
     */
    # define CYAN "\033[38;2;0;255;255m"
    /**
     * @def DARK_CYAN
     * @brief Dark Cyan Message
     * @details This Macro Defines Dark Cyan Color for Log Messages
     */
    # define DARK_CYAN "\033[38;2;0;128;128m"
    /**
     * @def LIGHT_CYAN
     * @brief Light Cyan Message
     * @details This Macro Defines Light Cyan Color for Log Messages
     */
    # define LIGHT_CYAN "\033[38;2;94;174;225m"
    /**
     * @def GRAY
     * @brief Gray Message
     * @details This Macro Defines Gray Color for Log Messages
     */
    # define GRAY "\033[38;2;128;128;128m"
    /**
     * @def BLUE_GRAY
     * @brief Blue Gray Message
     * @details This Macro Defines Blue Gray Color for Log Messages
     */
    # define BLUE_GRAY "\033[38;2;0;128;178m"
    /**
     * @def YELLOW_GRAY
     * @brief Yellow Gray Message
     * @details This Macro Defines Yellow Gray Color for Log Messages
     */
    # define YELLOW_GRAY "\033[38;2;178;128;0m"
    /**
     * @def YELLOW
     * @brief Yellow Message
     * @details This Macro Defines Yellow Color for Log Messages
     */
    # define YELLOW "\033[38;2;255;255;0m"
    /**
     * @def TAB
     * @brief Tab Message
     * @details This Macro Adds Tab to Log Messages
     */
    # define TAB "   "
    //-- Include IOStream
    # ifndef _GLIBCXX_IOSTREAM
        # include <iostream>
    # endif // _GLIBCXX_IOSTREAM
    //-- Include SStream
    # ifndef _GLIBCXX_SSTREAM
        # include <sstream>
    # endif // _GLIBCXX_SSTREAM
    //-- Include IO Control
    # ifndef _SYS_IOCTL_H
        # include <sys/ioctl.h>
    # endif // _SYS_IOCTL_H
    //-- Include Unistd
    # ifndef _UNISTD_H
        # include <unistd.h>
    # endif // _UNISTD_H
    /**
     * @brief Helper Function to Process Single Argument
     * @tparam T Type of Argument
     * @param oss Output String Stream
     * @param arg Argument
     */
    template<typename T> void inline processArgument(
        std::ostringstream& oss,
        const T& arg
    ) {
        oss << arg;
    }
    /**
     * @brief Base Case for Variadic Function
     * @note This Function Stops Recursion
     */
    void inline logger_helper(std::ostringstream&) {}
    /**
     * @brief Helper Function to Process Multiple Arguments
     * @note This Function Recursively Processes Arguments and Stops at Base Case
     * @tparam T Type of Argument
     * @tparam Arguments Type of Arguments
     * @param oss Output String Stream
     * @param first First Argument
     * @param rest Remaining Arguments
     */
    template<typename T, typename... Args> void inline logger_helper(
        std::ostringstream& oss,
        const T& first,
        const Args&... rest
    ) {
        //-- Process First Argument
        processArgument(oss, first);
        //-- Recursively Process Remaining Arguments
        logger_helper(oss, rest...);
    }
    /**
     * @brief Method to Print Log Messages
     * 
     * @tparam Args Arguments
     * @param _in Input Arguments
     */
    template<typename... Args> void logger(Args... _in) {
        std::ostringstream oss;
        //-- Check if C++ Version is 17 or Higher
        # if __cplusplus > 201703L
            (oss << ... << _in);
        # else
            logger_helper(oss, _in...);
        # endif // __cplusplus
        //-- Check if Logs Should be Shown
        if (SHOW_LOGS) std::cout << oss.str() << std::endl;
    }
    /**
     * @brief Method to Get Terminal Width
     * @return int Terminal Width
     */
    inline int getTerminalWidth() {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_col;
    }
    /**
     * @brief Method to Get Terminal Height
     * @return int Terminal Height
     */
    inline int getTerminalHeight() {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_row;
    }
    /**
     * @brief Method to Get Terminal Width in Pixels
     * @return int Terminal Width in Pixels
     */
    inline int getTerminalWidthPx() {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_xpixel;
    }
    /**
     * @brief Method to Get Terminal Height in Pixels
     * @return int Terminal Height in Pixels
     */
    inline int getTerminalHeightPx() {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_ypixel;
    }
# endif // RKLOGGER
