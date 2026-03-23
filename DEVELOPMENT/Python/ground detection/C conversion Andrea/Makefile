CC      = gcc
CXX     = g++
CFLAGS  = -O2 -Wall -Wno-unused-function -I.
LDFLAGS = -lm
COMMON  = team10_get_obstacle_info_standalone.c team10_get_obstacle_info.h

# OpenCV flags — auto-detected via pkg-config on Linux/macOS.
# On Windows with MinGW64, set these manually, e.g.:
#   OPENCV_DIR = C:/opencv/build
#   OPENCV_CFLAGS = -I$(OPENCV_DIR)/include
#   OPENCV_LIBS   = -L$(OPENCV_DIR)/x64/mingw/lib -lopencv_world490
OPENCV_CFLAGS ?= $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   ?= $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv 2>/dev/null)

.PHONY: all test slideshow clean

all: test_obstacle_detection

test: test_obstacle_detection
	@echo "Usage: ./test_obstacle_detection <yuv422> <w> <h> <outdir> [frames]"

test_obstacle_detection: test_obstacle_detection.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ test_obstacle_detection.c team10_get_obstacle_info_standalone.c $(LDFLAGS)

slideshow: visualize_slideshow.cpp $(COMMON)
	$(CXX) -O2 -Wall -I. $(OPENCV_CFLAGS) \
	    -o visualize_slideshow \
	    visualize_slideshow.cpp team10_get_obstacle_info_standalone.c \
	    $(LDFLAGS) $(OPENCV_LIBS)

clean:
	rm -f test_obstacle_detection visualize_slideshow visualize_slideshow.exe *.o
