CC      = gcc
CFLAGS  = -O2 -Wall -Wno-unused-function -I.
LDFLAGS = -lm
COMMON  = team10_get_obstacle_info_standalone.c team10_get_obstacle_info.h

.PHONY: all test clean

all: test_obstacle_detection

test: test_obstacle_detection
	@echo "Usage: ./test_obstacle_detection <yuv422> <w> <h> <outdir> [frames]"

test_obstacle_detection: test_obstacle_detection.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ test_obstacle_detection.c team10_get_obstacle_info_standalone.c $(LDFLAGS)

clean:
	rm -f test_obstacle_detection *.o
