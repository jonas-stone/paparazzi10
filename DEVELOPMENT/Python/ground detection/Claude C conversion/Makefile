# ──────────────────────────────────────────────────────────────────────────────
#  Makefile for team10 obstacle detection
#
#  Targets:
#    make test       – build the standalone test binary (no Paparazzi needed)
#    make validate   – build + run validation against Python reference
#    make clean      – remove build artifacts
# ──────────────────────────────────────────────────────────────────────────────

CC      = gcc
CFLAGS  = -O2 -Wall -Wno-unused-function -I.
LDFLAGS = -lm

# ── standalone test binary ───────────────────────────────────────────────────

TEST_BIN  = test_obstacle_detection
TEST_SRCS = test_obstacle_detection.c team10_get_obstacle_info_standalone.c

.PHONY: test validate clean

test: $(TEST_BIN)
	@echo ""
	@echo "Built $(TEST_BIN) successfully."
	@echo "Usage:  ./$(TEST_BIN)  <raw_yuv422_file>  <width>  <height>  <output_dir>"

$(TEST_BIN): $(TEST_SRCS) team10_get_obstacle_info.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRCS) $(LDFLAGS)

# ── validation run (requires Python + OpenCV) ────────────────────────────────

validate: $(TEST_BIN)
	@echo "Running Python validation..."
	python3 validate_c_vs_python.py --src-dir . --work-dir /tmp/obstacle_validation $(IMAGE)

clean:
	rm -f $(TEST_BIN) *.o
	rm -rf /tmp/obstacle_validation
