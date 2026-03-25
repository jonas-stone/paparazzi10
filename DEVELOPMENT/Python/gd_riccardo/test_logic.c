/*
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from enum import Enum
from random import random, randint
*/

// general variable definition
#define MAX_IMAGE_WIDTH = 520;

/*
class NavigationState(Enum):
    GO = 1;
    ROTATE = 2;
    OUT_OF_BOUNDS = 3;
    OBSTACLE_FOUND = 4;


class ObjectiveLocation(Enum):
    LEFT = 1;
    RIGHT = 2;
    CENTERLINE = 3;
*/

enum NavigationState {
  GO,
  ROTATE,
  OUT_OF_BOUNDS,
  OBSTACLE_FOUND
};

enum ObjectiveLocation {
  LEFT,
  RIGHT,
  CENTERLINE
};

// set global variables
uint8_t locked_rotate_cooldown_setting_frames = 10;
uint8_t locked_rotate_cooldown = 0;
uint8_t locked_go_cooldown_setting_frames = 40;
uint8_t locked_go_cooldown = 0;

uint8_t centerline_tolerance = 0.1 * MAX_IMAGE_WIDTH;
uint8_t heading_increment = 1;      // degrees
float   maxDistance = 2.5;    // meters

// set static variables
enum ObjectiveLocation point_location;
enum ObjectiveLocation target_location;

// set nav state variable
enum NavigationState nav_state;

def get_best_column():
    return (randint(0, 519), random())

def rotate(heading) -> None:
    print(f"Rotating by {heading}")

def set_waypoint(distance) -> None:
    print(f"Setting waypoint to {distance}")

def stop():
    print(f"Stopping")

def out_of_bounds_logic():
    print(f"OUT OF BOUNDS! setting state to {NavigationState.ROTATE}")

def autopilot_periodic() {

    global nav_state, locked_rotate_cooldown, locked_go_cooldown;
    global target_location, point_location;

    # 1. Perception Step
    best_column, confidence = get_best_column();
    
    # Map pixel to Location (Using 20px integer tolerance for C-compatibility)
    if best_column < (MAX_IMAGE_WIDTH / 2) - centerline_tolerance:
        point_location = ObjectiveLocation.LEFT;
    elif best_column > (MAX_IMAGE_WIDTH / 2) + centerline_tolerance:
        point_location = ObjectiveLocation.RIGHT;
    else:
        point_location = ObjectiveLocation.CENTERLINE;

    # 2. State Machine
    match nav_state:
        case NavigationState.ROTATE:

            # Decrease cooldown every frame
            if locked_rotate_cooldown != 0:
                locked_rotate_cooldown -= 1;
                # Always rotate toward the 'locked' target during cooldown
                rotate(heading=1 if target_location == ObjectiveLocation.LEFT else -1);
                return;

            # RULE: While rotating, ignore everything UNLESS it is centerline
            if point_location == ObjectiveLocation.CENTERLINE:
                nav_state = NavigationState.GO;
                locked_go_cooldown = locked_go_cooldown_setting_frames;
                return;
            else:
                # Continue rotating toward the target assigned when GO ended
                rotate(heading=1 if target_location == ObjectiveLocation.LEFT else -1);

        case NavigationState.GO:

            if locked_go_cooldown == locked_go_cooldown_setting_frames - 1:
                print("===============================-")
                set_waypoint(distance=waypoint_distance);
            
            if locked_go_cooldown != 0:
                locked_go_cooldown -= 1;
                print("Going...")
                return;

            # RULE: Once GO cooldown runs out, pick a NEW target and swap to ROTATE
            if locked_go_cooldown == 0:
                nav_state = NavigationState.ROTATE;
                locked_rotate_cooldown = locked_rotate_cooldown_setting_frames;
                
                # This is where the drone "looks" for the next point to chase
                target_location = point_location; 
                print(f"GO Finished. New Target Locked: {target_location}");

        // unimportant case
        case NavigationState.OBSTACLE_FOUND:
            stop();
            nav_state = NavigationState.ROTATE;

        // unimportant case
        case NavigationState.OUT_OF_BOUNDS:
            stop();
            nav_state = NavigationState.ROTATE;
}    

if __name__ == "__main__":
    
    hertz = 10;
    delta_t = 1 / hertz;

    # simulate initializing THING
    target_location = ObjectiveLocation.RIGHT

    while True:
        time.sleep(delta_t);
        main()