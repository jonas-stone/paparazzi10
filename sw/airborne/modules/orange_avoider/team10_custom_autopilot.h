#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

extern float obstacle_width_threshold;
extern float maxDistance;
extern float speed_multiplier;
// functions (TEAM 10 GROUND DETECTION)
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif

