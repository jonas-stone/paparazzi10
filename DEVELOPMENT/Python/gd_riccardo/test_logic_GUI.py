import pygame
import random
import math

# --- LOGIC CONSTANTS (KEEPING YOUR C-STYLE NAMES) ---
MAX_IMAGE_WIDTH = 520;
SCREEN_WIDTH = 800;
SCREEN_HEIGHT = 600;

class NavigationState:
    GO = 1; ROTATE = 2; OUT_OF_BOUNDS = 3; OBSTACLE_FOUND = 4;

class ObjectiveLocation:
    LEFT = 1; RIGHT = 2; CENTERLINE = 3;

# --- GLOBALS (STATE MACHINE) ---
nav_state = NavigationState.ROTATE;
locked_rotate_cooldown = 0;
locked_go_cooldown = 0;
locked_rotate_cooldown_setting_frames = 10;
locked_go_cooldown_setting_frames = 40;

centerline_tolerance = 0.1 * MAX_IMAGE_WIDTH;
target_location = ObjectiveLocation.RIGHT;
point_location = ObjectiveLocation.CENTERLINE;

# --- SIMULATION GLOBALS ---
drone_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2];
drone_angle = 0; # Degrees
obstacles = [[random.randint(0, SCREEN_WIDTH), random.randint(-1000, 0)] for _ in range(10)];
scroll_y = 0;

def get_best_column():
    """Simulates camera picking a point. 0 is left, 519 is right."""
    # We simulate this by checking where the closest obstacle is relative to drone
    return (random.randint(0, 519), random.random());

def rotate(direction):
    global drone_angle;
    drone_angle += direction * 3; # 3 degrees per frame

def main_logic():
    global nav_state, locked_rotate_cooldown, locked_go_cooldown;
    global target_location, point_location;

    # 1. Perception Step (Simulated)
    best_column, confidence = get_best_column();
    
    if best_column < (MAX_IMAGE_WIDTH / 2) - centerline_tolerance:
        point_location = ObjectiveLocation.LEFT;
    elif best_column > (MAX_IMAGE_WIDTH / 2) + centerline_tolerance:
        point_location = ObjectiveLocation.RIGHT;
    else:
        point_location = ObjectiveLocation.CENTERLINE;

    # 2. State Machine Logic
    if nav_state == NavigationState.ROTATE:
        if locked_rotate_cooldown != 0:
            locked_rotate_cooldown -= 1;
            rotate(1 if target_location == ObjectiveLocation.LEFT else -1);
            return;

        if point_location == ObjectiveLocation.CENTERLINE:
            nav_state = NavigationState.GO;
            locked_go_cooldown = locked_go_cooldown_setting_frames;
            return;
        else:
            rotate(1 if target_location == ObjectiveLocation.LEFT else -1);

    elif nav_state == NavigationState.GO:
        if locked_go_cooldown != 0:
            locked_go_cooldown -= 1;
            # Move "forward" relative to drone angle
            return;

        if locked_go_cooldown == 0:
            nav_state = NavigationState.ROTATE;
            locked_rotate_cooldown = locked_rotate_cooldown_setting_frames;
            target_location = point_location; 

# --- GUI / RENDERING ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

def draw_drone(surface, color, pos, angle):
    # Draw a triangle to represent the drone
    size = 20
    rad = math.radians(angle)
    p1 = (pos[0] + size * math.sin(rad), pos[1] - size * math.cos(rad))
    p2 = (pos[0] + size * math.sin(rad + 2.5), pos[1] - size * math.cos(rad + 2.5))
    p3 = (pos[0] + size * math.sin(rad - 2.5), pos[1] - size * math.cos(rad - 2.5))
    pygame.draw.polygon(surface, color, [p1, p2, p3])

running = True
while running:
    screen.fill((30, 30, 30)) # Dark Grey Carpet
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Run the "C-style" logic
    main_logic()

    # Simulation Movement
    if nav_state == NavigationState.GO:
        speed = 5
        rad = math.radians(drone_angle)
        # Move obstacles down to simulate drone moving up (Infinite carpet)
        scroll_y += speed
        for obs in obstacles:
            obs[1] += speed
            if obs[1] > SCREEN_HEIGHT: # Respawn at top
                obs[1] = -100; obs[0] = random.randint(0, SCREEN_WIDTH)

    # Drawing
    for obs in obstacles:
        pygame.draw.circle(screen, (200, 50, 50), (obs[0], obs[1]), 15)
    
    color = (0, 255, 0) if nav_state == NavigationState.GO else (0, 200, 255)
    draw_drone(screen, color, drone_pos, drone_angle)

    # UI Overlay
    font = pygame.font.SysFont(None, 24)
    img = font.render(f"State: {nav_state} | Target: {target_location} | RotCD: {locked_rotate_cooldown}", True, (255, 255, 255))
    screen.blit(img, (10, 10))

    pygame.display.flip()
    clock.tick(10) # 10Hz as per your delta_t

pygame.quit()