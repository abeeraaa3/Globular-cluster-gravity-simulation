import taichi as ti
import math

# Initialize Taichi with GPU, fall back to CPU
ti.init(arch=ti.cpu)
print("✓ Running on CPU")

# ============================================================================
# CONSTANTS
# ============================================================================

NUM_PARTICLES = 10000
GRAVITY = 1.0
BREATHING_FREQUENCY = 0.5
BREATHING_AMPLITUDE = 0.3
VELOCITY_DAMPING = 0.98
SOFTENING_LENGTH = 0.15
DOMAIN_SIZE = 10.0
DT = 0.016

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
BACKGROUND_COLOR = (0.0, 0.0, 0.01)

# ============================================================================
# Taichi fields for particle state
# ============================================================================

pos = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_PARTICLES,))
vel = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_PARTICLES,))
acc = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_PARTICLES,))
color = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_PARTICLES,))

# Pixel buffer for rendering
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))

# Global state
current_time = ti.field(dtype=ti.f32, shape=())
sim_paused = ti.field(dtype=ti.i32, shape=())
time_scale = ti.field(dtype=ti.f32, shape=())

# ============================================================================
# Initialization
# ============================================================================

@ti.kernel
def initialize_cluster():
    for i in range(NUM_PARTICLES):
        rand_r = ti.random(ti.f32)
        rand_theta = ti.random(ti.f32)
        rand_phi = ti.random(ti.f32)
        
        r = 3.0 * (rand_r ** (1.0/3.0))
        theta = 3.14159265 * rand_theta
        phi = 2.0 * 3.14159265 * rand_phi
        
        sin_theta = ti.sin(theta)
        cos_theta = ti.cos(theta)
        sin_phi = ti.sin(phi)
        cos_phi = ti.cos(phi)
        
        pos[i] = ti.Vector([
            r * sin_theta * cos_phi,
            r * sin_theta * sin_phi,
            r * cos_theta
        ])
        
        vel[i] = ti.Vector([
            (ti.random(ti.f32) - 0.5) * 0.5,
            (ti.random(ti.f32) - 0.5) * 0.5,
            (ti.random(ti.f32) - 0.5) * 0.5
        ])
        
        acc[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def reset_accelerations():
    for i in range(NUM_PARTICLES):
        acc[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def compute_forces():
    for i in range(NUM_PARTICLES):
        a_i = ti.Vector([0.0, 0.0, 0.0])
        
        for j in range(NUM_PARTICLES):
            if i != j:
                delta = pos[j] - pos[i]
                dist_sq = delta.dot(delta)
                dist_cubed = (dist_sq + SOFTENING_LENGTH ** 2) ** 1.5
                a_i += GRAVITY * delta / dist_cubed
        
        breathing = 1.0 + BREATHING_AMPLITUDE * ti.sin(
            2.0 * 3.14159265 * BREATHING_FREQUENCY * current_time[None]
        )
        
        acc[i] = a_i * breathing

@ti.kernel
def integrate():
    dt_actual = DT * time_scale[None]
    
    for i in range(NUM_PARTICLES):
        vel[i] += acc[i] * dt_actual
        vel[i] *= VELOCITY_DAMPING
        pos[i] += vel[i] * dt_actual
        
        for k in ti.static(range(3)):
            if pos[i][k] > DOMAIN_SIZE:
                pos[i][k] -= 2.0 * DOMAIN_SIZE
            elif pos[i][k] < -DOMAIN_SIZE:
                pos[i][k] += 2.0 * DOMAIN_SIZE

@ti.kernel
def update_colors():
    center = ti.Vector([0.0, 0.0, 0.0])
    max_dist = 0.0
    
    for i in range(NUM_PARTICLES):
        center += pos[i]
    center /= ti.cast(NUM_PARTICLES, ti.f32)
    
    for i in range(NUM_PARTICLES):
        dist = (pos[i] - center).norm()
        max_dist = ti.max(max_dist, dist)
    
    max_dist = ti.max(max_dist, 0.1)
    for i in range(NUM_PARTICLES):
        dist = (pos[i] - center).norm()
        dist_ratio = dist / max_dist
        dist_ratio = ti.min(ti.max(dist_ratio, 0.0), 1.0)
        
        if dist_ratio < 0.2:
            t = dist_ratio / 0.2
            color[i] = ti.Vector([1.0, 1.0, 1.0 - t * 0.5])
        elif dist_ratio < 0.5:
            t = (dist_ratio - 0.2) / 0.3
            color[i] = ti.Vector([1.0, 1.0 - t * 0.6, 0.3])
        elif dist_ratio < 0.75:
            t = (dist_ratio - 0.5) / 0.25
            color[i] = ti.Vector([1.0 - t * 0.3, 0.3 - t * 0.3, 0.5 + t * 0.4])
        else:
            t = (dist_ratio - 0.75) / 0.25
            color[i] = ti.Vector([0.7 - t * 0.7, 0.0, 0.9 + t * 0.1])
        
        brightness = 1.0 + (1.0 - dist_ratio) * 0.5
        color[i] *= brightness

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector(BACKGROUND_COLOR)

@ti.kernel
def render_particles():
    """Draw particles to pixel buffer"""
    for idx in range(NUM_PARTICLES):
        # Project 3D position to 2D screen
        p = pos[idx]
        x = p[0]
        y = p[1]
        z = p[2]
        
        # Orthographic projection with centering
        screen_x = ti.cast((x / DOMAIN_SIZE) * 0.4 * ti.cast(WINDOW_WIDTH, ti.f32) + ti.cast(WINDOW_WIDTH, ti.f32) / 2.0, ti.i32)
        screen_y = ti.cast((y / DOMAIN_SIZE) * 0.4 * ti.cast(WINDOW_HEIGHT, ti.f32) + ti.cast(WINDOW_HEIGHT, ti.f32) / 2.0, ti.i32)
        
        # Draw a small circle (3 pixel radius)
        r = 3
        c = color[idx]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                px = screen_x + dx
                py = screen_y + dy
                
                # Check bounds
                if 0 <= px < WINDOW_WIDTH and 0 <= py < WINDOW_HEIGHT:
                    # Only draw if within circle
                    if dx * dx + dy * dy <= r * r:
                        pixels[px, py] = c

# ============================================================================
# Main loop
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GLOBULAR CLUSTER SIMULATION - A Physics Screensaver")
    print("="*70)
    print("\n🎮 CONTROLS:")
    print("  🖱️  LEFT CLICK:     Pull particles toward cursor (gravity well)")
    print("  🖱️  RIGHT CLICK:    Spawn particles at cursor")
    print("  SPACE:            Pause/resume simulation")
    print("  W / S:            Speed up / slow down time")
    print("  R:                Reset to initial state")
    print("  ESC or Q:         Quit\n")
    print(f"📊 Running with {NUM_PARTICLES} particles")
    print("="*70 + "\n")
    
    initialize_cluster()
    current_time[None] = 0.0
    sim_paused[None] = 0
    time_scale[None] = 1.0
    
    window = ti.ui.Window("Globular Cluster", (WINDOW_WIDTH, WINDOW_HEIGHT), vsync=True)
    canvas = window.get_canvas()
    
    frame_count = 0
    
    while window.running:
        # Keyboard input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE or window.event.key == 'q':
                break
            elif window.event.key == ti.ui.SPACE:
                sim_paused[None] = 1 - sim_paused[None]
                status = "PAUSED" if sim_paused[None] else "RUNNING"
                print(f"⏸️  Simulation {status}")
            elif window.event.key == 'w':
                time_scale[None] = ti.min(time_scale[None] + 0.1, 5.0)
                print(f"⏩ Time scale: {time_scale[None]:.1f}x")
            elif window.event.key == 's':
                time_scale[None] = ti.max(time_scale[None] - 0.1, 0.1)
                print(f"⏪ Time scale: {time_scale[None]:.1f}x")
            elif window.event.key == 'r':
                initialize_cluster()
                current_time[None] = 0.0
                print("🔄 Reset to initial state")
        
        # Mouse interaction
        mouse_x, mouse_y = window.get_cursor_pos()
        world_x = (mouse_x - 0.5) * DOMAIN_SIZE * 2.0
        world_y = (mouse_y - 0.5) * DOMAIN_SIZE * 2.0
        world_z = 0.0
        
        if window.is_pressed(ti.ui.LMB):
            for i in range(NUM_PARTICLES):
                delta = ti.Vector([world_x, world_y, world_z]) - pos[i]
                dist = delta.norm() + SOFTENING_LENGTH
                vel[i] += delta / (dist ** 2) * 0.1
        
        if window.is_pressed(ti.ui.RMB):
            spawn_idx = frame_count % NUM_PARTICLES
            pos[spawn_idx] = ti.Vector([world_x, world_y, world_z])
            vel[spawn_idx] = ti.Vector([
                (ti.random(ti.f32) - 0.5) * 0.5,
                (ti.random(ti.f32) - 0.5) * 0.5,
                (ti.random(ti.f32) - 0.5) * 0.5
            ])
        
        # Physics step
        if not sim_paused[None]:
            reset_accelerations()
            compute_forces()
            integrate()
            current_time[None] += DT * time_scale[None]
        
        # Update colors and render
        update_colors()
        clear_pixels()
        render_particles()
        
        # Display pixel buffer
        canvas.set_image(pixels)
        
        window.show()
        frame_count += 1
        
        # Update title every 30 frames
        if frame_count % 30 == 0:
            window.set_title(f"Globular Cluster | Frame: {frame_count}")

if __name__ == "__main__":
    main()