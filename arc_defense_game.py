import pygame
import sys
import random
import math
import time
import os
from deep_arc_agent import DeepArcAgent

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
CENTER = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
CIRCLE_RADIUS = 30
ARC_RADIUS = 60
ARC_LENGTH = math.pi / 6  # 60 degrees
ARC_WIDTH = 5
BALL_RADIUS = 8
BALL_SPEED_RANGE = (3, 7)
ARC_ROTATION_SPEED = 0.1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Arc Defense Game")
clock = pygame.time.Clock()

class Ball:
    def __init__(self, pos):
        self.pos = list(pos)
        speed = random.uniform(*BALL_SPEED_RANGE)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = [speed * math.cos(angle), speed * math.sin(angle)]
    
    def move(self):
        # Store previous position for line segment collision check
        prev_pos = self.pos.copy()
        
        # Update position
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        
        # Bounce off walls
        if self.pos[0] <= BALL_RADIUS:
            self.pos[0] = BALL_RADIUS
            self.velocity[0] *= -1
        elif self.pos[0] >= WINDOW_WIDTH - BALL_RADIUS:
            self.pos[0] = WINDOW_WIDTH - BALL_RADIUS
            self.velocity[0] *= -1
            
        if self.pos[1] <= BALL_RADIUS:
            self.pos[1] = BALL_RADIUS
            self.velocity[1] *= -1
        elif self.pos[1] >= WINDOW_HEIGHT - BALL_RADIUS:
            self.pos[1] = WINDOW_HEIGHT - BALL_RADIUS
            self.velocity[1] *= -1
        
        return prev_pos
    
    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.pos[0]), int(self.pos[1])), BALL_RADIUS)
    
    def check_collision_with_center(self):
        distance = math.sqrt((self.pos[0] - CENTER[0])**2 + (self.pos[1] - CENTER[1])**2)
        return distance <= CIRCLE_RADIUS + BALL_RADIUS
    
    def check_collision_with_arc(self, arc_angle):
        # Get previous position from move method
        prev_pos = self.pos.copy()
        prev_pos[0] -= self.velocity[0]
        prev_pos[1] -= self.velocity[1]
        
        # Check multiple points along the ball's path
        steps = 5
        for i in range(steps + 1):
            t = i / steps
            check_pos = [
                prev_pos[0] + (self.pos[0] - prev_pos[0]) * t,
                prev_pos[1] + (self.pos[1] - prev_pos[1]) * t
            ]
            
            # Check if this point collides with arc
            distance = math.sqrt((check_pos[0] - CENTER[0])**2 + (check_pos[1] - CENTER[1])**2)
            if abs(distance - ARC_RADIUS) > BALL_RADIUS + ARC_WIDTH/2:
                continue
            
            # Check if point is within arc angle
            ball_angle = math.atan2(check_pos[1] - CENTER[1], check_pos[0] - CENTER[0])
            if ball_angle < 0:
                ball_angle += 2 * math.pi
                
            arc_start = arc_angle - ARC_LENGTH/2
            arc_end = arc_angle + ARC_LENGTH/2
            
            # Normalize angles
            while arc_start < 0:
                arc_start += 2 * math.pi
            while arc_end < 0:
                arc_end += 2 * math.pi
                
            # Handle case where arc crosses 0/2π boundary
            if arc_start > arc_end:
                if ball_angle >= arc_start or ball_angle <= arc_end:
                    return True
            else:
                if arc_start <= ball_angle <= arc_end:
                    return True
                    
        return False

# Game Stats
class GameStats:
    def __init__(self):
        self.reset()
        self.high_score = 0
        self.games_played = 0
        self.total_balls_destroyed = 0
    
    def reset(self):
        self.score = 0
        self.balls_destroyed = 0
        self.survival_time = 0
        self.start_time = time.time()
    
    def update(self):
        self.survival_time = time.time() - self.start_time
        if self.score > self.high_score:
            self.high_score = self.score

def create_random_ball():
    # Generate ball at random edge position
    side = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
    if side == 0:  # top
        x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
        y = BALL_RADIUS
    elif side == 1:  # right
        x = WINDOW_WIDTH - BALL_RADIUS
        y = random.randint(BALL_RADIUS, WINDOW_HEIGHT - BALL_RADIUS)
    elif side == 2:  # bottom
        x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
        y = WINDOW_HEIGHT - BALL_RADIUS
    else:  # left
        x = BALL_RADIUS
        y = random.randint(BALL_RADIUS, WINDOW_HEIGHT - BALL_RADIUS)
    
    return Ball([x, y])

def draw_game(screen, arc_angle, balls, game_over, stats, mode="ai", training=True):
    screen.fill(BLACK)
    
    # Draw center circle
    pygame.draw.circle(screen, WHITE, CENTER, CIRCLE_RADIUS)
    
    # Draw arc
    arc_start = arc_angle - ARC_LENGTH/2
    arc_end = arc_angle + ARC_LENGTH/2
    pygame.draw.arc(screen, BLUE, (CENTER[0] - ARC_RADIUS, CENTER[1] - ARC_RADIUS,
                                 ARC_RADIUS * 2, ARC_RADIUS * 2),
                   -arc_end, -arc_start, ARC_WIDTH)
    
    # Draw balls
    for ball in balls:
        ball.draw()
    
    # Draw stats
    font = pygame.font.Font(None, 36)
    y_offset = 10
    texts = [
        f'Score: {stats.score}',
        f'High Score: {stats.high_score}',
        f'Games Played: {stats.games_played}',
        f'Total Balls Destroyed: {stats.total_balls_destroyed}',
        f'Survival Time: {stats.survival_time:.1f}s',
        f'Mode: {mode}',
        f'Training: {training}'
    ]
    
    for text in texts:
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 40
    
    if game_over:
        font = pygame.font.Font(None, 74)
        text = font.render('Game Over!', True, RED)
        text_rect = text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2))
        screen.blit(text, text_rect)
    
    pygame.display.flip()

def reset_game():
    return 0, [], False  # arc_angle, balls, game_over

def main():
    arc_angle, balls, game_over = reset_game()
    stats = GameStats()
    mode = "ai"  # Start in AI mode
    training = True
    
    # Initialize Deep Learning agent
    agent = DeepArcAgent(
        state_size=6,
        hidden_size=64,
        action_size=3,
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.01,
        epsilon_decay=0.9995  # Slower decay for more exploration
    )
    
    # Load pretrained model if exists
    model_path = "arc_defender_model.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Loaded pretrained model")
    
    last_state = None
    last_action = None
    episode_rewards = 0
    
    # Ball generation timing
    last_ball_time = time.time()
    ball_generation_interval = 2.0  # seconds
    
    # Training stats
    episode_count = 0
    total_rewards = []
    
    while True:
        current_time = time.time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save model before quitting
                if training:
                    agent.save(model_path)
                    print("Model saved")
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    mode = "manual" if mode == "ai" else "ai"
                elif event.key == pygame.K_t:
                    training = not training
                elif event.key == pygame.K_s and training:
                    # Save model on demand
                    agent.save(model_path)
                    print("Model saved")
        
        if game_over:
            # Update stats
            stats.games_played += 1
            stats.update()
            
            if training and mode == "ai":
                episode_count += 1
                total_rewards.append(episode_rewards)
                avg_reward = sum(total_rewards[-100:]) / min(len(total_rewards), 100)
                print(f"Episode {episode_count}, Score: {stats.score}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Reset game
            arc_angle, balls, game_over = reset_game()
            stats.reset()
            last_state = None
            last_action = None
            episode_rewards = 0
            continue
        
        # Generate new balls periodically
        if current_time - last_ball_time >= ball_generation_interval:
            balls.append(create_random_ball())
            last_ball_time = current_time
        
        if mode == "manual":
            # Handle manual arc movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                arc_angle -= ARC_ROTATION_SPEED
            if keys[pygame.K_DOWN]:
                arc_angle += ARC_ROTATION_SPEED
        else:
            # AI control
            current_state = agent.get_state(arc_angle, balls, CENTER)
            action = agent.get_action(current_state)
            
            # Apply action
            if action == 1:  # Rotate clockwise
                arc_angle += ARC_ROTATION_SPEED
            elif action == 2:  # Rotate counter-clockwise
                arc_angle -= ARC_ROTATION_SPEED
            
            # Calculate reward
            reward = 0.01  # Small positive reward for surviving
            episode_rewards += reward
            
            # Update neural network if we have a previous state-action pair
            if training and last_state is not None and last_action is not None:
                agent.update(last_state, last_action, reward, current_state, False)
            
            last_state = current_state
            last_action = action
        
        # Update ball positions and check collisions
        balls_destroyed = 0
        for ball in balls[:]:
            prev_pos = ball.move()
            if ball.check_collision_with_center():
                game_over = True
                if training and mode == "ai" and last_state is not None and last_action is not None:
                    # Large negative reward for losing
                    reward = -100
                    episode_rewards += reward
                    next_state = agent.get_state(arc_angle, [], CENTER)
                    agent.update(last_state, last_action, reward, next_state, True)
                break
            if ball.check_collision_with_arc(arc_angle):
                balls.remove(ball)
                balls_destroyed += 1
                stats.score += 10
                stats.balls_destroyed += 1
                stats.total_balls_destroyed += 1
                if training and mode == "ai" and last_state is not None and last_action is not None:
                    # Positive reward for destroying balls
                    reward = 10
                    episode_rewards += reward
                    next_state = agent.get_state(arc_angle, balls, CENTER)
                    agent.update(last_state, last_action, reward, next_state, False)
        
        # Update stats
        stats.update()
        
        # Draw everything
        draw_game(screen, arc_angle, balls, game_over, stats, mode, training)
        clock.tick(60)

if __name__ == "__main__":
    main()
