# Self-Driving Car AI - by daniyal , muaaz , aymal

import math
import random
import sys
import os
import neat
import pygame
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- CONTROL PANEL ---
# TRUE = Load previous intelligence (Continuously evolve)
# FALSE = HARD RESET (Delete all previous memory and start from Gen 0)
LOAD_FROM_SAVE = True

GENERATIONS_TO_RUN = 5
SAVE_FILE = "smart_population.pkl"
LOG_FILE = "fitness_log.csv"
VIDEO_FILE = "intro.mp4" 

WIDTH = 1920
HEIGHT = 1200
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255) 

# AI PARAMETERS
MAX_SPEED = 30
TIME_PENALTY = 0.5 
STAGNATION_THRESHOLD = 50 

# UI Colors
HUD_BG_COLOR = (0, 0, 0, 150)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (0, 255, 255)

current_generation = 0 
video_has_played = True

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert() 
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [830, 920] 
        self.angle = 0
        self.speed = 0
        self.speed_set = False 
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = [] 
        self.alive = True 
        self.distance = 0 
        self.time = 0 
        self.corners = []
        self.stagnation_counter = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen) 

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            try:
                if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                    self.alive = False
                    break
            except:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1
        
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)
        
        if self.speed < 2:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0 
        
        if self.stagnation_counter > STAGNATION_THRESHOLD:
            self.alive = False

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0, 0] 
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)
        return_values[5] = self.speed / MAX_SPEED
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def log_fitness(generation, avg_fitness, best_fitness):
    """Appends generation stats to a CSV file."""
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Generation", "Avg Fitness", "Best Fitness"])
        
        writer.writerow([generation, avg_fitness, best_fitness])

def play_intro_video(video_path, screen):
    if not os.path.exists(video_path):
        print("Intro video not found. Skipping.")
        return

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    clock = pygame.time.Clock()
    
    print("Playing Intro... Press ESC or SPACE to skip.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                    running = False

        ret, frame = cap.read()
        if not ret: break 

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame) 
        frame = np.flipud(frame)
        
        frame_surface = pygame.surfarray.make_surface(frame)
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(video_fps)

    cap.release()
    screen.fill((0,0,0))
    pygame.display.flip()

def draw_hud(screen, font, generation, alive, best_fitness, avg_fitness, timer):
    hud_width, hud_height = 350, 220 
    hud_surface = pygame.Surface((hud_width, hud_height))
    hud_surface.set_alpha(150) 
    hud_surface.fill((0, 0, 0))
    screen.blit(hud_surface, (20, 20))
    
    lines = [
        f"Generation: {generation}",
        f"Cars Alive: {alive}",
        f"Best Fitness: {best_fitness:.1f}",
        f"Avg Fitness: {avg_fitness:.1f}",
        f"Time: {timer:.1f}s / 20.0s",
        "Press ESC to Save & Quit"
    ]
    
    for i, line in enumerate(lines):
        color = TEXT_COLOR
        if i == 5: color = HIGHLIGHT_COLOR 
        text = font.render(line, True, color)
        screen.blit(text, (30, 30 + i * 30))

def draw_summary(screen, font, generation, best_fitness, max_distance, duration):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0,0,0))
    screen.blit(overlay, (0,0))
    
    big_font = pygame.font.SysFont("Arial", 80, bold=True)
    info_font = pygame.font.SysFont("Arial", 50)

    title = big_font.render(f"GENERATION {generation} COMPLETE", True, HIGHLIGHT_COLOR)
    fitness_text = info_font.render(f"Best Fitness: {best_fitness:.1f}", True, TEXT_COLOR)
    distance_text = info_font.render(f"Max Distance: {int(max_distance)}", True, TEXT_COLOR)
    time_text = info_font.render(f"Gen Duration: {duration:.1f}s", True, TEXT_COLOR)
    
    center_x, center_y = WIDTH / 2, HEIGHT / 2
    
    screen.blit(title, title.get_rect(center=(center_x, center_y - 110)))
    screen.blit(fitness_text, fitness_text.get_rect(center=(center_x, center_y - 30)))
    screen.blit(distance_text, distance_text.get_rect(center=(center_x, center_y + 30)))
    screen.blit(time_text, time_text.get_rect(center=(center_x, center_y + 90)))
    
    pygame.display.flip()

def run_simulation(genomes, config):
    nets = []
    cars = []
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    
    global video_has_played
    if not video_has_played:
        play_intro_video(VIDEO_FILE, screen)
        video_has_played = True

    game_map = pygame.image.load('map.png').convert()

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("Consolas", 20, bold=True)
    
    global current_generation
    current_generation += 1

    counter = 0
    max_time = 30 * 20 

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: sys.exit(0)

        for i, car in enumerate(cars):
            if car.is_alive():
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0: car.angle += 10 
                elif choice == 1: car.angle -= 10 
                elif choice == 2: 
                    if(car.speed - 2 >= 12): car.speed -= 2 
                else: car.speed += 2 

        still_alive = 0
        frame_best_fitness = 0
        best_car_index = -1

        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()
                genomes[i][1].fitness -= TIME_PENALTY

            if genomes[i][1].fitness > frame_best_fitness:
                frame_best_fitness = genomes[i][1].fitness
                best_car_index = i

        total_fitness = sum(g.fitness for i, g in genomes)
        avg_fitness = total_fitness / len(genomes) if len(genomes) > 0 else 0

        if still_alive == 0 or counter >= max_time:
            final_best_fitness = max((g.fitness for i, g in genomes), default=0)
            final_max_dist = max((c.distance for c in cars), default=0)
            final_duration = counter / 30.0 

            log_fitness(current_generation, avg_fitness, final_best_fitness)

            draw_summary(screen, hud_font, current_generation, final_best_fitness, final_max_dist, final_duration)
            pygame.time.wait(2000) 
            break

        counter += 1
        
        screen.blit(game_map, (0, 0))
        for i, car in enumerate(cars):
            if car.is_alive():
                if i == best_car_index:
                    pygame.draw.circle(screen, HIGHLIGHT_COLOR, (int(car.center[0]), int(car.center[1])), 40, 3)
                car.draw(screen)
        
        draw_hud(screen, hud_font, current_generation, still_alive, frame_best_fitness, avg_fitness, counter/30.0)
        pygame.display.flip()
        clock.tick(30) 

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # --- [MODIFIED] STARTUP LOGIC ---
    population = None

    if LOAD_FROM_SAVE:
        # User wants to continue. Try loading.
        try:
            with open(SAVE_FILE, "rb") as f:
                print(f"Loading population from {SAVE_FILE}...")
                population = pickle.load(f)
                population.reporters.reporters = [] 
        except FileNotFoundError:
            print(f"Save file {SAVE_FILE} not found. Starting Fresh.")
    else:
        # User wants a RESET. Delete old files.
        print("HARD RESET: Deleting previous save and history...")
        if os.path.exists(SAVE_FILE):
            os.remove(SAVE_FILE)
            print(f"Deleted {SAVE_FILE}")
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            print(f"Deleted {LOG_FILE}")
    
    # If no save loaded (or we just deleted it), start fresh
    if population is None:
        print("Starting New Population (Fresh Start)...")
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    if hasattr(population, 'generation'):
        current_generation = population.generation
    
    print(f"Running for {GENERATIONS_TO_RUN} generations...")
    
    population.run(run_simulation, GENERATIONS_TO_RUN)

    # Save population
    print(f"Saving population to {SAVE_FILE}...")
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(population, f)
    print("Done!")

    # --- PLOT GRAPH ---
    print("Generating History Graph...")

    gens, avgs, bests = [], [], []
    
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                gens.append(int(row[0]))
                avgs.append(float(row[1]))
                bests.append(float(row[2]))

        plt.figure(figsize=(10, 6))
        plt.plot(gens, avgs, label="Average Fitness", color="blue", marker="o", linestyle="-")
        plt.plot(gens, bests, label="Best Fitness", color="orange", marker="x", linewidth=2, linestyle="-")
        
        plt.title("Evolutionary History (All Sessions)", fontsize=16)
        plt.xlabel("Generations", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()
    else:
        print("No history log found to plot.")