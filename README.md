<img width="1920" height="1095" alt="image" src="https://github.com/user-attachments/assets/1d3142e3-0790-4e3e-80ab-4c537ebc237c" /># Self-Driving Car AI using Python NEAT Genetic Algorithms Featuring persistent neural network memory.
An advanced autonomous driving simulation built with Pygame and NEAT. Features persistent neural network memory, real-time CSV analytics, and evolutionary performance visualization using Matplotlib.
Key Features
🧠 NeuroEvolution of Augmenting Topologies (NEAT): Implements a genetic algorithm to evolve neural networks that learn driving mechanics from scratch without human intervention.

💾 Persistent AI Memory: Unlike standard simulations, this project uses pickle to serialize trained populations. This allows the AI to save its "brain," enabling continuous training across multiple sessions without resetting progress.

📊 Evolutionary Analytics: Automatically logs generation data (Average Fitness vs. Best Fitness) to a CSV file (fitness_log.csv) in real-time.

📈 Performance Visualization: Integrated Matplotlib to parse the CSV log and generate a line graph at the end of every session, visualizing the learning curve and fitness spikes over generations.

⚡ Optimized Fitness Function: Engineered a custom reward system that prioritizes Speed + Distance while applying time penalties to discourage stagnation and "safe" camping behavior.

🎥 Multimedia HUD: Features a dynamic Heads-Up Display showing real-time telemetry (sensor data, speed, generation count) and a cinematic OpenCV intro sequence.


## 🛠️ Built With

* [Python 3.x](https://www.python.org/)
* [Pygame](https://www.pygame.org/) - Simulation engine
* [NEAT-Python](https://neat-python.readthedocs.io/) - Genetic algorithm library
* [Matplotlib](https://matplotlib.org/) - Data visualization
* [OpenCV](https://opencv.org/) - Video rendering
* [NumPy](https://numpy.org/) - Math operations

## 📦 Installation
1. **Install dependencies:**
    You can install the required libraries using pip:
    ```bash
    pip install pygame neat-python matplotlib numpy opencv-python
    ```

2.  **Verify Assets:**
    Ensure the following files are in the root directory:
    * `car.png` (Sprite)
    * `map.png` (Track)
    * `config.txt` (NEAT configuration)
    * `intro.mp4` (Optional intro video)

## 🎮 How to Run

Run the main script:

```bash
python main.py
⚙️ Control Panel (Configuration)You can modify the simulation behavior by changing the constants at the top of main.py:VariableTypeDescriptionLOAD_FROM_SAVETrue/FalseTrue: Loads previous brain from smart_population.pkl to continue training.False: HARD RESET. Deletes memory and starts from Generation 0.GENERATIONS_TO_RUNIntHow many generations the simulation runs before auto-saving and plotting graphs.MAX_SPEEDIntLimits the maximum speed of the agents.📊 Analytics & GraphsWhen the simulation ends (either by reaching the generation limit or pressing ESC), the program will:Save the current population to smart_population.pkl.Read fitness_log.csv.Display a Matplotlib graph showing the history of Average vs. Best Fitness over all recorded sessions.

👥 Credits & AcknowledgmentsCore Logic: Based on the tutorial by NeuralNine.Advanced Features & Engineering: Developed by Daniyal, Muaaz, and Aymal.Implemented persistent memory, CSV logging, Matplotlib integration, and fitness function optimization.
