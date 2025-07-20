import sys
import os
import pandas as pd
import time
import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QLabel, QPushButton,
                             QFileDialog, QTableWidget, QTableWidgetItem,
                             QProgressBar, QTextEdit, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QMessageBox,
                             QHeaderView, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from GA import AdvancedGeneticScheduler
import warnings
warnings.filterwarnings('ignore')

class FitnessPlotWidget(QWidget):
    """Widget for displaying real-time fitness plot"""

    def __init__(self):
        super().__init__()
        self.init_plot()

    def init_plot(self):
        """Initialize the matplotlib plot"""
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize plot data
        self.generations = []
        self.fitness_values = []
        self.initial_fitness = None
        self.best_fitness = None

        # Create the plot
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], '#2E86AB', linewidth=2.5, label='Improvement Progress',
                                 marker='o', markersize=4, markerfacecolor='#A23B72',
                                 markeredgecolor='white', markeredgewidth=1)
        self.ax.set_xlabel('Generation', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Improvement from Initial Fitness (%)', fontsize=10, fontweight='bold')
        self.ax.set_title('Genetic Algorithm Fitness Evolution', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        self.ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Set initial limits
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-0.05, 1.05)  # Start from bottom (worst) to top (best)

        # Style improvements
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#333333')
        self.ax.spines['bottom'].set_color('#333333')
        self.ax.set_facecolor('#FAFAFA')
        self.figure.patch.set_facecolor('white')
        self.figure.tight_layout()

        self.canvas.draw()

    def add_fitness_point(self, generation, fitness):
        """Add a new fitness point to the plot"""
        # Store initial fitness for normalization
        if self.initial_fitness is None:
            self.initial_fitness = fitness
            self.best_fitness = fitness

        # Update best fitness ONLY for tracking, not for normalization
        if fitness < self.best_fitness:
            self.best_fitness = fitness

        # Add data point
        self.generations.append(generation)
        self.fitness_values.append(fitness)

        # FIXED NORMALIZATION: Each point shows its improvement from INITIAL fitness
        # We normalize each fitness value independently against the INITIAL fitness
        normalized_values = []
        for f in self.fitness_values:
            if self.initial_fitness > 0:
                # Calculate improvement percentage from initial fitness
                improvement = (self.initial_fitness - f) / self.initial_fitness
                # Convert to 0-1 scale, clamped
                normalized_value = max(0.0, min(1.0, improvement))
            else:
                normalized_value = 0.0
            normalized_values.append(normalized_value)

        # Update plot
        self.line.set_data(self.generations, normalized_values)

        # Adjust x-axis dynamically
        if len(self.generations) > 1:
            max_gen = max(self.generations)
            if max_gen > self.ax.get_xlim()[1] * 0.8:
                # Set x-axis to show a bit more than current max
                new_xlim = max(max_gen * 1.3, 20)  # At least 20 for visibility
                self.ax.set_xlim(0, new_xlim)

        # Update title with current stats
        current_normalized = normalized_values[-1] if normalized_values else 0
        improvement_percent = current_normalized * 100
        self.ax.set_title(f'🧬 GA Fitness Evolution - Gen: {generation}, Improvement: {improvement_percent:.1f}%')

        self.canvas.draw()

    def clear_plot(self):
        """Clear the plot for new optimization"""
        self.generations = []
        self.fitness_values = []
        self.initial_fitness = None
        self.best_fitness = None

        self.line.set_data([], [])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-0.05, 1.05)  # Start from bottom, go to top
        self.ax.set_title('🧬 Genetic Algorithm Fitness Evolution - Ready to Start')

        self.canvas.draw()

class CustomGeneticScheduler(AdvancedGeneticScheduler):
    """Custom GA scheduler with configurable soft improvement threshold"""

    def __init__(self, excel_file: str, soft_improvement_threshold: float = 0.05, progress_callback=None):
        super().__init__(excel_file)
        self.soft_improvement_threshold = soft_improvement_threshold
        self.progress_callback = progress_callback

    def evolve_advanced(self):
        """Override to use custom soft improvement threshold"""
        # Store original method and modify behavior
        course_ids_list = self.course_ids
        population = self._generate_initial_population_constraint_aware(course_ids_list)

        best_schedule = None
        best_violations = {}
        initial_soft_fitness_at_hard_resolve = None
        hard_constraints_resolved_at_generation = -1

        print("\nStarting advanced evolution...")
        for generation in range(self.num_generations):
            fitnesses = []
            for individual in population:
                fitness, _ = self._calculate_fitness(individual)
                fitnesses.append(fitness)

            # Find best individual
            best_idx = fitnesses.index(min(fitnesses))
            current_best_fitness = fitnesses[best_idx]
            current_best_schedule = population[best_idx]
            _, current_violations = self._calculate_fitness(current_best_schedule)

            # Check if this is a new best
            if best_schedule is None or current_best_fitness < min([self._calculate_fitness(best_schedule)[0]]):
                best_schedule = current_best_schedule
                best_violations = current_violations
                print(f"Generation {generation + 1}: NEW BEST Fitness = {current_best_fitness:.2f}, Violations: {current_violations}")

                # Emit progress for plotting
                if self.progress_callback:
                    self.progress_callback(generation + 1, current_best_fitness)

                # Check hard constraints resolution
                hard_violations = current_violations['classroom_conflicts'] + current_violations['instructor_conflicts'] + current_violations['capacity_violations'] + current_violations['instructor_availability_violations']

                if hard_violations == 0 and hard_constraints_resolved_at_generation == -1:
                    hard_constraints_resolved_at_generation = generation + 1
                    initial_soft_fitness_at_hard_resolve = current_best_fitness
                    print(f"🎉 All hard constraints resolved at generation {generation + 1}! Initial soft fitness: {initial_soft_fitness_at_hard_resolve:.2f}")

                # Check soft constraints improvement (using custom threshold)
                if hard_constraints_resolved_at_generation != -1:
                    soft_improvement_ratio = (initial_soft_fitness_at_hard_resolve - current_best_fitness) / initial_soft_fitness_at_hard_resolve
                    if soft_improvement_ratio >= self.soft_improvement_threshold:
                        print(f"✅ {self.soft_improvement_threshold*100}% soft constraint improvement achieved at generation {generation + 1}! Final soft fitness: {current_best_fitness:.2f}")
                        break

            # Adapt parameters
            self._adapt_parameters(current_best_fitness)

            # Generate new population
            new_population = []

            # Elitism: carry over best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitnesses)

                offspring1, offspring2 = parent1, parent2
                if random.random() < self.current_crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)

                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)

                if random.random() < 0.2:
                    offspring1 = self._local_search(offspring1)
                if random.random() < 0.2 and len(new_population) + 1 < self.population_size:
                    offspring2 = self._local_search(offspring2)

                offspring1 = self._repair_individual(offspring1)
                offspring2 = self._repair_individual(offspring2)

                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)

            population = new_population

        print("\nEvolution complete.")
        print(f"Final Best Fitness: {self._calculate_fitness(best_schedule)[0]:.2f}")
        print(f"Final Violations: {best_violations}")

        return best_schedule, best_violations

class OptimizationWorker(QThread):
    """Worker thread for running GA optimization"""
    progress_update = pyqtSignal(int, float, dict)  # generation, fitness, violations
    finished = pyqtSignal(list, dict)  # best_schedule, violations
    error_occurred = pyqtSignal(str)
    generation_update = pyqtSignal(int, float, str)  # generation, fitness, status
    fitness_update = pyqtSignal(int, float)  # generation, fitness for plotting

    def __init__(self, excel_file, parameters):
        super().__init__()
        self.excel_file = excel_file
        self.parameters = parameters
        self.scheduler = None
        self.is_running = True

    def run(self):
        try:
            # Initialize scheduler with custom parameters
            self.scheduler = CustomGeneticScheduler(
                self.excel_file,
                self.parameters['soft_improvement'],
                progress_callback=self.emit_fitness_update
            )

            # Apply all custom parameters
            self.scheduler.population_size = self.parameters['population_size']
            self.scheduler.num_generations = self.parameters['num_generations']
            self.scheduler.mutation_rate = self.parameters['mutation_rate']
            self.scheduler.current_mutation_rate = self.parameters['mutation_rate']
            self.scheduler.crossover_rate = self.parameters['crossover_rate']
            self.scheduler.current_crossover_rate = self.parameters['crossover_rate']
            self.scheduler.elitism_count = self.parameters['elitism_count']
            self.scheduler.stagnation_limit = self.parameters['stagnation_limit']

            # Apply soft improvement parameter by modifying the GA code behavior
            # This will be used in the evolve_advanced method

            # Run optimization with progress tracking
            best_schedule, violations = self.scheduler.evolve_advanced()

            self.finished.emit(best_schedule, violations)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def emit_fitness_update(self, generation, fitness):
        """Emit fitness update for plotting"""
        self.fitness_update.emit(generation, fitness)

    def stop(self):
        """Stop the optimization"""
        self.is_running = False
        if self.scheduler:
            # We can't easily stop the GA mid-process, but we can terminate the thread
            self.terminate()

class GASchedulerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.excel_file = None
        self.optimization_worker = None
        self.best_schedule = None
        self.violations = None
        
        self.init_ui()
        self.apply_modern_style()
        
    def init_ui(self):
        self.setWindowTitle("🧬 Genetic Algorithm Class Schedule Optimization")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        self.create_header(main_layout)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_welcome_tab()
        self.create_data_tab()
        self.create_parameters_tab()
        self.create_optimization_tab()
        self.create_results_tab()
        
    def create_header(self, layout):
        """Create application header"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                margin: 5px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🧬 Genetic Algorithm Class Schedule Optimization")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                background: transparent;
                padding: 10px;
            }
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Version info
        version_label = QLabel("Advanced Optimization")
        version_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 12px;
                background: transparent;
                padding: 10px;
            }
        """)
        header_layout.addWidget(version_label)
        
        layout.addWidget(header_frame)

    def create_welcome_tab(self):
        """Create welcome/info tab"""
        welcome_widget = QWidget()
        layout = QVBoxLayout(welcome_widget)

        # Welcome message
        welcome_group = QGroupBox("🎉 Welcome!")
        welcome_layout = QVBoxLayout(welcome_group)

        welcome_text = QLabel("""
        <h2>🧬 Genetic Algorithm Class Schedule Optimization</h2>
        <p><b>This application uses advanced genetic algorithms to optimize university class schedules.</b></p>

        <h3>✨ Features:</h3>
        <ul>
        <li>🧠 <b>Smart Optimization:</b> Constraint-aware initialization and advanced crossover</li>
        <li>📊 <b>Real-time Monitoring:</b> Live tracking of optimization process</li>
        <li>⚙️ <b>Customizable Parameters:</b> Adjust GA settings to your needs</li>
        <li>📁 <b>Excel Integration:</b> Easy data loading and result export</li>
        <li>🎯 <b>Multi-constraint:</b> Automatic management of hard and soft constraints</li>
        </ul>

        <h3>🚀 Getting Started:</h3>
        <ol>
        <li><b>Data Loading:</b> Load your Excel file (Classrooms, Courses, Instructors, Students)</li>
        <li><b>Parameters:</b> Configure GA settings (default values work for most cases)</li>
        <li><b>Optimization:</b> Start the process and track progress</li>
        <li><b>Results:</b> View and save the optimized schedule</li>
        </ol>

        <p><i>💡 Tip: Start by going to the "📊 Data Loading" tab!</i></p>
        """)
        welcome_text.setWordWrap(True)
        welcome_text.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
        """)
        welcome_layout.addWidget(welcome_text)

        layout.addWidget(welcome_group)
        layout.addStretch()

        self.tab_widget.addTab(welcome_widget, "🏠 Home")

    def create_data_tab(self):
        """Create data loading and preview tab"""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)

        # File selection group
        file_group = QGroupBox("📁 Excel File Selection")
        file_layout = QVBoxLayout(file_group)

        # File selection row
        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected yet...")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        file_row.addWidget(self.file_label)

        self.browse_btn = QPushButton("📂 Browse File")
        self.browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(self.browse_btn)

        file_layout.addLayout(file_row)
        layout.addWidget(file_group)

        # Data preview group
        preview_group = QGroupBox("👁️ Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Sheet tabs
        self.sheet_tabs = QTabWidget()
        preview_layout.addWidget(self.sheet_tabs)

        layout.addWidget(preview_group)

        self.tab_widget.addTab(data_widget, "📊 Data Loading")

    def create_parameters_tab(self):
        """Create GA parameters configuration tab"""
        params_widget = QWidget()
        layout = QVBoxLayout(params_widget)

        # Presets group
        presets_group = QGroupBox("🎯 Parameter Presets")
        presets_layout = QHBoxLayout(presets_group)

        # Preset buttons
        self.preset_buttons = []
        presets = [
            ("🚀 Quick Test", {"population_size": 50, "num_generations": 100, "mutation_rate": 0.1, "crossover_rate": 0.8, "elitism_count": 5, "stagnation_limit": 20, "soft_improvement": 0.05}),
            ("⚡ Fast", {"population_size": 150, "num_generations": 1000, "mutation_rate": 0.08, "crossover_rate": 0.85, "elitism_count": 8, "stagnation_limit": 30, "soft_improvement": 0.15}),
            ("🎯 Balanced", {"population_size": 300, "num_generations": 5000, "mutation_rate": 0.05, "crossover_rate": 0.85, "elitism_count": 10, "stagnation_limit": 50, "soft_improvement": 0.3}),
            ("🔬 Thorough", {"population_size": 500, "num_generations": 8000, "mutation_rate": 0.03, "crossover_rate": 0.9, "elitism_count": 15, "stagnation_limit": 75, "soft_improvement": 0.40}),
            ("🏆 Maximum", {"population_size": 800, "num_generations": 15000, "mutation_rate": 0.02, "crossover_rate": 0.95, "elitism_count": 25, "stagnation_limit": 100, "soft_improvement": 0.50})
        ]

        for name, params in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, p=params: self.apply_preset(p))
            btn.setStyleSheet("QPushButton { padding: 8px 12px; margin: 2px; }")
            presets_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        layout.addWidget(presets_group)

        # Parameters group
        params_group = QGroupBox("⚙️ Genetic Algorithm Parameters")
        params_form = QFormLayout(params_group)

        # Population size
        self.population_spin = QSpinBox()
        self.population_spin.setRange(50, 1000)
        self.population_spin.setValue(300)
        self.population_spin.setSuffix(" individuals")
        params_form.addRow("👥 Population Size:", self.population_spin)

        # Number of generations
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(100, 20000)
        self.generations_spin.setValue(5000)
        self.generations_spin.setSuffix(" generations")
        params_form.addRow("🔄 Number of Generations:", self.generations_spin)

        # Mutation rate
        self.mutation_spin = QDoubleSpinBox()
        self.mutation_spin.setRange(0.001, 0.5)
        self.mutation_spin.setValue(0.05)
        self.mutation_spin.setSingleStep(0.001)
        self.mutation_spin.setDecimals(3)
        params_form.addRow("🧬 Mutation Rate:", self.mutation_spin)

        # Crossover rate
        self.crossover_spin = QDoubleSpinBox()
        self.crossover_spin.setRange(0.5, 1.0)
        self.crossover_spin.setValue(0.85)
        self.crossover_spin.setSingleStep(0.01)
        self.crossover_spin.setDecimals(3)
        params_form.addRow("🔀 Crossover Rate:", self.crossover_spin)

        # Elitism count
        self.elitism_spin = QSpinBox()
        self.elitism_spin.setRange(1, 50)
        self.elitism_spin.setValue(10)
        self.elitism_spin.setSuffix(" best individuals")
        params_form.addRow("🏆 Elitism Count:", self.elitism_spin)

        # Stagnation limit
        self.stagnation_spin = QSpinBox()
        self.stagnation_spin.setRange(10, 200)
        self.stagnation_spin.setValue(50)
        self.stagnation_spin.setSuffix(" generations")
        params_form.addRow("⏱️ Stagnation Limit:", self.stagnation_spin)

        # Soft fitness improvement rate
        self.soft_improvement_spin = QDoubleSpinBox()
        self.soft_improvement_spin.setRange(0.05, 0.95)
        self.soft_improvement_spin.setValue(0.2)  # Higher default for longer runs
        self.soft_improvement_spin.setSingleStep(0.05)
        self.soft_improvement_spin.setDecimals(2)
        self.soft_improvement_spin.setToolTip("Target improvement rate for soft constraints (higher = longer optimization)")
        params_form.addRow("🎯 Soft Fitness Improvement:", self.soft_improvement_spin)

        layout.addWidget(params_group)
        layout.addStretch()

        self.tab_widget.addTab(params_widget, "⚙️ Parameters")

    def apply_preset(self, params):
        """Apply preset parameters"""
        self.population_spin.setValue(params["population_size"])
        self.generations_spin.setValue(params["num_generations"])
        self.mutation_spin.setValue(params["mutation_rate"])
        self.crossover_spin.setValue(params["crossover_rate"])
        self.elitism_spin.setValue(params["elitism_count"])
        self.stagnation_spin.setValue(params["stagnation_limit"])
        self.soft_improvement_spin.setValue(params["soft_improvement"])

    def create_optimization_tab(self):
        """Create optimization control and monitoring tab"""
        opt_widget = QWidget()
        layout = QVBoxLayout(opt_widget)

        # Control buttons
        control_group = QGroupBox("🎮 Optimization Control")
        control_layout = QHBoxLayout(control_group)

        self.start_btn = QPushButton("🚀 Start Optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()
        layout.addWidget(control_group)

        # Progress monitoring
        progress_group = QGroupBox("📈 Progress Monitoring")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        # Fitness plot
        self.fitness_plot = FitnessPlotWidget()
        progress_layout.addWidget(self.fitness_plot)

        # Status text (smaller now)
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)

        self.tab_widget.addTab(opt_widget, "🚀 Optimization")

    def create_results_tab(self):
        """Create results display tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)

        # Results control
        control_group = QGroupBox("📋 Results Control")
        control_layout = QHBoxLayout(control_group)

        self.export_btn = QPushButton("💾 Export to Excel")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)

        control_layout.addStretch()
        layout.addWidget(control_group)

        # Results table
        results_group = QGroupBox("📊 Optimization Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group)

        self.tab_widget.addTab(results_widget, "📊 Results")

    def apply_modern_style(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #e1e1e1;
                border: 1px solid #c0c0c0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

    def browse_file(self):
        """Browse and select Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx *.xls)"
        )

        if file_path:
            self.excel_file = file_path
            self.file_label.setText(f"📁 {os.path.basename(file_path)}")
            self.file_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.start_btn.setEnabled(True)

            # Load and preview data
            self.load_data_preview()

    def load_data_preview(self):
        """Load and display data preview"""
        try:
            # Clear existing tabs
            self.sheet_tabs.clear()

            # Load each sheet
            sheets = ['Classrooms', 'Courses', 'Instructors', 'Students']

            for sheet_name in sheets:
                try:
                    df = pd.read_excel(self.excel_file, sheet_name=sheet_name)

                    # Create table widget
                    table = QTableWidget()
                    table.setRowCount(min(len(df), 100))  # Show max 100 rows
                    table.setColumnCount(len(df.columns))
                    table.setHorizontalHeaderLabels(df.columns.tolist())

                    # Fill table with data
                    for i in range(min(len(df), 100)):
                        for j, col in enumerate(df.columns):
                            item = QTableWidgetItem(str(df.iloc[i, j]))
                            table.setItem(i, j, item)

                    # Adjust column widths
                    table.horizontalHeader().setStretchLastSection(True)
                    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

                    # Add tab
                    self.sheet_tabs.addTab(table, f"{sheet_name} ({len(df)} records)")

                except Exception as e:
                    # Create error label if sheet doesn't exist
                    error_label = QLabel(f"❌ Failed to load {sheet_name} sheet: {str(e)}")
                    error_label.setAlignment(Qt.AlignCenter)
                    error_label.setStyleSheet("color: red; font-size: 14px; padding: 20px;")
                    self.sheet_tabs.addTab(error_label, f"{sheet_name} (Error)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file:\n{str(e)}")

    def start_optimization(self):
        """Start GA optimization"""
        if not self.excel_file:
            QMessageBox.warning(self, "Warning", "Please select an Excel file first!")
            return

        # Get all parameters
        parameters = {
            'population_size': self.population_spin.value(),
            'num_generations': self.generations_spin.value(),
            'mutation_rate': self.mutation_spin.value(),
            'crossover_rate': self.crossover_spin.value(),
            'elitism_count': self.elitism_spin.value(),
            'stagnation_limit': self.stagnation_spin.value(),
            'soft_improvement': self.soft_improvement_spin.value()
        }

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(parameters['num_generations'])
        self.status_text.clear()
        self.fitness_plot.clear_plot()  # Clear the plot for new optimization

        self.status_text.append("🚀 Starting optimization...")
        self.status_text.append(f"📊 Pop: {parameters['population_size']}, Gen: {parameters['num_generations']}")
        self.status_text.append(f"🧬 Mut: {parameters['mutation_rate']}, Cross: {parameters['crossover_rate']}")
        self.status_text.append("-" * 30)

        # Switch to optimization tab
        self.tab_widget.setCurrentIndex(3)

        # Start worker thread
        self.optimization_worker = OptimizationWorker(self.excel_file, parameters)
        self.optimization_worker.progress_update.connect(self.update_progress)
        self.optimization_worker.finished.connect(self.optimization_finished)
        self.optimization_worker.error_occurred.connect(self.optimization_error)
        self.optimization_worker.fitness_update.connect(self.fitness_plot.add_fitness_point)  # Connect to plot
        self.optimization_worker.start()

        # Start timer for realistic progress updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.realistic_progress_update)
        self.progress_timer.start(3000)  # Update every 3 seconds for more realistic speed
        self.current_generation = 0
        self.last_update_time = 0
        self.start_time = time.time()

    def realistic_progress_update(self):
        """More realistic progress updates that match actual GA speed"""
        if self.optimization_worker and self.optimization_worker.isRunning():
            max_gen = self.generations_spin.value()

            # Check if we have real fitness data from the plot
            if hasattr(self.fitness_plot, 'generations') and self.fitness_plot.generations:
                # Use real generation data from GA
                actual_generation = max(self.fitness_plot.generations)
                self.current_generation = actual_generation
                self.progress_bar.setValue(actual_generation)

                # Update status based on real progress
                if actual_generation % 10 == 0 or actual_generation <= 10:
                    self.status_text.append(f"📊 Gen {actual_generation}/{max_gen}")

                    # Auto-scroll to bottom
                    scrollbar = self.status_text.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
            else:
                # Fallback to time-based estimation only if no real data
                current_time = time.time()
                elapsed_time = current_time - self.start_time

                # Conservative estimation for early stages
                estimated_generation = min(int(elapsed_time * 0.5), max_gen)

                if estimated_generation > self.current_generation:
                    self.current_generation = estimated_generation
                    self.progress_bar.setValue(self.current_generation)

                    if self.current_generation % 20 == 0 or self.current_generation <= 5:
                        self.status_text.append(f"📊 Gen {self.current_generation}/{max_gen}")

                        # Auto-scroll to bottom
                        scrollbar = self.status_text.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, generation, fitness, violations):
        """Update progress display"""
        self.progress_bar.setValue(generation)
        self.status_text.append(f"📊 Generation {generation}: Fitness = {fitness:.2f}")

        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stop_optimization(self):
        """Stop optimization"""
        if self.optimization_worker:
            self.optimization_worker.stop()
            self.optimization_worker.wait(3000)  # Wait up to 3 seconds
            if self.optimization_worker.isRunning():
                self.optimization_worker.terminate()
                self.optimization_worker.wait()

        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_text.append("⏹️ Optimization stopped.")

    def optimization_finished(self, best_schedule, violations):
        """Handle optimization completion"""
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()

        self.best_schedule = best_schedule
        self.violations = violations

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)

        # Update progress bar to show completion
        if hasattr(self.fitness_plot, 'generations') and self.fitness_plot.generations:
            final_generation = max(self.fitness_plot.generations)
            self.progress_bar.setValue(final_generation)
        else:
            self.progress_bar.setValue(self.progress_bar.maximum())

        self.status_text.append("✅ Optimization completed!")
        self.status_text.append(f"📊 Total constraint violations: {sum(violations.values())}")

        # Update plot title to show completion
        if hasattr(self.fitness_plot, 'ax'):
            current_title = self.fitness_plot.ax.get_title()
            if "Gen:" in current_title:
                self.fitness_plot.ax.set_title(current_title.replace("GA Fitness Evolution", "✅ Optimization Complete"))
                self.fitness_plot.canvas.draw()

        # Display results
        self.display_results()

        # Switch to results tab
        self.tab_widget.setCurrentIndex(4)

    def optimization_error(self, error_message):
        """Handle optimization error"""
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.status_text.append(f"❌ Error occurred: {error_message}")
        QMessageBox.critical(self, "Optimization Error", f"Error during optimization:\n{error_message}")

    def display_results(self):
        """Display optimization results in table"""
        if not self.best_schedule:
            return

        # Setup table
        self.results_table.setRowCount(len(self.best_schedule))
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Course", "Instructor", "Classroom", "Day", "Time"
        ])

        # Fill table
        for i, assignment in enumerate(self.best_schedule):
            self.results_table.setItem(i, 0, QTableWidgetItem(assignment.course))
            self.results_table.setItem(i, 1, QTableWidgetItem(assignment.instructor))
            self.results_table.setItem(i, 2, QTableWidgetItem(assignment.classroom))
            self.results_table.setItem(i, 3, QTableWidgetItem(assignment.day))
            self.results_table.setItem(i, 4, QTableWidgetItem(assignment.time_slot))

        # Adjust column widths
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def export_results(self):
        """Export results to Excel"""
        if not self.best_schedule:
            QMessageBox.warning(self, "Warning", "No results available yet!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "optimized_schedule.xlsx",
            "Excel Files (*.xlsx)"
        )

        if file_path:
            try:
                # Create scheduler instance to use export method
                scheduler = CustomGeneticScheduler(self.excel_file)
                scheduler.export_enhanced_results(self.best_schedule, self.violations, file_path)

                QMessageBox.information(self, "Success", f"Results saved successfully:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving results:\n{str(e)}")

def main():
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("GA Scheduler")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("University Scheduler")

    # Create and show main window
    window = GASchedulerGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
