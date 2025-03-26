import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QTableWidget, QTableWidgetItem, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QGroupBox, QFormLayout, QLineEdit, QMessageBox,
                             QSplitter, QTextEdit, QCheckBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor
import torch
import time
import random

# Import the reliable scheduler 
from scheduler_main import SchedulingProblem, SchedulingConfig, ReliableScheduler, create_adaptive_config

class SchedulerWorker(QThread):
    """Worker thread for running the scheduler algorithm without freezing the UI"""
    progress_update = pyqtSignal(int, dict)
    finished = pyqtSignal(object, float, dict)
    error = pyqtSignal(str)
    
    def __init__(self, problem, config):
        super().__init__()
        self.problem = problem
        self.config = config
        self.running = True
        
    def run(self):
        try:
            # Create our scheduler
            scheduler = ReliableScheduler(self.problem, self.config)
            
            # Store original evolve method to monkey patch
            original_evolve = scheduler.evolve
            
            def wrapped_evolve():
                """Custom evolve method to update UI with progress"""
                start_time = time.time()
                population = None
                try:
                    # Initialize population
                    population = scheduler.initialize_population()
                    generations_without_improvement = 0
                    population_refreshes = 0
                    best_violations = None
                    
                    for generation in range(self.config.num_generations):
                        if not self.running:
                            return None, 0.0, {}
                            
                        # Process population in batches
                        all_fitness_scores = []
                        all_violations = []
                        
                        batch_size = self.config.batch_size
                        for i in range(0, len(population), batch_size):
                            batch = population[i:i+batch_size]
                            fitness_scores, violations = scheduler.calculate_fitness_batch(batch)
                            all_fitness_scores.append(fitness_scores)
                            all_violations.append(violations)
                        
                        fitness_scores = torch.cat(all_fitness_scores)
                        
                        # Update best solution
                        best_idx = torch.argmax(fitness_scores)
                        current_best_fitness = fitness_scores[best_idx].item()
                        
                        if not hasattr(scheduler, 'best_solution') or scheduler.best_solution is None or current_best_fitness > scheduler.best_fitness:
                            scheduler.best_fitness = current_best_fitness
                            scheduler.best_solution = population[best_idx].clone()
                            batch_idx = best_idx // batch_size
                            if batch_idx < len(all_violations):
                                best_violations = all_violations[batch_idx]
                            generations_without_improvement = 0
                        else:
                            generations_without_improvement += 1
                        
                        if not hasattr(scheduler, 'fitness_history'):
                            scheduler.fitness_history = []
                        scheduler.fitness_history.append(scheduler.best_fitness)
                        
                        # Emit progress update
                        elapsed = time.time() - start_time
                        progress = int((generation / self.config.num_generations) * 100)
                        
                        self.progress_update.emit(progress, {
                            "generation": generation,
                            "fitness": scheduler.best_fitness,
                            "violations": best_violations or {},
                            "elapsed": elapsed,
                            "eta": (elapsed / (generation + 1)) * (self.config.num_generations - generation) if generation > 0 else 0
                        })
                        
                        # Check stopping conditions
                        if scheduler.best_fitness >= self.config.target_fitness:
                            print("Target fitness reached!")
                            break
                        
                        # Apply local search to best solution occasionally
                        if generation % 10 == 0 and scheduler.best_solution is not None:
                            improved = scheduler._local_search(scheduler.best_solution)
                            improved_fitness, _ = scheduler.calculate_fitness_batch(improved.unsqueeze(0))
                            improved_fitness = improved_fitness.item()
                            
                            if improved_fitness > scheduler.best_fitness:
                                scheduler.best_solution = improved
                                scheduler.best_fitness = improved_fitness
                                print(f"Improved with local search: {scheduler.best_fitness:.4f}")
                                generations_without_improvement = 0
                        
                        # Refresh population if stagnant
                        if generations_without_improvement >= 15:
                            if population_refreshes < 3:
                                print("Refreshing population...")
                                population = scheduler._maintain_diversity(population, fitness_scores)
                                generations_without_improvement = 0
                                population_refreshes += 1
                            else:
                                print("No improvement for many generations. Stopping early.")
                                break
                        
                        # Create new population
                        new_population = []
                        
                        # Elitism - keep best individuals
                        elite_indices = torch.argsort(fitness_scores, descending=True)[:self.config.elite_size]
                        for idx in elite_indices:
                            new_population.append(population[idx].clone())
                        
                        # Breeding
                        while len(new_population) < self.config.population_size:
                            # Select parents
                            parent1 = scheduler._tournament_select(population, fitness_scores)
                            parent2 = scheduler._tournament_select(population, fitness_scores)
                            
                            # Create offspring
                            child1, child2 = scheduler._crossover(parent1, parent2)
                            
                            # Mutation
                            if random.random() < self.config.mutation_rate:
                                child1 = scheduler._mutate(child1)
                            if random.random() < self.config.mutation_rate:
                                child2 = scheduler._mutate(child2)
                            
                            # Add to new population
                            new_population.append(child1)
                            if len(new_population) < self.config.population_size:
                                new_population.append(child2)
                        
                        # Update population
                        population = torch.stack(new_population[:self.config.population_size])
                        
                        # Memory management
                        if generation % 10 == 0 and self.config.device == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # Final local search
                    if scheduler.best_solution is not None:
                        print("Performing final optimization...")
                        
                        # Save original solution in case local search doesn't help
                        original_solution = scheduler.best_solution.clone()
                        original_fitness = scheduler.best_fitness
                        
                        # Try more aggressive local search
                        improved_solution = scheduler._local_search(scheduler.best_solution, max_attempts=100)
                        improved_fitness, improved_violations = scheduler.calculate_fitness_batch(improved_solution.unsqueeze(0))
                        improved_fitness = improved_fitness.item()
                        
                        # Only update if actually improved
                        if improved_fitness > original_fitness:
                            scheduler.best_solution = improved_solution
                            scheduler.best_fitness = improved_fitness
                            best_violations = improved_violations
                            print(f"Final optimization improved fitness to {scheduler.best_fitness:.4f}")
                        else:
                            # Keep original solution
                            scheduler.best_solution = original_solution
                            # Recompute violations to ensure consistency
                            _, best_violations = scheduler.calculate_fitness_batch(scheduler.best_solution.unsqueeze(0))
                    
                    return scheduler.best_solution, scheduler.best_fitness, best_violations
                    
                except Exception as e:
                    print(f"Error during evolution: {e}")
                    if hasattr(scheduler, 'best_solution') and scheduler.best_solution is not None:
                        return scheduler.best_solution, scheduler.best_fitness, best_violations or {}
                    elif population is not None and len(population) > 0:
                        # If we have any population, return the best individual
                        fitness_scores, violations = scheduler.calculate_fitness_batch(population)
                        best_idx = torch.argmax(fitness_scores)
                        return population[best_idx], fitness_scores[best_idx].item(), violations
                    else:
                        # Return a random solution if nothing else
                        random_solution = scheduler._create_random_schedule()
                        fitness, violations = scheduler.calculate_fitness_batch(random_solution.unsqueeze(0))
                        return random_solution, fitness.item(), violations
            
            # Replace evolve method with our wrapped version
            scheduler.evolve = wrapped_evolve
            
            # Call the evolve method
            solution, fitness, violations = scheduler.evolve()
            
            # Emit the result
            if solution is not None:
                self.finished.emit(solution, fitness, violations)
        
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self.running = False


class FitnessPlot(FigureCanvas):
    """A canvas that updates itself every second with fitness progress"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        self.fitness_history = []
        self.generation_history = []
        
        # Set some plot properties
        self.axes.set_title('Fitness Progress')
        self.axes.set_xlabel('Generation')
        self.axes.set_ylabel('Fitness Score')
        self.axes.grid(True)
        
        self.fig.tight_layout()
    
    def update_plot(self, generation, fitness):
        """Update the plot with new data"""
        self.generation_history.append(generation)
        self.fitness_history.append(fitness)
        
        self.axes.clear()
        self.axes.plot(self.generation_history, self.fitness_history, 'b-')
        self.axes.set_title('Fitness Progress')
        self.axes.set_xlabel('Generation')
        self.axes.set_ylabel('Fitness Score')
        self.axes.grid(True)
        
        # Set y-axis limits to be between 0 and 1
        self.axes.set_ylim(0, 1)
        
        # Add current fitness to plot
        if len(self.fitness_history) > 0:
            current_fitness = self.fitness_history[-1]
            self.axes.text(0.02, 0.95, f'Current Fitness: {current_fitness:.4f}', 
                          transform=self.axes.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', 
                                                           facecolor='white', 
                                                           alpha=0.8))
        
        self.fig.tight_layout()
        self.draw()
    
    def reset(self):
        """Reset the plot"""
        self.fitness_history = []
        self.generation_history = []
        self.axes.clear()
        self.axes.set_title('Fitness Progress')
        self.axes.set_xlabel('Generation')
        self.axes.set_ylabel('Fitness Score')
        self.axes.grid(True)
        self.fig.tight_layout()
        self.draw()


class SchedulerMainWindow(QMainWindow):
    """Main application window for the course scheduler"""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Academic Course Scheduler")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize member variables
        self.problem = None
        self.config = None
        self.scheduler_worker = None
        self.output_file = "schedule_output.xlsx"
        
        # Set up the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget for different sections
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.setup_input_tab()
        self.setup_config_tab()
        self.setup_optimization_tab()
        self.setup_results_tab()
        
        # Status bar for displaying messages
        self.statusBar().showMessage("Ready")
        
        # Show the window
        self.show()
    
    def setup_input_tab(self):
        """Set up the data input tab"""
        input_tab = QWidget()
        layout = QVBoxLayout(input_tab)
        
        # Input file selection
        input_group = QGroupBox("Input Data File")
        input_layout = QFormLayout()
        input_group.setLayout(input_layout)
        
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setReadOnly(True)
        self.input_file_edit.setPlaceholderText("Select an Excel file with school data...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input_file)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.input_file_edit)
        file_layout.addWidget(browse_btn)
        
        input_layout.addRow("Data File:", file_layout)
        
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_data)
        input_layout.addRow("", load_btn)
        
        layout.addWidget(input_group)
        
        # Data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        
        # Add tabs for different data types
        preview_tabs = QTabWidget()
        
        # Create tables for each data type
        self.classrooms_table = QTableWidget()
        self.courses_table = QTableWidget()
        self.instructors_table = QTableWidget()
        self.students_table = QTableWidget()
        
        preview_tabs.addTab(self.classrooms_table, "Classrooms")
        preview_tabs.addTab(self.courses_table, "Courses")
        preview_tabs.addTab(self.instructors_table, "Instructors")
        preview_tabs.addTab(self.students_table, "Students")
        
        preview_layout.addWidget(preview_tabs)
        
        layout.addWidget(preview_group)
        
        # Add tab to main tab widget
        self.tabs.addTab(input_tab, "Data Input")
    
    def setup_config_tab(self):
        """Set up the configuration tab"""
        config_tab = QWidget()
        layout = QVBoxLayout(config_tab)
        
        # Algorithm parameters
        algo_group = QGroupBox("Genetic Algorithm Parameters")
        algo_layout = QFormLayout()
        algo_group.setLayout(algo_layout)
        
        # Population size
        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(50, 2000)
        self.population_size_spin.setValue(800)
        self.population_size_spin.setSingleStep(50)
        algo_layout.addRow("Population Size:", self.population_size_spin)
        
        # Number of generations
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(50, 5000)
        self.generations_spin.setValue(1000)
        self.generations_spin.setSingleStep(50)
        algo_layout.addRow("Number of Generations:", self.generations_spin)
        
        # Mutation rate
        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.0, 1.0)
        self.mutation_rate_spin.setValue(0.6)
        self.mutation_rate_spin.setSingleStep(0.05)
        algo_layout.addRow("Mutation Rate:", self.mutation_rate_spin)
        
        # Crossover rate
        self.crossover_rate_spin = QDoubleSpinBox()
        self.crossover_rate_spin.setRange(0.0, 1.0)
        self.crossover_rate_spin.setValue(0.85)
        self.crossover_rate_spin.setSingleStep(0.05)
        algo_layout.addRow("Crossover Rate:", self.crossover_rate_spin)
        
        # Elite size
        self.elite_size_spin = QSpinBox()
        self.elite_size_spin.setRange(1, 100)
        self.elite_size_spin.setValue(50)
        algo_layout.addRow("Elite Size:", self.elite_size_spin)
        
        # Target fitness
        self.target_fitness_spin = QDoubleSpinBox()
        self.target_fitness_spin.setRange(0.5, 1.0)
        self.target_fitness_spin.setValue(0.90)
        self.target_fitness_spin.setSingleStep(0.01)
        algo_layout.addRow("Target Fitness:", self.target_fitness_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(128)
        algo_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Number of phases
        self.phases_spin = QSpinBox()
        self.phases_spin.setRange(1, 5)
        self.phases_spin.setValue(3)
        algo_layout.addRow("Number of Phases:", self.phases_spin)
        
        # Sample size
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(50, 1000)
        self.sample_size_spin.setValue(200)
        algo_layout.addRow("Sample Size:", self.sample_size_spin)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        algo_layout.addRow("Computation Device:", self.device_combo)
        
        # Mixed precision
        self.mixed_precision_check = QCheckBox("Use Mixed Precision")
        self.mixed_precision_check.setChecked(True)
        if not torch.cuda.is_available():
            self.mixed_precision_check.setEnabled(False)
            self.mixed_precision_check.setToolTip("Mixed precision requires CUDA")
        algo_layout.addRow("", self.mixed_precision_check)
        
        # Use adaptive config
        self.adaptive_config_check = QCheckBox("Use Adaptive Configuration")
        self.adaptive_config_check.setChecked(True)
        self.adaptive_config_check.toggled.connect(self.toggle_adaptive_config)
        algo_layout.addRow("", self.adaptive_config_check)
        
        layout.addWidget(algo_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout()
        output_group.setLayout(output_layout)
        
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)
        self.output_file_edit.setText("schedule_output.xlsx")
        
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output_file)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(self.output_file_edit)
        output_file_layout.addWidget(output_browse_btn)
        
        output_layout.addRow("Output File:", output_file_layout)
        
        layout.addWidget(output_group)
        
        # Add some vertical space
        layout.addStretch(1)
        
        # Create config button
        create_config_btn = QPushButton("Create Configuration")
        create_config_btn.clicked.connect(self.create_configuration)
        layout.addWidget(create_config_btn)
        
        # Add tab to main tab widget
        self.tabs.addTab(config_tab, "Configuration")
        
    def setup_optimization_tab(self):
        """Set up the optimization tab"""
        optimization_tab = QWidget()
        layout = QVBoxLayout(optimization_tab)
        
        # Split the tab into two sections
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Top section - controls and progress
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        self.start_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop Optimization")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        top_layout.addLayout(controls_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        top_layout.addLayout(progress_layout)
        
        # Status, fitness, and time information
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Not started")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        
        status_layout.addWidget(QLabel("Current Fitness:"))
        self.fitness_label = QLabel("0.0000")
        status_layout.addWidget(self.fitness_label)
        
        status_layout.addStretch(1)
        
        status_layout.addWidget(QLabel("Elapsed:"))
        self.elapsed_label = QLabel("0s")
        status_layout.addWidget(self.elapsed_label)
        
        status_layout.addWidget(QLabel("ETA:"))
        self.eta_label = QLabel("--")
        status_layout.addWidget(self.eta_label)
        
        top_layout.addLayout(status_layout)
        
        # Add the top widget to the splitter
        splitter.addWidget(top_widget)
        
        # Bottom section - Visualization
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        
        # Tabs for different visualizations
        viz_tabs = QTabWidget()
        
        # Fitness progress plot
        fitness_widget = QWidget()
        fitness_layout = QVBoxLayout(fitness_widget)
        self.fitness_plot = FitnessPlot(width=5, height=4)
        fitness_layout.addWidget(self.fitness_plot)
        viz_tabs.addTab(fitness_widget, "Fitness Progress")
        
        # Violations tracking
        violations_widget = QWidget()
        violations_layout = QVBoxLayout(violations_widget)
        
        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(2)
        self.violations_table.setHorizontalHeaderLabels(["Constraint", "Violations"])
        self.violations_table.horizontalHeader().setStretchLastSection(True)
        
        violations_layout.addWidget(self.violations_table)
        viz_tabs.addTab(violations_widget, "Constraint Violations")
        
        # Add the visualization tabs to the bottom layout
        bottom_layout.addWidget(viz_tabs)
        
        # Add the bottom widget to the splitter
        splitter.addWidget(bottom_widget)
        
        # Set the initial splitter sizes
        splitter.setSizes([200, 600])
        
        # Add tab to main tab widget
        self.tabs.addTab(optimization_tab, "Optimization")
    
    def setup_results_tab(self):
        """Set up the results tab"""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        # Results control buttons
        controls_layout = QHBoxLayout()
        
        self.load_results_btn = QPushButton("Load Results")
        self.load_results_btn.clicked.connect(self.load_results)
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)
        
        controls_layout.addWidget(self.load_results_btn)
        controls_layout.addWidget(self.export_results_btn)
        controls_layout.addStretch(1)
        
        layout.addLayout(controls_layout)
        
        # Tabs for different result views
        results_tabs = QTabWidget()
        
        # Schedule table
        self.schedule_table = QTableWidget()
        results_tabs.addTab(self.schedule_table, "Schedule")
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        results_tabs.addTab(self.metrics_table, "Metrics")
        
        # Schedule visualization
        schedule_viz_widget = QWidget()
        schedule_viz_layout = QVBoxLayout(schedule_viz_widget)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Day", "Classroom", "Instructor"])
        filter_layout.addWidget(self.filter_combo)
        
        self.filter_value_combo = QComboBox()
        filter_layout.addWidget(self.filter_value_combo)
        
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_schedule_filter)
        filter_layout.addWidget(self.apply_filter_btn)
        
        filter_layout.addStretch(1)
        
        schedule_viz_layout.addLayout(filter_layout)
        
        # Filtered schedule table
        self.filtered_schedule_table = QTableWidget()
        schedule_viz_layout.addWidget(self.filtered_schedule_table)
        
        results_tabs.addTab(schedule_viz_widget, "Schedule View")
        
        layout.addWidget(results_tabs)
        
        # Add tab to main tab widget
        self.tabs.addTab(results_tab, "Results")
    
    def browse_input_file(self):
        """Open file dialog to select input Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Data File", "", "Excel Files (*.xlsx *.xls)"
        )
        
        if file_path:
            self.input_file_edit.setText(file_path)
    
    def browse_output_file(self):
        """Open file dialog to select output Excel file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "schedule_output.xlsx", "Excel Files (*.xlsx)"
        )
        
        if file_path:
            self.output_file_edit.setText(file_path)
            self.output_file = file_path
    
    def load_data(self):
        """Load data from the input file"""
        input_file = self.input_file_edit.text()
        
        if not input_file:
            QMessageBox.warning(self, "Input Error", "Please select an input data file.")
            return
        
        if not os.path.exists(input_file):
            QMessageBox.warning(self, "File Error", f"File not found: {input_file}")
            return
        
        try:
            self.statusBar().showMessage("Loading data...")
            
            # Load the data using the reliable scheduler code
            try:
                self.problem = SchedulingProblem(input_file)
                
                # Update tables
                self.update_table_from_df(self.classrooms_table, pd.DataFrame(self.problem.classrooms))
                self.update_table_from_df(self.courses_table, pd.DataFrame(self.problem.courses))
                self.update_table_from_df(self.instructors_table, pd.DataFrame(self.problem.instructors))
                self.update_table_from_df(self.students_table, pd.DataFrame(self.problem.students))
                
                # Enable the start button
                self.start_btn.setEnabled(True)
                
                # Update status
                self.statusBar().showMessage(f"Data loaded successfully. {len(self.problem.courses)} courses, {len(self.problem.classrooms)} classrooms.")
                
                # Switch to config tab
                self.tabs.setCurrentIndex(1)
                
            except Exception as e:
                # If there's an error loading with the reliable scheduler, try pandas directly
                self.statusBar().showMessage("Falling back to direct data loading...")
                
                xl = pd.ExcelFile(input_file)
                classrooms_df = pd.read_excel(xl, "Classrooms")
                courses_df = pd.read_excel(xl, "Courses")
                instructors_df = pd.read_excel(xl, "Instructors")
                students_df = pd.read_excel(xl, "Students")
                
                # Update the tables
                self.update_table_from_df(self.classrooms_table, classrooms_df)
                self.update_table_from_df(self.courses_table, courses_df)
                self.update_table_from_df(self.instructors_table, instructors_df)
                self.update_table_from_df(self.students_table, students_df)
                
                # Create the problem object
                self.problem = SchedulingProblem(input_file)
                
                # Enable the start button
                self.start_btn.setEnabled(True)
                
                # Update status
                self.statusBar().showMessage(f"Data loaded with fallback method. {len(courses_df)} courses, {len(classrooms_df)} classrooms.")
                
                # Switch to config tab
                self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"An error occurred: {str(e)}")
            self.statusBar().showMessage("Error loading data")
    
    def update_table_from_df(self, table, df):
        """Update a QTableWidget from a pandas DataFrame"""
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns)
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = str(df.iloc[i, j])
                table.setItem(i, j, QTableWidgetItem(value))
        
        # Resize columns to contents
        table.resizeColumnsToContents()
    
    def toggle_adaptive_config(self, checked):
        """Enable/disable config fields based on adaptive config checkbox"""
        enabled = not checked
        
        self.population_size_spin.setEnabled(enabled)
        self.generations_spin.setEnabled(enabled)
        self.mutation_rate_spin.setEnabled(enabled)
        self.crossover_rate_spin.setEnabled(enabled)
        self.elite_size_spin.setEnabled(enabled)
        self.target_fitness_spin.setEnabled(enabled)
        self.batch_size_spin.setEnabled(enabled)
        self.phases_spin.setEnabled(enabled)
        self.sample_size_spin.setEnabled(enabled)
    
    def create_configuration(self):
        """Create configuration from UI settings"""
        if self.problem is None:
            QMessageBox.warning(self, "Missing Data", "Please load data first.")
            return
        
        try:
            # Create configuration
            if self.adaptive_config_check.isChecked():
                # Use adaptive configuration
                self.config = create_adaptive_config(self.problem)
                
                # Update the UI with adaptive values
                self.population_size_spin.setValue(self.config.population_size)
                self.generations_spin.setValue(self.config.num_generations)
                self.mutation_rate_spin.setValue(self.config.mutation_rate)
                self.crossover_rate_spin.setValue(self.config.crossover_rate)
                self.elite_size_spin.setValue(self.config.elite_size)
                self.target_fitness_spin.setValue(self.config.target_fitness)
                self.batch_size_spin.setValue(self.config.batch_size)
                self.phases_spin.setValue(self.config.num_phases)
                self.sample_size_spin.setValue(self.config.sample_size)
            else:
                # Use manual configuration
                device = "cuda" if self.device_combo.currentText() == "CUDA (GPU)" else "cpu"
                
                self.config = SchedulingConfig(
                    population_size=self.population_size_spin.value(),
                    num_generations=self.generations_spin.value(),
                    mutation_rate=self.mutation_rate_spin.value(),
                    crossover_rate=self.crossover_rate_spin.value(),
                    elite_size=self.elite_size_spin.value(),
                    tournament_size=max(3, int(self.population_size_spin.value() * 0.1)),
                    target_fitness=self.target_fitness_spin.value(),
                    batch_size=self.batch_size_spin.value(),
                    num_phases=self.phases_spin.value(),
                    device=device,
                    use_mixed_precision=self.mixed_precision_check.isChecked(),
                    sample_size=self.sample_size_spin.value()
                )
            
            self.statusBar().showMessage("Configuration created successfully")
            
            # Enable start button
            self.start_btn.setEnabled(True)
            
            # Switch to optimization tab
            self.tabs.setCurrentIndex(2)
            
            QMessageBox.information(self, "Configuration Created", 
                                   "Optimization configuration has been created successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Creating Configuration", f"An error occurred: {str(e)}")
    
    def start_optimization(self):
        """Start the optimization process"""
        if self.problem is None or self.config is None:
            QMessageBox.warning(self, "Missing Configuration", 
                               "Please load data and create configuration first.")
            return
        
        # If there's an existing worker thread, make sure it's cleaned up
        if self.scheduler_worker and self.scheduler_worker.isRunning():
            self.scheduler_worker.stop()
            self.scheduler_worker.wait()
        
        # Reset the progress visualization
        self.progress_bar.setValue(0)
        self.status_label.setText("Running")
        self.fitness_label.setText("0.0000")
        self.elapsed_label.setText("0s")
        self.eta_label.setText("--")
        self.fitness_plot.reset()
        
        # Clear the violations table and set up rows with new priorities
        self.violations_table.setRowCount(7)
        self.violations_table.setItem(0, 0, QTableWidgetItem("1. Instructor Availability (HIGHEST)"))
        self.violations_table.setItem(1, 0, QTableWidgetItem("2. Room Type Compatibility (HIGH)"))
        self.violations_table.setItem(2, 0, QTableWidgetItem("3. Room Double Booking (HIGH)"))
        self.violations_table.setItem(3, 0, QTableWidgetItem("4. Instructor Conflicts"))
        self.violations_table.setItem(4, 0, QTableWidgetItem("5. Classroom Capacity"))
        self.violations_table.setItem(5, 0, QTableWidgetItem("Over Capacity Rooms"))
        self.violations_table.setItem(6, 0, QTableWidgetItem("Students with Conflicts (LOWEST)"))
        
        for i in range(7):
            self.violations_table.setItem(i, 1, QTableWidgetItem("0"))
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create and start the worker thread
        self.scheduler_worker = SchedulerWorker(self.problem, self.config)
        self.scheduler_worker.progress_update.connect(self.update_progress)
        self.scheduler_worker.finished.connect(self.optimization_finished)
        self.scheduler_worker.error.connect(self.optimization_error)
        self.scheduler_worker.start()
        
        self.statusBar().showMessage("Optimization started")
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.scheduler_worker and self.scheduler_worker.isRunning():
            self.status_label.setText("Stopping...")
            self.statusBar().showMessage("Stopping optimization...")
            
            # Set the running flag to false to stop the algorithm
            self.scheduler_worker.stop()
            
            # Wait for the thread to terminate (with timeout)
            if not self.scheduler_worker.wait(3000):  # 3 second timeout
                self.scheduler_worker.terminate()
                self.scheduler_worker.wait()
            
            # Re-enable the start button
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopped")
            self.statusBar().showMessage("Optimization stopped by user")
    
    def update_progress(self, progress, data):
        """Update progress information from the worker thread"""
        self.progress_bar.setValue(progress)
        
        # Update fitness
        if 'fitness' in data:
            self.fitness_label.setText(f"{data['fitness']:.4f}")
            # Update fitness plot
            self.fitness_plot.update_plot(data['generation'], data['fitness'])
        
        # Update time information
        if 'elapsed' in data:
            elapsed_seconds = data['elapsed']
            if elapsed_seconds < 60:
                elapsed_text = f"{elapsed_seconds:.1f}s"
            else:
                minutes = int(elapsed_seconds // 60)
                seconds = elapsed_seconds % 60
                elapsed_text = f"{minutes}m {seconds:.1f}s"
            self.elapsed_label.setText(elapsed_text)
        
        if 'eta' in data:
            eta_seconds = data['eta']
            if eta_seconds < 60:
                eta_text = f"{eta_seconds:.1f}s"
            else:
                minutes = int(eta_seconds // 60)
                seconds = eta_seconds % 60
                eta_text = f"{minutes}m {seconds:.1f}s"
            self.eta_label.setText(eta_text)
        
        # Update violations table
        if 'violations' in data:
            violations = data['violations']
            
            # Set up rows for new constraint priorities if needed
            if self.violations_table.rowCount() < 7:
                self.violations_table.setRowCount(7)
                self.violations_table.setItem(0, 0, QTableWidgetItem("1. Instructor Availability (HIGHEST)"))
                self.violations_table.setItem(1, 0, QTableWidgetItem("2. Room Type Compatibility (HIGH)"))
                self.violations_table.setItem(2, 0, QTableWidgetItem("3. Room Double Booking (HIGH)"))
                self.violations_table.setItem(3, 0, QTableWidgetItem("4. Instructor Conflicts"))
                self.violations_table.setItem(4, 0, QTableWidgetItem("5. Classroom Capacity"))
                self.violations_table.setItem(5, 0, QTableWidgetItem("Over Capacity Rooms"))
                self.violations_table.setItem(6, 0, QTableWidgetItem("Students with Conflicts (LOWEST)"))
            
            # Update with new labels
            if 'Instructor Availability' in violations:
                self.violations_table.setItem(0, 1, QTableWidgetItem(str(violations['Instructor Availability'])))
            
            if 'Classroom Compatibility' in violations:
                self.violations_table.setItem(1, 1, QTableWidgetItem(str(violations['Classroom Compatibility'])))
            
            if 'Room Conflicts' in violations:
                self.violations_table.setItem(2, 1, QTableWidgetItem(str(violations['Room Conflicts'])))
            
            if 'Instructor Booking Violations' in violations:
                self.violations_table.setItem(3, 1, QTableWidgetItem(str(violations['Instructor Booking Violations'])))
            
            if 'Classroom Capacity' in violations:
                self.violations_table.setItem(4, 1, QTableWidgetItem(str(violations['Classroom Capacity'])))
            
            if 'Over Capacity Rooms' in violations:
                self.violations_table.setItem(5, 1, QTableWidgetItem(
                    f"{violations['Over Capacity Rooms']} (Allowed: {violations.get('Allowed Capacity Violations', 0)})"
                ))
            
            if 'Students with Conflicts' in violations:
                self.violations_table.setItem(6, 1, QTableWidgetItem(str(violations['Students with Conflicts'])))
    
    def optimization_finished(self, solution, fitness, violations):
        """Handle completion of optimization"""
        # Reset UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Completed")
        
        # Clean up worker thread
        if self.scheduler_worker:
            if self.scheduler_worker.isRunning():
                self.scheduler_worker.stop()
                self.scheduler_worker.wait()
            self.scheduler_worker = None
        
        # If no solution was found (e.g., stopped early)
        if solution is None:
            self.statusBar().showMessage("Optimization terminated without a solution")
            return
        
        # Create scheduler object to export results
        scheduler = ReliableScheduler(self.problem, self.config)
        scheduler.best_solution = solution
        scheduler.best_fitness = fitness
        
        try:
            # Export results to the specified output file
            scheduler.export_results(self.output_file)
            
            # Enable export button
            self.export_results_btn.setEnabled(True)
            
            # Display completion message
            QMessageBox.information(self, "Optimization Complete", 
                                   f"Optimization completed with fitness score: {fitness:.4f}\n"
                                   f"Results exported to: {self.output_file}")
            
            # Load the results into the results tab
            self.load_results()
            
            # Switch to results tab
            self.tabs.setCurrentIndex(3)
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting results: {str(e)}")
        
        self.statusBar().showMessage(f"Optimization completed. Fitness: {fitness:.4f}")
    
    def optimization_error(self, error_msg):
        """Handle errors from the optimization process"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error")
        
        # Clean up worker thread
        if self.scheduler_worker:
            if self.scheduler_worker.isRunning():
                self.scheduler_worker.stop()
                self.scheduler_worker.wait()
            self.scheduler_worker = None
        
        QMessageBox.critical(self, "Optimization Error", f"An error occurred: {error_msg}")
        self.statusBar().showMessage("Optimization failed with error")
    
    def load_results(self):
        """Load results from the output Excel file"""
        if not os.path.exists(self.output_file):
            QMessageBox.warning(self, "File Not Found", f"Output file not found: {self.output_file}")
            return
        
        try:
            # Load the schedule and metrics data
            xl = pd.ExcelFile(self.output_file)
            schedule_df = pd.read_excel(xl, "Schedule")
            metrics_df = pd.read_excel(xl, "Metrics")
            
            # Update the schedule table
            self.update_table_from_df(self.schedule_table, schedule_df)
            
            # Update metrics table
            self.metrics_table.setRowCount(metrics_df.shape[0])
            for i in range(metrics_df.shape[0]):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(str(metrics_df.iloc[i, 0])))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(str(metrics_df.iloc[i, 1])))
            
            # Update filter options
            self.update_filter_options(schedule_df)
            
            # Enable export button
            self.export_results_btn.setEnabled(True)
            
            self.statusBar().showMessage("Results loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Results", f"An error occurred: {str(e)}")
    
    def update_filter_options(self, schedule_df):
        """Update filter options based on schedule data"""
        self.filter_combo.currentTextChanged.connect(lambda: self.update_filter_values(schedule_df))
        
        # Initial update
        self.update_filter_values(schedule_df)
    
    def update_filter_values(self, schedule_df):
        """Update filter value options based on selected filter type"""
        filter_type = self.filter_combo.currentText()
        self.filter_value_combo.clear()
        
        if filter_type == "Day":
            unique_values = schedule_df["Day"].unique()
            self.filter_value_combo.addItems(unique_values)
        elif filter_type == "Classroom":
            unique_values = schedule_df["Classroom"].unique()
            self.filter_value_combo.addItems(unique_values)
        elif filter_type == "Instructor":
            unique_values = schedule_df["Instructor"].unique()
            self.filter_value_combo.addItems(unique_values)
    
    def apply_schedule_filter(self):
        """Apply filter to the schedule view"""
        if not os.path.exists(self.output_file):
            return
        
        try:
            schedule_df = pd.read_excel(self.output_file, "Schedule")
            
            filter_type = self.filter_combo.currentText()
            filter_value = self.filter_value_combo.currentText()
            
            if filter_type == "Day":
                filtered_df = schedule_df[schedule_df["Day"] == filter_value]
            elif filter_type == "Classroom":
                filtered_df = schedule_df[schedule_df["Classroom"] == filter_value]
            elif filter_type == "Instructor":
                filtered_df = schedule_df[schedule_df["Instructor"] == filter_value]
            else:
                filtered_df = schedule_df
            
            # Sort the filtered results
            if filter_type == "Day":
                filtered_df = filtered_df.sort_values(["Start Time", "Classroom"])
            else:
                filtered_df = filtered_df.sort_values(["Day", "Start Time"])
            
            # Update the filtered table
            self.update_table_from_df(self.filtered_schedule_table, filtered_df)
            
        except Exception as e:
            QMessageBox.warning(self, "Filter Error", f"Error applying filter: {str(e)}")
    
    def export_results(self):
        """Export results to a new file"""
        if not os.path.exists(self.output_file):
            QMessageBox.warning(self, "File Not Found", f"Output file not found: {self.output_file}")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                # Just copy the output file to the new location
                import shutil
                shutil.copy2(self.output_file, file_path)
                
                QMessageBox.information(self, "Export Successful", 
                                       f"Results exported to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred: {str(e)}")

# For direct execution
if __name__ == "__main__":
    # Enable high DPI scaling
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    window = SchedulerMainWindow()
    sys.exit(app.exec_())   