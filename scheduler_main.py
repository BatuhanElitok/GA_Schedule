import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import random
from datetime import datetime
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SchedulingConfig:
    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    elite_size: int
    tournament_size: int
    target_fitness: float
    batch_size: int
    num_phases: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TimeSlot:
    day: str
    start_time: float
    end_time: float

    def overlaps(self, other: 'TimeSlot') -> bool:
        return (self.day == other.day and
                not (self.end_time <= other.start_time or 
                     self.start_time >= other.end_time))

class SchedulingProblem:
    def __init__(self, input_file: str):
        self.load_data(input_file)
        self.initialize_time_slots()
        self.preprocess_data()
    
    def load_data(self, input_file: str):
        print(f"Loading data from {input_file}...")
        xl = pd.ExcelFile(input_file)
        self.classrooms = pd.read_excel(xl, "Classrooms").to_dict('records')
        self.courses = pd.read_excel(xl, "Courses").to_dict('records')
        self.instructors = pd.read_excel(xl, "Instructors").to_dict('records')
        self.students = pd.read_excel(xl, "Students").to_dict('records')
        print("Data loaded successfully!")

    def initialize_time_slots(self):
        self.time_slots = []
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        times = [8.40, 9.40, 10.40, 11.40, 12.40, 13.40, 14.40, 15.40, 16.40]
        
        for day in days:
            for i in range(len(times) - 2):
                self.time_slots.append(TimeSlot(
                    day=day,
                    start_time=times[i],
                    end_time=times[i + 2] + 1.0
                ))

    def preprocess_data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Course enrollments
        course_name_to_idx = {course['Name']: i for i, course in enumerate(self.courses)}
        enrollment_indices = []
        
        for student_idx, student in enumerate(self.students):
            courses = student['Courses'].split(', ') if isinstance(student['Courses'], str) else []
            for course_name in courses:
                if course_name in course_name_to_idx:
                    course_idx = course_name_to_idx[course_name]
                    enrollment_indices.append([course_idx, student_idx])
        
        if enrollment_indices:
            indices = torch.tensor(enrollment_indices, dtype=torch.long).t()
            values = torch.ones(len(enrollment_indices))
            self.student_course_matrix = torch.sparse_coo_tensor(
                indices, values, 
                (len(self.courses), len(self.students)),
                device=device
            )
        else:
            self.student_course_matrix = torch.sparse_coo_tensor(
                size=(len(self.courses), len(self.students)),
                device=device
            )
        
        # Course sizes
        course_sizes = []
        for course in self.courses:
            size = sum(1 for student in self.students 
                      if isinstance(student['Courses'], str) and 
                      course['Name'] in student['Courses'].split(', '))
            course_sizes.append(size)
        
        self.course_enrollments = torch.tensor(course_sizes, dtype=torch.float32, device=device)
        
        # Classroom capacities
        self.classroom_capacities = torch.tensor(
            [c['Capacity'] for c in self.classrooms],
            dtype=torch.float32,
            device=device
        )
        
        # Classroom compatibility
        compatibility_matrix = torch.zeros(
            (len(self.courses), len(self.classrooms)),
            dtype=torch.bool,
            device=device
        )
        for course_idx, course in enumerate(self.courses):
            for classroom_idx, classroom in enumerate(self.classrooms):
                compatibility_matrix[course_idx, classroom_idx] = (
                    course['Classroom Type'] == classroom['Type']
                )
        self.classroom_compatibility = compatibility_matrix
        
        # Instructor availability
        availability_matrix = torch.zeros(
            (len(self.instructors), len(self.time_slots)),
            dtype=torch.bool,
            device=device
        )
        for instr_idx, instructor in enumerate(self.instructors):
            for slot_idx, time_slot in enumerate(self.time_slots):
                day_slots = instructor.get(f"{time_slot.day} Slots", "")
                if isinstance(day_slots, str):
                    for slot_range in day_slots.split(" | "):
                        if not slot_range:
                            continue
                        try:
                            start, end = map(float, slot_range.split("-"))
                            if start <= time_slot.start_time and end >= time_slot.end_time:
                                availability_matrix[instr_idx, slot_idx] = True
                        except (ValueError, TypeError):
                            continue
        
        self.instructor_availability = availability_matrix

def create_adaptive_config(problem: SchedulingProblem) -> SchedulingConfig:
    num_courses = len(problem.courses)
    
    # Scale configuration based on problem size
    population_size = min(800, int(num_courses * 3))
    num_generations = min(1000, int(num_courses * 4))
    batch_size = min(128, population_size)  # Adjust based on GPU memory
    
    return SchedulingConfig(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=0.6,
        crossover_rate=0.85,
        elite_size=int(population_size * 0.15),
        tournament_size=int(population_size * 0.1),
        target_fitness=0.90,
        batch_size=batch_size,
        num_phases=3
    )

class GeneticScheduler:
    def __init__(self, problem: SchedulingProblem, config: SchedulingConfig):
        self.problem = problem
        self.config = config
        self.device = torch.device(config.device)
        self.best_solution = None
        self.best_fitness = 0.0
        self.fitness_history = []
        
        # Move data to device
        self.course_enrollments = problem.course_enrollments.to(self.device)
        self.classroom_capacities = problem.classroom_capacities.to(self.device)
        self.classroom_compatibility = problem.classroom_compatibility.to(self.device)
        self.instructor_availability = problem.instructor_availability.to(self.device)
        self.student_course_matrix = problem.student_course_matrix.to(self.device)

    def initialize_population(self) -> torch.Tensor:
        num_courses = len(self.problem.courses)
        population = torch.zeros(
            (self.config.population_size, num_courses, 3),
            device=self.device
        )
        
        # Initialize with semi-intelligent assignments
        for i in range(self.config.population_size):
            schedule = self._create_intelligent_schedule()
            population[i] = schedule
        
        return population

    def _create_intelligent_schedule(self) -> torch.Tensor:
        num_courses = len(self.problem.courses)
        schedule = torch.zeros(num_courses, 3, device=self.device)
        
        # Sort courses by enrollment size (descending)
        course_sizes = self.course_enrollments.cpu().numpy()
        course_order = np.argsort(-course_sizes)
        
        for course_idx in course_order:
            # Find compatible classrooms
            compatible_rooms = torch.where(self.classroom_compatibility[course_idx])[0]
            if len(compatible_rooms) == 0:
                # Fallback to random assignment if no compatible rooms
                schedule[course_idx, 1] = random.randint(0, len(self.problem.classrooms)-1)
            else:
                # Choose random compatible room
                schedule[course_idx, 1] = compatible_rooms[
                    random.randint(0, len(compatible_rooms)-1)
                ]
            
            # Assign random time slot and instructor
            schedule[course_idx, 0] = random.randint(0, len(self.problem.time_slots)-1)
            schedule[course_idx, 2] = random.randint(0, len(self.problem.instructors)-1)
        
        return schedule

    @autocast()
    def calculate_fitness_batch(self, schedules: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Calculate fitness scores for a batch of schedules with proper type handling.
        """
        batch_size = len(schedules)
        penalties = torch.zeros(batch_size, device=self.device)
        
        # Split schedules into components
        time_slots = schedules[:, :, 0].long()
        classrooms = schedules[:, :, 1].long()
        instructors = schedules[:, :, 2].long()
        
        # 1. Classroom capacity violations (count actual overflow)
        capacity_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            for c in range(len(self.problem.courses)):
                room_idx = classrooms[b, c]
                course_size = self.course_enrollments[c].long()
                room_capacity = self.classroom_capacities[room_idx].long()
                if course_size > room_capacity:
                    overflow = int((course_size - room_capacity).item())
                    capacity_violations[b] += overflow
        penalties += capacity_violations * 5.0
        
        # 2. Classroom type compatibility
        compatibility_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            for c in range(len(self.problem.courses)):
                if not self.classroom_compatibility[c, classrooms[b, c]]:
                    compatibility_violations[b] += 1
        penalties += compatibility_violations * 10.0
        
        # 3. Instructor conflicts (same instructor, same time slot)
        instructor_conflicts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            instructor_schedule = {}
            for c in range(len(self.problem.courses)):
                time = int(time_slots[b, c])
                instructor = int(instructors[b, c])
                day = time // 10  # Get the day portion
                
                key = (instructor, day, time)
                if key in instructor_schedule:
                    instructor_conflicts[b] += 1
                else:
                    instructor_schedule[key] = c
        penalties += instructor_conflicts * 8.0
        
        # 4. Student conflicts (count affected students)
        affected_students = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        indices = self.student_course_matrix._indices()
        values = self.student_course_matrix._values()
        
        # Sample students for efficiency
        unique_students = torch.unique(indices[1])
        sample_size = min(200, len(unique_students))
        sampled_students = unique_students[torch.randperm(len(unique_students))[:sample_size]]
        
        for b in range(batch_size):
            students_with_conflicts = set()  # Track unique students with conflicts
            for student_idx in sampled_students:
                # Get courses for this student
                student_mask = indices[1] == student_idx
                student_courses = indices[0][student_mask]
                
                if len(student_courses) > 0:
                    # Get time slots for student's courses
                    course_times = time_slots[b][student_courses]
                    days = course_times // 10  # Group by day
                    
                    has_conflict = False
                    # Check conflicts within each day
                    for day in torch.unique(days):
                        day_times = course_times[days == day]
                        # Sort times to check for overlaps
                        day_times_sorted, _ = torch.sort(day_times)
                        
                        # Check consecutive times for overlaps
                        for i in range(len(day_times_sorted) - 1):
                            if day_times_sorted[i] == day_times_sorted[i + 1]:
                                has_conflict = True
                                break
                        
                        if has_conflict:
                            students_with_conflicts.add(int(student_idx))
                            break
            
            # Scale the count based on sampling
            if sample_size < len(unique_students):
                scale_factor = len(unique_students) / sample_size
                affected_students[b] = int(len(students_with_conflicts) * scale_factor)
            else:
                affected_students[b] = len(students_with_conflicts)
        
        penalties += affected_students * 2.0
        
        # Calculate total violations for reporting
        total_violations = {
            "Classroom Capacity": int(torch.sum(capacity_violations).item()),
            "Classroom Compatibility": int(torch.sum(compatibility_violations).item()),
            "Instructor Conflicts": int(torch.sum(instructor_conflicts).item()),
            "Students with Conflicts": int(torch.sum(affected_students).item())
        }
        
        # Calculate fitness scores (0 to 1)
        max_penalties = 25.0 * len(self.problem.courses)
        fitness_scores = torch.exp(-penalties / max_penalties)
        
        return torch.clamp(fitness_scores, 0.0, 1.0), total_violations


    def _tournament_select(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        tournament_size = self.config.tournament_size
        idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = fitness_scores[idx]
        winner_idx = idx[torch.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.config.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        # Multi-point crossover
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Swap segments between crossover points
        child1[crossover_points[0]:crossover_points[1]] = parent2[crossover_points[0]:crossover_points[1]]
        child2[crossover_points[0]:crossover_points[1]] = parent1[crossover_points[0]:crossover_points[1]]
        
        return child1, child2

    def _mutate(self, schedule: torch.Tensor) -> torch.Tensor:
        mutated = schedule.clone()
        num_mutations = max(4, int(len(schedule) * self.config.mutation_rate))
        
        for _ in range(num_mutations):
            idx = random.randint(0, len(schedule) - 1)
            mutation_type = random.random()
            
            if mutation_type < 0.5:  # Classroom mutation
                compatible_rooms = torch.where(self.classroom_compatibility[idx])[0]
                if len(compatible_rooms) > 0:
                    mutated[idx, 1] = compatible_rooms[
                        random.randint(0, len(compatible_rooms)-1)
                    ]
            
            elif mutation_type < 0.8:  # Time slot mutation
                # Try to find a time slot with fewer conflicts
                current_conflicts = float('inf')
                best_time_slot = None
                
                for _ in range(5):  # Try 5 random time slots
                    time_slot = random.randint(0, len(self.problem.time_slots) - 1)
                    mutated[idx, 0] = time_slot
                    conflicts = self._count_local_conflicts(mutated, idx)
                    
                    if conflicts < current_conflicts:
                        current_conflicts = conflicts
                        best_time_slot = time_slot
                
                if best_time_slot is not None:
                    mutated[idx, 0] = best_time_slot
            
            else:  # Instructor mutation
                # Try to find an available instructor
                available_instructors = []
                time_slot = int(mutated[idx, 0])
                
                for i in range(len(self.problem.instructors)):
                    if self.instructor_availability[i, time_slot]:
                        available_instructors.append(i)
                
                if available_instructors:
                    mutated[idx, 2] = random.choice(available_instructors)
        
        return mutated

    def _count_local_conflicts(self, schedule: torch.Tensor, idx: int) -> int:
        conflicts = 0
        time_slot = int(schedule[idx, 0])
        instructor = int(schedule[idx, 2])
        classroom = int(schedule[idx, 1])
        
        # Check instructor conflicts
        for i in range(len(schedule)):
            if i != idx:
                if (int(schedule[i, 0]) == time_slot and 
                    int(schedule[i, 2]) == instructor):
                    conflicts += 1
        
        # Check classroom conflicts
        for i in range(len(schedule)):
            if i != idx:
                if (int(schedule[i, 0]) == time_slot and 
                    int(schedule[i, 1]) == classroom):
                    conflicts += 1
        
        # Check classroom compatibility
        if not self.classroom_compatibility[idx, classroom]:
            conflicts += 1
        
        return conflicts

    def _local_search(self, solution: torch.Tensor, max_attempts: int = 20) -> torch.Tensor:
        best_solution = solution.clone()
        best_fitness, _ = self.calculate_fitness_batch(best_solution.unsqueeze(0))
        best_fitness = best_fitness.item()
        
        for _ in range(max_attempts):
            candidate = self._mutate(best_solution)
            fitness, _ = self.calculate_fitness_batch(candidate.unsqueeze(0))
            fitness = fitness.item()
            
            if fitness > best_fitness:
                best_solution = candidate
                best_fitness = fitness
        
        return best_solution

    def _maintain_diversity(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        # Use k-means clustering to identify similar solutions
        flattened_pop = population.reshape(len(population), -1).cpu().numpy()
        num_clusters = min(20, len(population) // 25)
        
        kmeans = KMeans(n_clusters=num_clusters, n_init=1)
        clusters = kmeans.fit_predict(flattened_pop)
        
        # Replace worst solutions in each cluster with new random solutions
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_fitness = fitness_scores[cluster_indices]
                worst_idx = cluster_indices[torch.argmin(cluster_fitness)]
                population[worst_idx] = self._create_intelligent_schedule()
        
        return population

    def evolve(self) -> Tuple[torch.Tensor, float, Dict[str, int]]:
        population = self.initialize_population()
        generations_without_improvement = 0
        population_refreshes = 0
        self.best_violations = None
        
        print("Starting evolution...")
        try:
            for generation in range(self.config.num_generations):
                if generation % 5 == 0:
                    print(f"Generation {generation}/{self.config.num_generations}")
                
                # Process population in batches
                all_fitness_scores = []
                all_violations = []
                
                for i in range(0, len(population), self.config.batch_size):
                    batch = population[i:i + self.config.batch_size]
                    fitness_scores, violations = self.calculate_fitness_batch(batch)
                    all_fitness_scores.append(fitness_scores)
                    all_violations.append(violations)
                
                fitness_scores = torch.cat(all_fitness_scores)
                
                # Update best solution
                best_idx = torch.argmax(fitness_scores)
                current_best_fitness = fitness_scores[best_idx].item()
                
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_solution = population[best_idx].clone()
                    self.best_violations = all_violations[best_idx // self.config.batch_size]
                    generations_without_improvement = 0
                    print(f"New best fitness: {self.best_fitness:.4f}")
                else:
                    generations_without_improvement += 1
                
                self.fitness_history.append(self.best_fitness)
                
                # Check stopping conditions
                if self.best_fitness >= self.config.target_fitness:
                    print("Target fitness reached!")
                    break
                
                if generations_without_improvement >= 20:
                    if population_refreshes < 3:
                        print("Refreshing population...")
                        population = self._maintain_diversity(population, fitness_scores)
                        generations_without_improvement = 0
                        population_refreshes += 1
                    else:
                        print("No improvement for 20 generations. Stopping early.")
                        break
                
                # Create new population
                new_population = []
                
                # Elitism
                elite_indices = torch.argsort(fitness_scores, descending=True)[:self.config.elite_size]
                new_population.extend([population[idx].clone() for idx in elite_indices])
                
                # Breeding
                while len(new_population) < self.config.population_size:
                    parent1 = self._tournament_select(population, fitness_scores)
                    parent2 = self._tournament_select(population, fitness_scores)
                    
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    if random.random() < self.config.mutation_rate:
                        child1 = self._mutate(child1)
                    if random.random() < self.config.mutation_rate:
                        child2 = self._mutate(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < self.config.population_size:
                        new_population.append(child2)
                
                population = torch.stack(new_population[:self.config.population_size])
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Saving best solution found so far...")
        
        return self.best_solution, self.best_fitness, self.best_violations

    def export_results(self, output_file: str):
        if self.best_solution is None:
            raise ValueError("No solution found yet. Run evolve() first.")
        
        solution = self.best_solution.cpu().numpy()
        schedule_data = []
        total_capacity_violations = 0
        courses_over_capacity = 0
        
        for course_idx, (time_slot_idx, classroom_idx, instructor_idx) in enumerate(solution):
            time_slot = self.problem.time_slots[int(time_slot_idx)]
            classroom = self.problem.classrooms[int(classroom_idx)]
            instructor = self.problem.instructors[int(instructor_idx)]
            course = self.problem.courses[course_idx]
            enrollment = int(self.course_enrollments[course_idx].item())
            
            capacity_violation = max(0, enrollment - classroom["Capacity"])
            if capacity_violation > 0:
                courses_over_capacity += 1
                total_capacity_violations += capacity_violation
            
            schedule_data.append({
                "Course": course["Name"],
                "Course ID": course["ID"],
                "Day": time_slot.day,
                "Start Time": f"{time_slot.start_time:.2f}",
                "End Time": f"{time_slot.end_time:.2f}",
                "Classroom": classroom["Name"],
                "Instructor": instructor["Name"],
                "Enrollment": enrollment,
                "Room Capacity": classroom["Capacity"],
                "Capacity Violation": capacity_violation
            })
        
        # Create summary metrics
        metrics_data = [{
            "Metric": "Final Fitness Score",
            "Value": f"{self.best_fitness:.4f}"
        }]
        
        # Add violations with clearer labeling
        _, violations = self.calculate_fitness_batch(self.best_solution.unsqueeze(0))
        metrics_data.extend([
            {
                "Metric": "Courses Exceeding Capacity",
                "Value": str(courses_over_capacity)
            },
            {
                "Metric": "Total Students Over Capacity",
                "Value": str(total_capacity_violations)
            },
            {
                "Metric": "Classroom Type Mismatches",
                "Value": str(violations["Classroom Compatibility"])
            },
            {
                "Metric": "Instructor Time Conflicts",
                "Value": str(violations["Instructor Conflicts"])
            },
            {
                "Metric": "Students Affected by Time Conflicts",
                "Value": str(violations["Students with Conflicts"])
            }
        ])
        
        # Export to Excel
        with pd.ExcelWriter(output_file) as writer:
            # Export schedule
            schedule_df = pd.DataFrame(schedule_data)
            schedule_df['Start Time'] = pd.to_numeric(schedule_df['Start Time'])
            schedule_df = schedule_df.sort_values(['Day', 'Start Time'])
            schedule_df.to_excel(writer, sheet_name="Schedule", index=False)
            
            # Export metrics
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Metrics", index=False)

def main():
    input_file = "school_input_data.xlsx"
    print(f"Loading problem data from {input_file}...")
    
    problem = SchedulingProblem(input_file)
    config = create_adaptive_config(problem)
    
    print("\nProblem Size:")
    print(f"Number of Courses: {len(problem.courses)}")
    print(f"Number of Classrooms: {len(problem.classrooms)}")
    print(f"Number of Instructors: {len(problem.instructors)}")
    print(f"Number of Students: {len(problem.students)}")
    
    print("\nConfiguration:")
    print(f"Population Size: {config.population_size}")
    print(f"Number of Generations: {config.num_generations}")
    print(f"Mutation Rate: {config.mutation_rate:.2f}")
    print(f"Elite Size: {config.elite_size}")
    print(f"Using Device: {config.device}")
    
    scheduler = GeneticScheduler(problem, config)
    
    print("\nStarting optimization...")
    start_time = datetime.now()
    best_solution, best_fitness, violations = scheduler.evolve()
    end_time = datetime.now()
    
    print(f"\nOptimization completed in: {end_time - start_time}")
    print(f"Best fitness achieved: {best_fitness:.4f}")
    
    output_file = "schedule_output.xlsx"
    scheduler.export_results(output_file)
    print(f"\nResults exported to: {output_file}")

if __name__ == "__main__":
    main()