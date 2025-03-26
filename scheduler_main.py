import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import random
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from sklearn.cluster import KMeans
import warnings
import os
import gc
from functools import lru_cache

warnings.filterwarnings('ignore')

# Enable faster PyTorch operations
torch.backends.cudnn.benchmark = True

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
    use_mixed_precision: bool = True
    sample_size: int = 200  # For statistical sampling of constraints

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
        
        # Course enrollments - optimized sparse matrix creation
        course_name_to_idx = {course['Name']: i for i, course in enumerate(self.courses)}
        enrollment_indices = []
        
        # Process in batches to reduce memory pressure
        batch_size = 1000
        for student_idx_batch_start in range(0, len(self.students), batch_size):
            student_idx_batch_end = min(student_idx_batch_start + batch_size, len(self.students))
            for student_idx in range(student_idx_batch_start, student_idx_batch_end):
                student = self.students[student_idx]
                courses = student['Courses'].split(', ') if isinstance(student['Courses'], str) else []
                for course_name in courses:
                    if course_name in course_name_to_idx:
                        course_idx = course_name_to_idx[course_name]
                        enrollment_indices.append([course_idx, student_idx])
        
        if enrollment_indices:
            indices = torch.tensor(enrollment_indices, dtype=torch.long).t()
            values = torch.ones(len(enrollment_indices), dtype=torch.float16)  # Use float16 to save memory
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
            
        # Use float16 for all tensors to save memory
        
        # Course sizes - with fast vectorized calculations
        # Pre-compute a mapping of course names to sizes
        course_sizes_dict = {}
        for course_idx, course in enumerate(self.courses):
            course_name = course['Name']
            course_sizes_dict[course_name] = 0
            
        # Count enrollments
        for student in self.students:
            courses = student['Courses'].split(', ') if isinstance(student['Courses'], str) else []
            for course_name in courses:
                if course_name in course_sizes_dict:
                    course_sizes_dict[course_name] += 1
                    
        # Convert to tensor
        course_sizes = [course_sizes_dict[course['Name']] for course in self.courses]
        self.course_enrollments = torch.tensor(course_sizes, dtype=torch.float16, device=device)
        
        # Classroom capacities
        self.classroom_capacities = torch.tensor(
            [c['Capacity'] for c in self.classrooms],
            dtype=torch.float16,
            device=device
        )
        
        # Classroom compatibility - optimized with boolean tensor
        compatibility_matrix = torch.zeros(
            (len(self.courses), len(self.classrooms)),
            dtype=torch.bool,
            device=device
        )
        
        # Pre-compute unique classroom types
        classroom_types = {classroom['Type'] for classroom in self.classrooms}
        type_to_classrooms = {t: [] for t in classroom_types}
        
        for idx, classroom in enumerate(self.classrooms):
            type_to_classrooms[classroom['Type']].append(idx)
            
        # Assign compatibility more efficiently
        for course_idx, course in enumerate(self.courses):
            required_type = course['Classroom Type']
            for classroom_idx in type_to_classrooms.get(required_type, []):
                compatibility_matrix[course_idx, classroom_idx] = True
                
        self.classroom_compatibility = compatibility_matrix
        
        # Instructor availability - optimize with boolean operations
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
        
        # Add caching for frequent operations
        self._cached_day_groups = {}

    def get_day_groups(self, time_slots_tensor):
        """Cache day groups calculation for time slots"""
        key = hash(str(time_slots_tensor.tolist()))
        if key not in self._cached_day_groups:
            days = time_slots_tensor // 10
            self._cached_day_groups[key] = days
        return self._cached_day_groups[key]

def create_adaptive_config(problem: SchedulingProblem) -> SchedulingConfig:
    num_courses = len(problem.courses)
    
    # Scale configuration based on problem size
    population_size = min(800, max(100, int(num_courses * 2)))
    num_generations = min(1000, max(200, int(num_courses * 3)))
    
    # Adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_mem < 4:  # Small GPU
            batch_size = min(32, population_size)
        elif gpu_mem < 8:  # Medium GPU
            batch_size = min(64, population_size)
        else:  # Large GPU
            batch_size = min(128, population_size)
    else:
        batch_size = min(32, population_size)
    
    # Adjust sampling for large problems
    sample_size = min(500, max(50, int(len(problem.students) * 0.05)))
    
    return SchedulingConfig(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=0.6,
        crossover_rate=0.85,
        elite_size=int(population_size * 0.15),
        tournament_size=int(population_size * 0.1),
        target_fitness=0.90,
        batch_size=batch_size,
        num_phases=3,
        sample_size=sample_size
    )

class GeneticScheduler:
    def __init__(self, problem: SchedulingProblem, config: SchedulingConfig):
        self.problem = problem
        self.config = config
        self.device = torch.device(config.device)
        self.best_solution = None
        self.best_fitness = 0.0
        self.fitness_history = []
        self.scaler = GradScaler() if config.use_mixed_precision and self.device.type == 'cuda' else None
        
        # Move data to device and convert to appropriate precision
        self.course_enrollments = problem.course_enrollments.to(self.device)
        self.classroom_capacities = problem.classroom_capacities.to(self.device)
        self.classroom_compatibility = problem.classroom_compatibility.to(self.device)
        self.instructor_availability = problem.instructor_availability.to(self.device)
        self.student_course_matrix = problem.student_course_matrix.to(self.device)
        
        # Precalculate common values
        self.num_courses = len(problem.courses)
        self.num_classrooms = len(problem.classrooms)
        self.num_instructors = len(problem.instructors)
        self.num_time_slots = len(problem.time_slots)
        
        # Cache for conflict checking
        self.conflict_cache = {}

    def initialize_population(self) -> torch.Tensor:
        population = torch.zeros(
            (self.config.population_size, self.num_courses, 3),
            device=self.device
        )
        
        # Initialize with semi-intelligent assignments, parallelized across the population
        for i in range(self.config.population_size):
            schedule = self._create_intelligent_schedule()
            population[i] = schedule
            
            # Clear CUDA cache periodically to prevent memory buildup
            if i % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return population

    def _create_intelligent_schedule(self) -> torch.Tensor:
        schedule = torch.zeros(self.num_courses, 3, device=self.device)
        
        # Sort courses by enrollment size (descending)
        course_sizes = self.course_enrollments.cpu().numpy()
        course_order = np.argsort(-course_sizes)
        
        for course_idx in course_order:
            # Find compatible classrooms
            compatible_rooms = torch.where(self.classroom_compatibility[course_idx])[0]
            if len(compatible_rooms) == 0:
                # Fallback to random assignment if no compatible rooms
                schedule[course_idx, 1] = random.randint(0, self.num_classrooms-1)
            else:
                # Choose random compatible room with preference for size-appropriate rooms
                course_size = self.course_enrollments[course_idx].item()
                
                # Check if we can find a room with appropriate capacity
                suitable_rooms = []
                for room_idx in compatible_rooms.cpu().numpy():
                    room_capacity = self.classroom_capacities[room_idx].item()
                    if room_capacity >= course_size:
                        suitable_rooms.append(room_idx)
                
                if suitable_rooms:
                    # Choose smallest room that fits
                    capacities = [self.classroom_capacities[idx].item() for idx in suitable_rooms]
                    best_room = suitable_rooms[np.argmin(capacities)]
                    schedule[course_idx, 1] = best_room
                else:
                    # If no suitable rooms, choose random compatible room
                    schedule[course_idx, 1] = compatible_rooms[
                        random.randint(0, len(compatible_rooms)-1)
                    ]
            
            # Assign instructor that can teach this course
            course_name = self.problem.courses[course_idx]['Name']
            eligible_instructors = []
            
            for i, instructor in enumerate(self.problem.instructors):
                if 'Courses' in instructor and isinstance(instructor['Courses'], str):
                    teachable_courses = instructor['Courses'].split(', ')
                    if course_name in teachable_courses:
                        eligible_instructors.append(i)
            
            if eligible_instructors:
                schedule[course_idx, 2] = random.choice(eligible_instructors)
            else:
                schedule[course_idx, 2] = random.randint(0, self.num_instructors-1)
            
            # Find a time slot with fewer conflicts
            best_slot = 0
            min_conflicts = float('inf')
            
            # Sample a few time slots rather than checking all
            sample_slots = random.sample(range(self.num_time_slots), min(10, self.num_time_slots))
            
            for slot in sample_slots:
                schedule[course_idx, 0] = slot
                conflicts = self._count_local_conflicts(schedule, course_idx)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_slot = slot
            
            schedule[course_idx, 0] = best_slot
        
        return schedule

    @autocast()
    def calculate_fitness_batch(self, schedules: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Calculate fitness scores for a batch of schedules with optimized performance.
        """
        batch_size = len(schedules)
        penalties = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # Split schedules into components and cache them
        time_slots = schedules[:, :, 0].long()
        classrooms = schedules[:, :, 1].long()
        instructors = schedules[:, :, 2].long()
        
        # 1. Classroom capacity violations (optimized counting)
        capacity_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Process in efficient tensor operations where possible
        for b in range(batch_size):
            # Get course enrollments and classroom capacities for this schedule
            b_room_idxs = classrooms[b]
            b_room_capacities = self.classroom_capacities[b_room_idxs]
            
            # Calculate overflow in a vectorized way
            overflow = torch.clamp(self.course_enrollments - b_room_capacities, min=0)
            capacity_violations[b] = overflow.sum().long()
        
        penalties += capacity_violations * 5.0
        
        # 2. Classroom type compatibility (vectorized)
        compatibility_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for b in range(batch_size):
            # Check course-classroom compatibility for all courses at once
            course_indices = torch.arange(self.num_courses, device=self.device)
            room_indices = classrooms[b]
            
            # Count incompatible assignments
            incompatible_mask = ~self.classroom_compatibility[course_indices, room_indices]
            compatibility_violations[b] = incompatible_mask.sum().long()
        
        penalties += compatibility_violations * 10.0
        
        # 3. Instructor conflicts (same instructor, same time slot)
        instructor_conflicts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for b in range(batch_size):
            # Create a tensor to track instructor-timeslot combinations
            instr_time_combinations = torch.zeros(
                (self.num_instructors, self.num_time_slots), 
                dtype=torch.int, 
                device=self.device
            )
            
            # Accumulate counts for each combination
            for c in range(self.num_courses):
                instr = instructors[b, c].item()
                time = time_slots[b, c].item()
                instr_time_combinations[instr, time] += 1
            
            # Count conflicts (combinations used more than once)
            conflicts = torch.sum(torch.clamp(instr_time_combinations - 1, min=0))
            instructor_conflicts[b] = conflicts
        
        penalties += instructor_conflicts * 8.0
        
        # 4. Student conflicts (using statistical sampling for large datasets)
        affected_students = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Get student-course relationships from sparse matrix
        indices = self.student_course_matrix._indices()
        
        # Sample students for efficiency 
        unique_students = torch.unique(indices[1])
        sample_size = min(self.config.sample_size, len(unique_students))
        
        if len(unique_students) > 0:
            sampled_students = unique_students[torch.randperm(len(unique_students))[:sample_size]]
            
            for b in range(batch_size):
                students_with_conflicts = set()
                
                for student_idx in sampled_students:
                    # Get courses for this student
                    student_mask = indices[1] == student_idx
                    student_courses = indices[0][student_mask]
                    
                    if len(student_courses) > 1:  # Only check if enrolled in multiple courses
                        # Get time slots for student's courses
                        course_times = time_slots[b][student_courses]
                        
                        # Get day for each time slot (integer division by 10 to get day index)
                        days = self.problem.get_day_groups(course_times)
                        
                        # Check conflicts within each day
                        for day in torch.unique(days):
                            day_courses = student_courses[days == day]
                            day_times = course_times[days == day]
                            
                            if len(day_times) > 1:
                                # Check for duplicates (same time slot)
                                unique_times = torch.unique(day_times)
                                if len(unique_times) < len(day_times):
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
        
        # Calculate fitness scores (0 to 1) using a more stable calculation
        max_penalties = 25.0 * self.num_courses
        fitness_scores = torch.exp(-penalties / max_penalties)
        
        return torch.clamp(fitness_scores, 0.0, 1.0), total_violations

    def _tournament_select(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        """Optimized tournament selection with tensor operations"""
        tournament_size = min(self.config.tournament_size, len(population))
        idx = torch.randperm(len(population), device=self.device)[:tournament_size]
        tournament_fitness = fitness_scores[idx]
        winner_idx = idx[torch.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized crossover with fewer tensor copies"""
        if random.random() > self.config.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        start, end = crossover_points
        
        # Create children efficiently
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Swap segments
        child1[start:end] = parent2[start:end]
        child2[start:end] = parent1[start:end]
        
        return child1, child2

    def _mutate(self, schedule: torch.Tensor) -> torch.Tensor:
        """More efficient mutation with targeted changes"""
        mutated = schedule.clone()
        
        # Scale mutations with problem size (fewer mutations for large problems)
        mutation_factor = max(0.2, min(1.0, 100 / self.num_courses))
        num_mutations = max(3, int(len(schedule) * self.config.mutation_rate * mutation_factor))
        
        # Select courses to mutate (prefer ones with conflicts)
        course_conflicts = torch.zeros(self.num_courses, device=self.device)
        for i in range(self.num_courses):
            course_conflicts[i] = self._count_local_conflicts(schedule, i)
        
        # Higher probability of selecting courses with more conflicts
        conflict_weights = torch.softmax(course_conflicts, dim=0)
        
        # Convert to CPU for numpy random choice
        weights_cpu = conflict_weights.cpu().numpy()
        indices_to_mutate = np.random.choice(
            self.num_courses, 
            size=min(num_mutations, self.num_courses),
            replace=False, 
            p=weights_cpu / weights_cpu.sum()
        )
        
        for idx in indices_to_mutate:
            mutation_type = random.random()
            
            if mutation_type < 0.5:  # Classroom mutation - prefer appropriate rooms
                course_size = self.course_enrollments[idx].item()
                compatible_rooms = torch.where(self.classroom_compatibility[idx])[0]
                
                if len(compatible_rooms) > 0:
                    # Find rooms with sufficient capacity
                    suitable_rooms = []
                    for room_idx in compatible_rooms.cpu().numpy():
                        if self.classroom_capacities[room_idx].item() >= course_size:
                            suitable_rooms.append(room_idx)
                    
                    if suitable_rooms:
                        # Choose a room with closest capacity match
                        capacities = [self.classroom_capacities[r].item() for r in suitable_rooms]
                        room_idx = suitable_rooms[np.argmin(np.abs(np.array(capacities) - course_size))]
                        mutated[idx, 1] = room_idx
                    else:
                        # If no suitable rooms, choose random compatible
                        mutated[idx, 1] = compatible_rooms[
                            random.randint(0, len(compatible_rooms)-1)
                        ]
            
            elif mutation_type < 0.8:  # Time slot mutation
                # Try to find time slots with fewer conflicts
                best_time_slot = None
                current_conflicts = self._count_local_conflicts(mutated, idx)
                best_conflicts = current_conflicts
                
                # Try several time slots 
                candidate_slots = random.sample(
                    range(self.num_time_slots), 
                    min(5, self.num_time_slots)
                )
                
                for time_slot in candidate_slots:
                    # Save original time slot
                    original_time = mutated[idx, 0].item()
                    
                    # Try new time slot
                    mutated[idx, 0] = time_slot
                    conflicts = self._count_local_conflicts(mutated, idx)
                    
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_time_slot = time_slot
                    
                    # Restore original for next test
                    mutated[idx, 0] = original_time
                
                if best_time_slot is not None and best_conflicts < current_conflicts:
                    mutated[idx, 0] = best_time_slot
            
            else:  # Instructor mutation with availability check
                time_slot = int(mutated[idx, 0].item())
                course_name = self.problem.courses[idx]['Name']
                
                # Find instructors who can teach this course
                eligible_instructors = []
                for i, instructor in enumerate(self.problem.instructors):
                    if ('Courses' in instructor and 
                        isinstance(instructor['Courses'], str) and
                        course_name in instructor['Courses'].split(', ') and
                        self.instructor_availability[i, time_slot]):
                        eligible_instructors.append(i)
                
                if eligible_instructors:
                    mutated[idx, 2] = random.choice(eligible_instructors)
                else:
                    # Fallback to any available instructor
                    available_instructors = torch.where(
                        self.instructor_availability[:, time_slot]
                    )[0].cpu().numpy()
                    
                    if len(available_instructors) > 0:
                        mutated[idx, 2] = random.choice(available_instructors)
        
        return mutated

    def _count_local_conflicts(self, schedule: torch.Tensor, idx: int) -> int:
        """Count conflicts for a specific course in a schedule (with caching)"""
        # Create a cache key
        key = (schedule[idx, 0].item(), schedule[idx, 1].item(), schedule[idx, 2].item(), idx)
        
        if key in self.conflict_cache:
            return self.conflict_cache[key]
        
        conflicts = 0
        time_slot = int(schedule[idx, 0].item())
        instructor = int(schedule[idx, 2].item())
        classroom = int(schedule[idx, 1].item())
        
        # Check instructor conflicts
        for i in range(len(schedule)):
            if i != idx and int(schedule[i, 0].item()) == time_slot and int(schedule[i, 2].item()) == instructor:
                conflicts += 1
        
        # Check classroom conflicts
        for i in range(len(schedule)):
            if i != idx and int(schedule[i, 0].item()) == time_slot and int(schedule[i, 1].item()) == classroom:
                conflicts += 1
        
        # Check classroom compatibility
        if not self.classroom_compatibility[idx, classroom]:
            conflicts += 1
        
        # Capacity violation
        course_size = self.course_enrollments[idx].item()
        room_capacity = self.classroom_capacities[classroom].item()
        if course_size > room_capacity:
            conflicts += 1
        
        # Cache the result
        self.conflict_cache[key] = conflicts
        
        # Limit cache size to prevent memory issues
        if len(self.conflict_cache) > 10000:
            # Clear oldest entries
            for _ in range(1000):
                self.conflict_cache.pop(next(iter(self.conflict_cache)))
        
        return conflicts

    def _local_search(self, solution: torch.Tensor, max_attempts: int = 20) -> torch.Tensor:
        """Optimized local search with more targeted changes"""
        best_solution = solution.clone()
        best_fitness, _ = self.calculate_fitness_batch(best_solution.unsqueeze(0))
        best_fitness = best_fitness.item()
        
        # Find most problematic courses
        course_conflicts = torch.zeros(self.num_courses, device=self.device)
        for i in range(self.num_courses):
            course_conflicts[i] = self._count_local_conflicts(best_solution, i)
        
        # Focus on the courses with the most conflicts
        problematic_courses = torch.argsort(course_conflicts, descending=True)[:max(5, self.num_courses // 10)]
        
        for _ in range(max_attempts):
            # Create a new candidate by mutating problematic courses
            candidate = best_solution.clone()
            
            for course_idx in problematic_courses:
                # Try several options for this course
                best_course_solution = None
                best_course_conflicts = self._count_local_conflicts(candidate, course_idx)
                
                # Try different time slots
                original_time = candidate[course_idx, 0].item()
                original_room = candidate[course_idx, 1].item()
                original_instructor = candidate[course_idx, 2].item()
                
                # Test different time slots
                for time_slot in random.sample(range(self.num_time_slots), min(3, self.num_time_slots)):
                    candidate[course_idx, 0] = time_slot
                    conflicts = self._count_local_conflicts(candidate, course_idx)
                    if conflicts < best_course_conflicts:
                        best_course_conflicts = conflicts
                        best_course_solution = (time_slot, original_room, original_instructor)
                
                # Test different rooms
                candidate[course_idx, 0] = original_time
                compatible_rooms = torch.where(self.classroom_compatibility[course_idx])[0]
                if len(compatible_rooms) > 0:
                    for room_idx in random.sample(compatible_rooms.cpu().numpy().tolist(), 
                                                min(3, len(compatible_rooms))):
                        candidate[course_idx, 1] = room_idx
                        conflicts = self._count_local_conflicts(candidate, course_idx)
                        if conflicts < best_course_conflicts:
                            best_course_conflicts = conflicts
                            best_course_solution = (original_time, room_idx, original_instructor)
                
                # Reset and apply best solution if found
                candidate[course_idx, 0] = original_time
                candidate[course_idx, 1] = original_room
                
                if best_course_solution:
                    candidate[course_idx, 0] = best_course_solution[0]
                    candidate[course_idx, 1] = best_course_solution[1]
                    candidate[course_idx, 2] = best_course_solution[2]
            
            # Evaluate the complete candidate
            fitness, _ = self.calculate_fitness_batch(candidate.unsqueeze(0))
            fitness = fitness.item()
            
            if fitness > best_fitness:
                best_solution = candidate.clone()
                best_fitness = fitness
                
                # Reset conflict cache as we have a new best solution
                self.conflict_cache = {}
        
        return best_solution

    def _maintain_diversity(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        """Improved diversity maintenance with cluster-based approach"""
        # Using a smaller number of clusters for large populations
        num_clusters = min(20, max(3, len(population) // 50))
        
        # Convert to CPU for clustering
        pop_cpu = population.reshape(len(population), -1).cpu().numpy()
        
        # Standardize the data for better clustering
        pop_mean = np.mean(pop_cpu, axis=0)
        pop_std = np.std(pop_cpu, axis=0) + 1e-5  # avoid division by zero
        pop_standardized = (pop_cpu - pop_mean) / pop_std
        
        # Use mini-batch KMeans for efficiency with large populations
        if len(population) > 1000:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100, n_init=1)
        else:
            kmeans = KMeans(n_clusters=num_clusters, n_init=1)
            
        clusters = kmeans.fit_predict(pop_standardized)
        
        # Replace worst solutions in each cluster with new solutions or crossover of best
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 1:
                cluster_fitness = fitness_scores[cluster_indices]
                
                # Find the best solution in this cluster
                best_idx = cluster_indices[torch.argmax(cluster_fitness)]
                best_solution = population[best_idx]
                
                # Replace the worst solutions
                num_to_replace = max(1, len(cluster_indices) // 4)
                worst_indices = [
                    cluster_indices[i] for i in 
                    torch.argsort(cluster_fitness)[:num_to_replace].cpu().numpy()
                ]
                
                for worst_idx in worst_indices:
                    # Replace with new solution or mutated version of best
                    if random.random() < 0.3:
                        population[worst_idx] = self._create_intelligent_schedule()
                    else:
                        # Create a hybrid by mixing with the best solution
                        population[worst_idx] = self._mutate(best_solution)
        
        # Clear any accumulated caches
        self.conflict_cache = {}
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return population

    def evolve(self) -> Tuple[torch.Tensor, float, Dict[str, int]]:
        population = self.initialize_population()
        generations_without_improvement = 0
        population_refreshes = 0
        self.best_violations = None
        self.best_fitness = 0.0
        self.best_solution = None
        
        print("Starting evolution...")
        try:
            for generation in range(self.config.num_generations):
                # Print progress periodically
                if generation % 5 == 0:
                    print(f"Generation {generation}/{self.config.num_generations}", 
                          f"Best fitness: {self.best_fitness:.4f}")
                    
                    if self.device.type == 'cuda':
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        print(f"GPU memory usage: {allocated:.2f} GB")
                        
                        # Force garbage collection and cache clearing
                        if allocated > 3.0:  # If using more than 3GB
                            self.conflict_cache = {}
                            gc.collect()
                            torch.cuda.empty_cache()
                
                # Process population in batches
                all_fitness_scores = []
                all_violations = []
                batch_size = self.config.batch_size
                
                for i in range(0, len(population), batch_size):
                    batch = population[i:i + batch_size]
                    
                    # Calculate fitness with mixed precision if enabled
                    if self.config.use_mixed_precision and self.device.type == 'cuda':
                        with autocast():
                            fitness_scores, violations = self.calculate_fitness_batch(batch)
                    else:
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
                    self.best_violations = all_violations[best_idx // batch_size]
                    generations_without_improvement = 0
                    print(f"New best fitness: {self.best_fitness:.4f}")
                else:
                    generations_without_improvement += 1
                
                self.fitness_history.append(self.best_fitness)
                
                # Check stopping conditions
                if self.best_fitness >= self.config.target_fitness:
                    print("Target fitness reached!")
                    break
                
                # Apply local search to best solutions periodically
                if generations_without_improvement >= 5 and generation % 5 == 0:
                    # Apply local search to top solutions
                    top_indices = torch.argsort(fitness_scores, descending=True)[:5]
                    for idx in top_indices:
                        improved = self._local_search(population[idx], max_attempts=10)
                        improved_fitness, _ = self.calculate_fitness_batch(improved.unsqueeze(0))
                        
                        if improved_fitness.item() > fitness_scores[idx].item():
                            population[idx] = improved
                            fitness_scores[idx] = improved_fitness.item()
                            
                            # Update best solution if improved
                            if improved_fitness.item() > self.best_fitness:
                                self.best_fitness = improved_fitness.item()
                                self.best_solution = improved.clone()
                                generations_without_improvement = 0
                                print(f"New best fitness (local search): {self.best_fitness:.4f}")
                
                # Population refresh if no improvement
                if generations_without_improvement >= 15:
                    if population_refreshes < 3:
                        print("Refreshing population...")
                        population = self._maintain_diversity(population, fitness_scores)
                        generations_without_improvement = 0
                        population_refreshes += 1
                    else:
                        print("No improvement for multiple generations. Stopping early.")
                        break
                
                # Create new population
                new_population = []
                
                # Elitism - copy best solutions directly
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
                
                # Memory management
                if self.device.type == 'cuda' and generation % 10 == 0:
                    torch.cuda.empty_cache()
                    
                # Clear the conflict cache periodically
                if len(self.conflict_cache) > 50000:
                    self.conflict_cache = {}
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Saving best solution found so far...")
        
        # Apply one final local search to the best solution
        if self.best_solution is not None:
            print("Applying final optimization to best solution...")
            self.best_solution = self._local_search(self.best_solution, max_attempts=50)
            final_fitness, final_violations = self.calculate_fitness_batch(self.best_solution.unsqueeze(0))
            self.best_fitness = final_fitness.item()
            self.best_violations = final_violations
            print(f"Final fitness: {self.best_fitness:.4f}")
        
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
    print(f"Using Mixed Precision: {config.use_mixed_precision}")
    
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