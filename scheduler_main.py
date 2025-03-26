import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import random
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Enable faster PyTorch operations when available
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
    sample_size: int = 200

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
        print(f"Loading data from {input_file}...")
        self.load_data(input_file)
        self.initialize_time_slots()
        self.preprocess_data()
        
        # Create lookup tables for faster access
        self._create_lookup_tables()
        print("Problem initialization complete.")
    
    def load_data(self, input_file: str):
        xl = pd.ExcelFile(input_file)
        self.classrooms = pd.read_excel(xl, "Classrooms").to_dict('records')
        self.courses = pd.read_excel(xl, "Courses").to_dict('records')
        self.instructors = pd.read_excel(xl, "Instructors").to_dict('records')
        self.students = pd.read_excel(xl, "Students").to_dict('records')
        print(f"Data loaded: {len(self.courses)} courses, {len(self.classrooms)} classrooms, " 
              f"{len(self.instructors)} instructors, {len(self.students)} students")

    def initialize_time_slots(self):
        print("Initializing time slots...")
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
        print(f"Created {len(self.time_slots)} time slots")

    def _create_lookup_tables(self):
        """Create lookup tables for faster access to relationships"""
        print("Creating lookup tables for faster access...")
        
        # Courses for each student
        self.student_courses = {}
        
        # Students for each course
        self.course_students = {}
        
        # Time slots by day
        self.day_time_slots = {}
        for i, slot in enumerate(self.time_slots):
            if slot.day not in self.day_time_slots:
                self.day_time_slots[slot.day] = []
            self.day_time_slots[slot.day].append(i)
        
        # Compatible rooms for each course
        self.course_compatible_rooms = {}
        
        # Process student enrollment data
        for student_idx, student in enumerate(self.students):
            if isinstance(student['Courses'], str):
                course_names = student['Courses'].split(', ')
                course_indices = []
                
                for course_name in course_names:
                    for course_idx, course in enumerate(self.courses):
                        if course['Name'] == course_name:
                            course_indices.append(course_idx)
                            
                            # Add to course_students
                            if course_idx not in self.course_students:
                                self.course_students[course_idx] = []
                            self.course_students[course_idx].append(student_idx)
                            
                            break
                
                self.student_courses[student_idx] = course_indices
        
        # Process room compatibility
        for course_idx, course in enumerate(self.courses):
            compatible_rooms = []
            course_type = course['Classroom Type']
            
            for room_idx, room in enumerate(self.classrooms):
                if room['Type'] == course_type:
                    compatible_rooms.append(room_idx)
            
            self.course_compatible_rooms[course_idx] = compatible_rooms
        
        print("Lookup tables created.")

    def preprocess_data(self):
        print("Preprocessing data...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Course enrollments
        print("Computing course enrollments...")
        course_sizes = []
        for course_idx, course in enumerate(self.courses):
            course_name = course['Name']
            enrollment = 0
            
            for student in self.students:
                if isinstance(student['Courses'], str) and course_name in student['Courses'].split(', '):
                    enrollment += 1
            
            course_sizes.append(enrollment)
        
        self.course_enrollments = torch.tensor(course_sizes, dtype=torch.float32, device=device)
        
        # Classroom capacities
        print("Processing classroom capacities...")
        self.classroom_capacities = torch.tensor(
            [c['Capacity'] for c in self.classrooms],
            dtype=torch.float32,
            device=device
        )
        
        # Classroom compatibility
        print("Building compatibility matrix...")
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
        print("Processing instructor availability...")
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
        print("Data preprocessing complete.")

def create_adaptive_config(problem: SchedulingProblem) -> SchedulingConfig:
    """Create adaptive configuration based on problem size"""
    print("Creating adaptive configuration...")
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
    
    config = SchedulingConfig(
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
    
    print(f"Created configuration with population_size={config.population_size}, "
          f"num_generations={config.num_generations}, batch_size={config.batch_size}")
    return config

class ReliableScheduler:
    """Reliable genetic algorithm scheduler optimized for performance"""
    def __init__(self, problem: SchedulingProblem, config: SchedulingConfig):
        print("Initializing scheduler...")
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
        
        # Conflict cache
        self.conflict_cache = {}
        print("Scheduler initialized successfully.")

    def initialize_population(self) -> torch.Tensor:
        """Initialize population with semi-intelligent schedules"""
        print(f"Initializing population of {self.config.population_size} schedules...")
        population = torch.zeros(
            (self.config.population_size, len(self.problem.courses), 3),
            device=self.device
        )
        
        # Create schedules
        for i in range(self.config.population_size):
            if i % 10 == 0:
                print(f"Creating schedule {i+1}/{self.config.population_size}")
            try:
                schedule = self._create_intelligent_schedule()
                population[i] = schedule
            except Exception as e:
                print(f"Error creating schedule: {e}")
                # Fallback to random schedule
                population[i] = self._create_random_schedule()
            
            # Clear cache periodically to prevent memory buildup
            if i % 20 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print("Population initialization complete.")
        return population

    def _create_random_schedule(self) -> torch.Tensor:
        """Create a completely random schedule"""
        schedule = torch.zeros(len(self.problem.courses), 3, device=self.device)
        
        # Assign random time slots, rooms, and instructors
        for course_idx in range(len(self.problem.courses)):
            schedule[course_idx, 0] = random.randint(0, len(self.problem.time_slots)-1)
            schedule[course_idx, 1] = random.randint(0, len(self.problem.classrooms)-1)
            schedule[course_idx, 2] = random.randint(0, len(self.problem.instructors)-1)
        
        return schedule

    def _create_intelligent_schedule(self) -> torch.Tensor:
        """Create an intelligent initial schedule"""
        schedule = torch.zeros(len(self.problem.courses), 3, device=self.device)
        
        # Sort courses by enrollment size (descending)
        course_sizes = self.course_enrollments.cpu().numpy()
        course_order = np.argsort(-course_sizes)
        
        for idx, course_idx in enumerate(course_order):
            # Find compatible classrooms
            compatible_rooms = torch.where(self.classroom_compatibility[course_idx])[0]
            if len(compatible_rooms) == 0:
                # Fallback to random assignment if no compatible rooms
                schedule[course_idx, 1] = random.randint(0, len(self.problem.classrooms)-1)
            else:
                # Choose random compatible room
                random_room_idx = random.randint(0, len(compatible_rooms)-1)
                schedule[course_idx, 1] = compatible_rooms[random_room_idx]
            
            # Assign random time slot
            schedule[course_idx, 0] = random.randint(0, len(self.problem.time_slots)-1)
            
            # Assign random instructor
            schedule[course_idx, 2] = random.randint(0, len(self.problem.instructors)-1)
        
        return schedule

    def calculate_fitness_batch(self, schedules: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Calculate fitness scores for a batch of schedules with prioritized constraints"""
        batch_size = len(schedules)
        penalties = torch.zeros(batch_size, device=self.device)
        
        # Split schedules into components
        time_slots = schedules[:, :, 0].long()
        classrooms = schedules[:, :, 1].long()
        instructors = schedules[:, :, 2].long()
        
        # Track hard constraint violations
        instructor_availability_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        room_type_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        room_booking_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        instructor_booking_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Renamed for clarity
        capacity_violations = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # 1. HIGHEST PRIORITY: Instructor availability violations
        for b in range(batch_size):
            for c in range(len(self.problem.courses)):
                instr_idx = int(instructors[b, c].item())
                time_idx = int(time_slots[b, c].item())
                if not self.problem.instructor_availability[instr_idx, time_idx]:
                    instructor_availability_violations[b] += 1
        
        # Apply very high penalty for instructor availability violations
        penalties += instructor_availability_violations * 50.0
        
        # 2. Room type compatibility - high priority
        for b in range(batch_size):
            for c in range(len(self.problem.courses)):
                if not self.classroom_compatibility[c, classrooms[b, c]]:
                    room_type_violations[b] += 1
        
        # Apply high penalty for room type violations
        penalties += room_type_violations * 40.0
        
        # 3. Room booking conflicts (same room, same time) - high priority
        for b in range(batch_size):
            room_schedule = {}
            for c in range(len(self.problem.courses)):
                time = int(time_slots[b, c].item())
                room = int(classrooms[b, c].item())
                
                key = (room, time)
                if key in room_schedule:
                    room_booking_violations[b] += 1
                else:
                    room_schedule[key] = c
        
        # Apply high penalty for room double booking
        penalties += room_booking_violations * 30.0
        
        # 4. Instructor booking conflicts (same instructor, same time)
        for b in range(batch_size):
            instructor_schedule = {}
            for c in range(len(self.problem.courses)):
                time = int(time_slots[b, c].item())
                instr = int(instructors[b, c].item())
                
                key = (instr, time)
                if key in instructor_schedule:
                    instructor_booking_violations[b] += 1
                else:
                    instructor_schedule[key] = c
        
        # Apply penalty for instructor double booking
        penalties += instructor_booking_violations * 20.0
        
        # 5. Classroom capacity violations - medium priority but allow some violations
        over_capacity_rooms = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        total_capacity_violation = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for b in range(batch_size):
            for c in range(len(self.problem.courses)):
                room_idx = classrooms[b, c]
                course_size = self.course_enrollments[c].long()
                room_capacity = self.classroom_capacities[room_idx].long()
                if course_size > room_capacity:
                    overflow = int((course_size - room_capacity).item())
                    total_capacity_violation[b] += overflow
                    over_capacity_rooms[b] += 1
        
        # Calculate allowable violations (5% of classrooms)
        allowed_violations = int(0.05 * len(self.problem.classrooms))
        
        # Only penalize if more than the allowed number of rooms have capacity violations
        for b in range(batch_size):
            if over_capacity_rooms[b] > allowed_violations:
                # Penalize excessive violations
                excess = over_capacity_rooms[b] - allowed_violations
                penalties[b] += excess * 20.0
                
                # Add smaller penalty for the size of the violation
                penalties[b] += total_capacity_violation[b] * 0.2
        
        # 6. LOWEST PRIORITY: Student conflicts
        affected_students = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Sample students for efficiency
        sample_size = min(self.config.sample_size, len(self.problem.students))
        if len(self.problem.students) > 0:
            sampled_students = random.sample(range(len(self.problem.students)), sample_size)
            
            for b in range(batch_size):
                students_with_conflicts = set()
                
                for student_idx in sampled_students:
                    # Get courses for this student
                    if student_idx in self.problem.student_courses:
                        student_courses = self.problem.student_courses[student_idx]
                        
                        # Check all pairs of courses for time conflicts
                        for i in range(len(student_courses)):
                            for j in range(i+1, len(student_courses)):
                                course1 = student_courses[i]
                                course2 = student_courses[j]
                                
                                time1 = int(time_slots[b, course1].item())
                                time2 = int(time_slots[b, course2].item())
                                
                                # Check if same time slot
                                if time1 == time2:
                                    students_with_conflicts.add(student_idx)
                                    break
                            
                            # No need to check more if conflict found
                            if student_idx in students_with_conflicts:
                                break
                
                # Scale count based on sampling
                if sample_size < len(self.problem.students):
                    scale_factor = len(self.problem.students) / sample_size
                    affected_students[b] = int(len(students_with_conflicts) * scale_factor)
                else:
                    affected_students[b] = len(students_with_conflicts)
        
        # Apply lowest penalty for student conflicts
        penalties += affected_students * 1.0
        
        # Calculate total violations for reporting
        total_violations = {
            "Instructor Availability": int(torch.sum(instructor_availability_violations).item()),
            "Classroom Compatibility": int(torch.sum(room_type_violations).item()),
            "Room Conflicts": int(torch.sum(room_booking_violations).item()),
            "Instructor Booking Violations": int(torch.sum(instructor_booking_violations).item()),  # Renamed
            "Classroom Capacity": int(torch.sum(total_capacity_violation).item()),
            "Over Capacity Rooms": int(torch.sum(over_capacity_rooms).item()),
            "Allowed Capacity Violations": allowed_violations,
            "Students with Conflicts": int(torch.sum(affected_students).item())
        }
        
        # Calculate fitness scores (0 to 1)
        max_penalties = 100.0 * len(self.problem.courses)
        fitness_scores = torch.exp(-penalties / max_penalties)
        
        return torch.clamp(fitness_scores, 0.0, 1.0), total_violations

    def _tournament_select(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        """Select a parent using tournament selection"""
        tournament_size = min(self.config.tournament_size, len(population))
        idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = fitness_scores[idx]
        winner_idx = idx[torch.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform crossover between parents"""
        if random.random() > self.config.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Swap segments between crossover points
        child1[crossover_points[0]:crossover_points[1]] = parent2[crossover_points[0]:crossover_points[1]]
        child2[crossover_points[0]:crossover_points[1]] = parent1[crossover_points[0]:crossover_points[1]]
        
        return child1, child2

    def _mutate(self, schedule: torch.Tensor) -> torch.Tensor:
        """Mutate a schedule with enhanced mutation strategies and constraint priorities"""
        mutated = schedule.clone()
        
        # Adjust number of mutations based on schedule size
        num_mutations = max(3, int(len(schedule) * self.config.mutation_rate))
        
        # First, check for hard constraint violations - instructor availability, room type, double booking
        hard_violations = self._find_hard_constraint_violations(schedule)
        
        # If hard violations exist, focus mutations on fixing those courses
        if hard_violations and len(hard_violations) > 0:
            # Focus 80% of mutations on hard constraint violations
            hard_violation_count = min(int(num_mutations * 0.8), len(hard_violations))
            
            # Select courses with hard violations for mutation
            hard_courses_to_mutate = random.sample(list(hard_violations.keys()), hard_violation_count)
            
            # Fill remaining mutations with random courses
            remaining = num_mutations - hard_violation_count
            all_courses = set(range(len(schedule)))
            available = list(all_courses - set(hard_courses_to_mutate))
            
            if remaining > 0 and available:
                hard_courses_to_mutate.extend(random.sample(available, min(remaining, len(available))))
                
            courses_to_mutate = hard_courses_to_mutate
        else:
            # If no hard violations, use standard mutation
            courses_to_mutate = random.sample(range(len(schedule)), num_mutations)
        
        # Apply mutations
        for idx in courses_to_mutate:
            # Check what constraints this course violates
            violations = self._check_course_violations(schedule, idx)
            
            # Different mutation strategies based on constraints violated
            if "instructor_availability" in violations:
                # Priority 1: Fix instructor availability by assigning available instructor
                self._fix_instructor_availability(mutated, idx)
                
            elif "room_type" in violations:
                # Priority 2: Fix room type compatibility
                self._fix_room_compatibility(mutated, idx)
                
            elif "room_booking" in violations:
                # Priority 3: Fix room double booking by changing time or room
                if random.random() < 0.7:
                    # Usually better to change time slot
                    self._fix_time_slot_conflicts(mutated, idx)
                else:
                    # Sometimes change room instead
                    self._fix_room_conflicts(mutated, idx)
                
            else:
                # No hard violations, use standard mutation
                mutation_type = random.random()
                
                if mutation_type < 0.4:  # Time slot mutation
                    self._fix_time_slot_conflicts(mutated, idx)
                    
                elif mutation_type < 0.8:  # Classroom mutation
                    self._fix_room_compatibility(mutated, idx, optimize_capacity=True)
                    
                else:  # Instructor mutation
                    # Make sure new instructor is available
                    self._fix_instructor_availability(mutated, idx)
        
        return mutated
        
    def _check_course_violations(self, schedule: torch.Tensor, idx: int) -> Set[str]:
        """Check what constraints a specific course violates"""
        violations = set()
        
        # Extract course details
        time_idx = int(schedule[idx, 0].item())
        room_idx = int(schedule[idx, 1].item())
        instr_idx = int(schedule[idx, 2].item())
        
        # 1. Check instructor availability
        if not self.problem.instructor_availability[instr_idx, time_idx]:
            violations.add("instructor_availability")
        
        # 2. Check room type compatibility
        if not self.classroom_compatibility[idx, room_idx]:
            violations.add("room_type")
        
        # 3. Check for room double booking
        for c_idx in range(len(schedule)):
            if c_idx != idx:
                other_time = int(schedule[c_idx, 0].item())
                other_room = int(schedule[c_idx, 1].item())
                
                if time_idx == other_time and room_idx == other_room:
                    violations.add("room_booking")
                    break
        
        # 4. Check for instructor double booking
        for c_idx in range(len(schedule)):
            if c_idx != idx:
                other_time = int(schedule[c_idx, 0].item())
                other_instr = int(schedule[c_idx, 2].item())
                
                if time_idx == other_time and instr_idx == other_instr:
                    violations.add("instructor_booking")
                    break
        
        # 5. Check for capacity violations
        course_size = self.course_enrollments[idx].item()
        room_capacity = self.classroom_capacities[room_idx].item()
        
        if course_size > room_capacity:
            violations.add("capacity")
        
        return violations
        
    def _find_hard_constraint_violations(self, schedule: torch.Tensor) -> Dict[int, Set[str]]:
        """Find courses with hard constraint violations (availability, type, double booking)"""
        hard_violations = {}
        
        for c_idx in range(len(schedule)):
            violations = self._check_course_violations(schedule, c_idx)
            
            # Only consider hard constraints
            hard_constraints = {"instructor_availability", "room_type", "room_booking", "instructor_booking"}
            course_hard_violations = violations.intersection(hard_constraints)
            
            if course_hard_violations:
                hard_violations[c_idx] = course_hard_violations
        
        return hard_violations
        
    def _fix_instructor_availability(self, schedule: torch.Tensor, idx: int):
        """Fix instructor availability violation by assigning available instructor"""
        time_idx = int(schedule[idx, 0].item())
        
        # Find instructors available at this time
        available_instructors = torch.where(self.problem.instructor_availability[:, time_idx])[0].cpu().numpy()
        
        if len(available_instructors) > 0:
            # Choose random available instructor
            schedule[idx, 2] = available_instructors[random.randint(0, len(available_instructors)-1)]
        else:
            # If no instructors available at this time, change the time instead
            self._fix_time_slot_conflicts(schedule, idx, check_instructor=True)
    
    def _fix_room_compatibility(self, schedule: torch.Tensor, idx: int, optimize_capacity=False):
        """Fix room type compatibility violation"""
        # Find compatible rooms
        compatible_rooms = torch.where(self.classroom_compatibility[idx])[0].cpu().numpy()
        
        if len(compatible_rooms) > 0:
            if optimize_capacity:
                # Find room with appropriate capacity
                course_size = self.course_enrollments[idx].item()
                suitable_rooms = []
                
                for room_idx in compatible_rooms:
                    capacity = self.classroom_capacities[room_idx].item()
                    if capacity >= course_size:
                        suitable_rooms.append((room_idx, capacity))
                
                if suitable_rooms:
                    # Choose room with closest capacity
                    suitable_rooms.sort(key=lambda x: x[1] - course_size)
                    schedule[idx, 1] = suitable_rooms[0][0]
                else:
                    # If no suitable room, choose largest compatible room
                    room_capacities = [(r, self.classroom_capacities[r].item()) for r in compatible_rooms]
                    room_capacities.sort(key=lambda x: x[1], reverse=True)
                    schedule[idx, 1] = room_capacities[0][0]
            else:
                # Just choose any compatible room
                schedule[idx, 1] = compatible_rooms[random.randint(0, len(compatible_rooms)-1)]
        else:
            # If no compatible rooms (should not happen), assign random room
            schedule[idx, 1] = random.randint(0, len(self.problem.classrooms) - 1)
    
    def _fix_time_slot_conflicts(self, schedule: torch.Tensor, idx: int, check_instructor=False):
        """Fix time slot conflicts by finding a better time slot"""
        room_idx = int(schedule[idx, 1].item())
        instr_idx = int(schedule[idx, 2].item())
        
        # Find all time slots where the instructor is available
        if check_instructor:
            available_slots = torch.where(self.problem.instructor_availability[instr_idx])[0].cpu().numpy()
        else:
            available_slots = list(range(len(self.problem.time_slots)))
        
        if not available_slots:
            # If no slots available, use a random slot
            schedule[idx, 0] = random.randint(0, len(self.problem.time_slots) - 1)
            return
        
        # Shuffle available slots for randomness
        random.shuffle(available_slots)
        
        # Try each available slot to find one with no conflicts
        for slot_idx in available_slots[:10]:  # Limit to 10 slots for efficiency
            # Check if this slot causes room conflicts
            has_conflict = False
            
            for c_idx in range(len(schedule)):
                if c_idx != idx:
                    other_time = int(schedule[c_idx, 0].item())
                    other_room = int(schedule[c_idx, 1].item())
                    other_instr = int(schedule[c_idx, 2].item())
                    
                    if slot_idx == other_time:
                        # Check room conflict
                        if room_idx == other_room:
                            has_conflict = True
                            break
                        
                        # Check instructor conflict
                        if instr_idx == other_instr:
                            has_conflict = True
                            break
            
            if not has_conflict:
                # Found a good slot
                schedule[idx, 0] = slot_idx
                return
        
        # If all slots have conflicts, choose a random slot
        schedule[idx, 0] = random.choice(available_slots)
    
    def _fix_room_conflicts(self, schedule: torch.Tensor, idx: int):
        """Fix room conflicts by assigning a different room"""
        time_idx = int(schedule[idx, 0].item())
        
        # Find all rooms used at this time
        used_rooms = set()
        for c_idx in range(len(schedule)):
            if c_idx != idx and int(schedule[c_idx, 0].item()) == time_idx:
                used_rooms.add(int(schedule[c_idx, 1].item()))
        
        # Find compatible rooms that are not used
        compatible_rooms = torch.where(self.classroom_compatibility[idx])[0].cpu().numpy()
        available_rooms = [r for r in compatible_rooms if r not in used_rooms]
        
        if available_rooms:
            # Use a compatible and available room
            course_size = self.course_enrollments[idx].item()
            suitable_rooms = []
            
            for room_idx in available_rooms:
                capacity = self.classroom_capacities[room_idx].item()
                if capacity >= course_size:
                    suitable_rooms.append((room_idx, capacity))
            
            if suitable_rooms:
                # Choose room with best capacity fit
                suitable_rooms.sort(key=lambda x: x[1] - course_size)
                schedule[idx, 1] = suitable_rooms[0][0]
            else:
                # If no suitable room, choose any available room
                schedule[idx, 1] = available_rooms[random.randint(0, len(available_rooms)-1)]
        else:
            # If no compatible rooms available, change the time instead
            self._fix_time_slot_conflicts(schedule, idx)

    def _maintain_diversity(self, population: torch.Tensor, fitness_scores: torch.Tensor) -> torch.Tensor:
        """Maintain diversity by replacing worst solutions"""
        # Find lowest fitness individuals
        num_to_replace = max(5, len(population) // 10)
        worst_indices = torch.argsort(fitness_scores)[:num_to_replace]
        
        # Replace with new individuals
        for idx in worst_indices:
            # 50% chance of completely new schedule, 50% chance of mutation from best
            if random.random() < 0.5:
                population[idx] = self._create_intelligent_schedule()
            else:
                # Get a good individual to mutate
                best_idx = torch.argsort(fitness_scores, descending=True)[0]
                population[idx] = self._mutate(population[best_idx])
        
        return population

    def _local_search(self, solution: torch.Tensor, max_attempts: int = 20) -> torch.Tensor:
        """Perform local search to improve a solution"""
        best_solution = solution.clone()
        best_fitness, _ = self.calculate_fitness_batch(best_solution.unsqueeze(0))
        best_fitness = best_fitness.item()
        original_fitness = best_fitness
        
        for _ in range(max_attempts):
            candidate = self._mutate(best_solution)
            fitness, _ = self.calculate_fitness_batch(candidate.unsqueeze(0))
            fitness = fitness.item()
            
            if fitness > best_fitness:
                best_solution = candidate
                best_fitness = fitness
        
        # Only return the improved solution if it's actually better
        if best_fitness > original_fitness:
            print(f"Local search improved fitness: {original_fitness:.4f} -> {best_fitness:.4f}")
            return best_solution
        else:
            print("Local search did not find improvement, keeping original solution")
            return solution  # Return original if no improvement found

    def evolve(self) -> Tuple[torch.Tensor, float, Dict[str, int]]:
        """Evolve a population to find an optimal schedule"""
        print("Starting evolution process...")
        start_time = time.time()
        
        try:
            # Initialize population
            population = self.initialize_population()
            generations_without_improvement = 0
            population_refreshes = 0
            best_violations = None
            
            for generation in range(self.config.num_generations):
                if generation % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"Generation {generation}/{self.config.num_generations} "
                          f"(Elapsed: {elapsed:.1f}s)")
                
                # Process population in batches
                all_fitness_scores = []
                all_violations = []
                
                batch_size = self.config.batch_size
                for i in range(0, len(population), batch_size):
                    batch = population[i:i+batch_size]
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
                    batch_idx = best_idx // batch_size
                    if batch_idx < len(all_violations):
                        best_violations = all_violations[batch_idx]
                    generations_without_improvement = 0
                    print(f"New best fitness: {self.best_fitness:.4f}")
                else:
                    generations_without_improvement += 1
                
                self.fitness_history.append(self.best_fitness)
                
                # Check stopping conditions
                if self.best_fitness >= self.config.target_fitness:
                    print("Target fitness reached!")
                    break
                
                # Apply local search to best solution occasionally
                if generation % 10 == 0 and self.best_solution is not None:
                    improved = self._local_search(self.best_solution)
                    improved_fitness, _ = self.calculate_fitness_batch(improved.unsqueeze(0))
                    improved_fitness = improved_fitness.item()
                    
                    if improved_fitness > self.best_fitness:
                        self.best_solution = improved
                        self.best_fitness = improved_fitness
                        print(f"Improved with local search: {self.best_fitness:.4f}")
                        generations_without_improvement = 0
                
                # Refresh population if stagnant
                if generations_without_improvement >= 15:
                    if population_refreshes < 3:
                        print("Refreshing population...")
                        population = self._maintain_diversity(population, fitness_scores)
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
                    parent1 = self._tournament_select(population, fitness_scores)
                    parent2 = self._tournament_select(population, fitness_scores)
                    
                    # Create offspring
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # Mutation
                    if random.random() < self.config.mutation_rate:
                        child1 = self._mutate(child1)
                    if random.random() < self.config.mutation_rate:
                        child2 = self._mutate(child2)
                    
                    # Add to new population
                    new_population.append(child1)
                    if len(new_population) < self.config.population_size:
                        new_population.append(child2)
                
                # Update population
                population = torch.stack(new_population[:self.config.population_size])
                
                # Memory management
                if generation % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Final local search
            if self.best_solution is not None:
                print("Performing final optimization...")
                # Save original solution in case local search doesn't help
                original_solution = self.best_solution.clone()
                original_fitness = self.best_fitness
                
                # Try more aggressive local search
                improved_solution = self._local_search(self.best_solution, max_attempts=100)
                improved_fitness, improved_violations = self.calculate_fitness_batch(improved_solution.unsqueeze(0))
                improved_fitness = improved_fitness.item()
                
                # Only update if actually improved
                if improved_fitness > original_fitness:
                    self.best_solution = improved_solution
                    self.best_fitness = improved_fitness
                    best_violations = improved_violations
                    print(f"Final optimization improved fitness to {self.best_fitness:.4f}")
                else:
                    # Keep original solution
                    self.best_solution = original_solution
                    # Recompute violations to ensure consistency
                    _, best_violations = self.calculate_fitness_batch(self.best_solution.unsqueeze(0))
            
            elapsed = time.time() - start_time
            print(f"Evolution completed in {elapsed:.1f}s with fitness {self.best_fitness:.4f}")
            
            return self.best_solution, self.best_fitness, best_violations
            
        except KeyboardInterrupt:
            print("\nEvolution interrupted by user. Returning best solution found so far.")
            return self.best_solution, self.best_fitness, best_violations
        except Exception as e:
            print(f"Error during evolution: {e}")
            if self.best_solution is not None:
                return self.best_solution, self.best_fitness, best_violations
            else:
                # Return a random solution if no solution found yet
                random_solution = self._create_random_schedule().unsqueeze(0)
                fitness, violations = self.calculate_fitness_batch(random_solution)
                return random_solution[0], fitness.item(), violations

    def export_results(self, output_file: str):
        """Export the best solution to an Excel file"""
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
        
        # Get violations
        _, violations = self.calculate_fitness_batch(self.best_solution.unsqueeze(0))
        
        # Add violations with clearer labeling
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
                "Value": str(violations.get("Instructor Booking Violations", 0))
            },
            {
                "Metric": "Room Double Bookings",
                "Value": str(violations["Room Conflicts"])
            },
            {
                "Metric": "Students Affected by Time Conflicts",
                "Value": str(violations["Students with Conflicts"])
            }
        ])
        
        # Export to Excel
        print(f"Exporting results to {output_file}...")
        with pd.ExcelWriter(output_file) as writer:
            # Export schedule
            schedule_df = pd.DataFrame(schedule_data)
            schedule_df['Start Time'] = pd.to_numeric(schedule_df['Start Time'])
            schedule_df = schedule_df.sort_values(['Day', 'Start Time'])
            schedule_df.to_excel(writer, sheet_name="Schedule", index=False)
            
            # Export metrics
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Metrics", index=False)
        
        print("Export complete.")

def main():
    """Main function to run the scheduler"""
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
    
    scheduler = ReliableScheduler(problem, config)
    
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