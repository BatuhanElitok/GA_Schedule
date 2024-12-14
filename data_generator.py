import pandas as pd
import random
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SchoolDataConfig:
    """
    Configuration parameters for generating sample school data.
    """
    num_classrooms: int
    num_courses: int
    num_instructors: int
    num_students: int
    time_slots: List[float] = None

    def __post_init__(self):
        if self.time_slots is None:
            self.time_slots = [8.40, 9.40, 10.40, 11.40, 12.40, 13.40, 14.40, 15.40, 16.40]

class SchoolDataGenerator:
    """
    Generates sample school data based on the provided configuration.
    """
    def __init__(self, config: SchoolDataConfig):
        self.config = config

    def generate_classrooms(self) -> List[Dict]:
        """
        Generates a list of classrooms with randomly assigned names, capacities, and room types.
        """
        classrooms = []
        for i in range(self.config.num_classrooms):
            name = f"Classroom_{chr(65 + i)}"
            capacity = random.randint(30, 100)
            room_type = random.choice(["Regular", "Lab"])
            classrooms.append({"Name": name, "Capacity": capacity, "Type": room_type})
        return classrooms

    def generate_courses(self) -> List[Dict]:
        """
        Generates a list of courses with randomly assigned IDs, names, and classroom types.
        """
        courses = []
        for i in range(1, self.config.num_courses + 1):
            course_id = f"C{i:03}"
            course_name = f"Course_{i}"
            classroom_type = random.choice(["Regular", "Lab"])
            courses.append({"ID": course_id, "Name": course_name, "Classroom Type": classroom_type})
        return courses

    def _generate_availability_slots(self) -> Dict[str, str]:
        """
        Generates a dictionary of available time slots for each day of the week.
        """
        slots = {}
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            available_periods = []
            used_slots = set()

            # Generate at least one block of three consecutive hours
            start_idx = random.randint(0, len(self.config.time_slots) - 3)
            end_idx = start_idx + 3
            start_time = self.config.time_slots[start_idx]
            end_time = self.config.time_slots[end_idx - 1] + 1.0
            used_slots.update(range(start_idx, end_idx))
            available_periods.append(f"{start_time:.2f}-{end_time:.2f}")

            # Optionally add one more non-overlapping block
            if random.choice([True, False]):
                while True:
                    start_idx = random.randint(0, len(self.config.time_slots) - 3)
                    end_idx = start_idx + 3
                    if not used_slots.intersection(range(start_idx, end_idx)):
                        start_time = self.config.time_slots[start_idx]
                        end_time = self.config.time_slots[end_idx - 1] + 1.0
                        available_periods.append(f"{start_time:.2f}-{end_time:.2f}")
                        break

            slots[day + " Slots"] = " | ".join(available_periods)
        return slots

    def generate_instructors(self, courses: List[Dict]) -> List[Dict]:
        """
        Generates a list of instructors with randomly assigned names, teachable courses, and available time slots.
        """
        instructors = []
        for i in range(1, self.config.num_instructors + 1):
            name = f"Instructor_{i}"
            teachable_courses = random.sample([course['Name'] for course in courses], 
                                           random.randint(1, 5))
            slots = self._generate_availability_slots()
            instructor = {"Name": name, "Courses": ", ".join(teachable_courses), **slots}
            instructors.append(instructor)
        return instructors

    def generate_students(self, courses: List[Dict]) -> List[Dict]:
        """
        Generates a list of students with randomly assigned names and enrolled courses.
        """
        students = []
        for i in range(1, self.config.num_students + 1):
            name = f"student_{i}"
            enrolled_courses = random.sample([course['Name'] for course in courses], 
                                          random.randint(1, 7))
            students.append({"Name": name, "Courses": ", ".join(enrolled_courses)})
        return students

    def generate_all_data(self) -> Dict:
        """
        Generates all the school data entities and returns them as a dictionary.
        """
        classrooms = self.generate_classrooms()
        courses = self.generate_courses()
        instructors = self.generate_instructors(courses)
        students = self.generate_students(courses)
        
        return {
            'classrooms': classrooms,
            'courses': courses,
            'instructors': instructors,
            'students': students
        }

    def export_to_excel(self, data: Dict, filename: str):
        """
        Exports the generated school data to an Excel file.
        """
        with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(data['classrooms']).to_excel(writer, sheet_name="Classrooms", index=False)
            pd.DataFrame(data['courses']).to_excel(writer, sheet_name="Courses", index=False)
            pd.DataFrame(data['instructors']).to_excel(writer, sheet_name="Instructors", index=False)
            pd.DataFrame(data['students']).to_excel(writer, sheet_name="Students", index=False)
        print(f"Excel file '{filename}' has been created.")

def create_sample_data(
    num_classrooms: int = 10 * 30,
    num_courses: int = 20 * 30,
    num_instructors: int = 12 * 30,
    num_students: int = 60 * 30,
    export_excel: bool = False,
    excel_filename: str = 'school_data.xlsx'
) -> Dict:
    """
    Generates sample school data with the specified configuration and optionally exports it to an Excel file.

    Args:
        num_classrooms: Number of classrooms to generate.
        num_courses: Number of courses to generate.
        num_instructors: Number of instructors to generate.
        num_students: Number of students to generate.
        export_excel: Whether to export the data to an Excel file.
        excel_filename: Name of the Excel file to create if exporting.

    Returns:
        A dictionary containing the generated school data.
    """
    config = SchoolDataConfig(
        num_classrooms=num_classrooms,
        num_courses=num_courses,
        num_instructors=num_instructors,
        num_students=num_students
    )
    
    generator = SchoolDataGenerator(config)
    data = generator.generate_all_data()
    
    if export_excel:
        generator.export_to_excel(data, excel_filename)
    
    return data

create_sample_data(
    num_classrooms=8*5,
    num_courses=10*5,
    num_instructors=10*5,
    num_students=150*5,
    export_excel=True,
    excel_filename='school_input_data.xlsx'
)