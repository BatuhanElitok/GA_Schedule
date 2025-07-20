import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional

def create_realistic_schedule_data(
    num_classrooms=40,
    num_courses=75,
    num_instructors=50,
    num_students=1000,
    export_excel=True,
    excel_filename='realistic_school_data.xlsx'
) -> Dict:
    """
    Generate realistic sample data for university scheduling with proper time slots
    COMPLETELY FIXED - No overlapping time slots guaranteed
    """
    print(f"🏗️  Generating REALISTIC scheduling data:")
    print(f"   📍 {num_classrooms} classrooms")
    print(f"   📚 {num_courses} courses") 
    print(f"   👨‍🏫 {num_instructors} instructors")
    print(f"   👥 {num_students} students")
    
    # Define proper scheduling constants
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    classroom_types = ["Lecture Hall", "Lab", "Seminar Room", "Computer Lab"]
    course_prefixes = ["CS", "MATH", "PHYS", "CHEM", "BIO", "ENG", "HIST", "ECON", "PSYCH", "ART", 
                      "MUSIC", "PHIL", "SOC", "ANTH", "GEOL", "STAT", "MGMT", "FIN", "MKT", "ACC"]
    
    # FIXED: Proper time slots for 3-hour courses
    VALID_START_TIMES = [8, 9, 10, 11, 12, 13, 14, 15]  # Can start at these hours
    COURSE_DURATION = 3  # Each course is 3 hours
    MAX_END_TIME = 18    # University closes at 6 PM
    
    print("🏢 Generating classrooms...")
    # Generate classrooms with realistic distribution
    classrooms = []
    classroom_counts = {
        "Lecture Hall": int(num_classrooms * 0.4),  # 40% lecture halls
        "Lab": int(num_classrooms * 0.25),          # 25% labs
        "Seminar Room": int(num_classrooms * 0.25), # 25% seminar rooms
        "Computer Lab": int(num_classrooms * 0.1)   # 10% computer labs
    }
    
    # Fill remaining classrooms
    remaining = num_classrooms - sum(classroom_counts.values())
    classroom_counts["Lecture Hall"] += remaining
    
    classroom_id = 101
    for room_type, count in classroom_counts.items():
        for _ in range(count):
            # Realistic capacity based on room type
            if room_type == "Lecture Hall":
                capacity = random.randint(80, 250)
            elif room_type == "Lab":
                capacity = random.randint(20, 45)
            elif room_type == "Computer Lab":
                capacity = random.randint(25, 60)
            else:  # Seminar Room
                capacity = random.randint(15, 35)
            
            classrooms.append({
                "Name": f"Room {classroom_id}",
                "Type": room_type,
                "Capacity": capacity
            })
            classroom_id += 1
    
    print("📚 Generating courses...")
    # Generate courses with realistic requirements
    courses = []
    course_names = set()
    
    courses_per_prefix = num_courses // len(course_prefixes)
    remaining_courses = num_courses % len(course_prefixes)
    
    for i, prefix in enumerate(course_prefixes):
        courses_to_generate = courses_per_prefix + (1 if i < remaining_courses else 0)
        
        for j in range(courses_to_generate):
            # Generate unique course number
            course_number = 100 + (j * 3) + random.randint(0, 2)
            course_name = f"{prefix} {course_number}"
            
            # Ensure uniqueness
            while course_name in course_names:
                course_number += 1
                course_name = f"{prefix} {course_number}"
            
            course_names.add(course_name)
            
            # Determine classroom type based on course prefix
            if prefix in ["CS", "PHYS", "CHEM", "BIO", "GEOL"]:
                classroom_type_weights = [0.3, 0.4, 0.1, 0.2]  # Lab-heavy
            elif prefix in ["MATH", "ECON", "STAT", "FIN", "ACC"]:
                classroom_type_weights = [0.7, 0.0, 0.2, 0.1]  # Lecture-heavy
            elif prefix in ["MGMT", "MKT", "SOC", "PSYCH"]:
                classroom_type_weights = [0.4, 0.0, 0.6, 0.0]  # Seminar-heavy
            else:
                classroom_type_weights = [0.5, 0.1, 0.3, 0.1]  # Mixed
            
            classroom_type = np.random.choice(classroom_types, p=classroom_type_weights)
            
            # Realistic expected enrollment
            base_enrollment = {
                "Lecture Hall": random.randint(60, 180),
                "Lab": random.randint(16, 30),
                "Computer Lab": random.randint(20, 45),
                "Seminar Room": random.randint(12, 28)
            }[classroom_type]
            
            # Adjust for course level
            if course_number >= 400:
                base_enrollment = int(base_enrollment * 0.6)  # Graduate level
            elif course_number >= 300:
                base_enrollment = int(base_enrollment * 0.8)  # Upper level
            
            courses.append({
                "Name": course_name,
                "Classroom Type": classroom_type,
                "Expected Enrollment": base_enrollment
            })
    
    print("👨‍🏫 Generating instructors with FOOLPROOF time slots...")
    # Generate instructors with guaranteed non-overlapping time slots
    instructors = []
    course_instructor_mapping = {course["Name"]: [] for course in courses}
    
    # Calculate course assignments
    all_course_names = [course["Name"] for course in courses]
    unassigned_courses = all_course_names.copy()
    
    print("   📋 Assigning courses to instructors...")
    for i in range(num_instructors):
        instructor_name = f"Instructor {i+1}"
        
        # Determine course load
        if i < num_instructors * 0.1:  # 10% senior faculty
            num_courses_for_instructor = random.randint(5, 7)
        elif i < num_instructors * 0.3:  # 20% part-time
            num_courses_for_instructor = random.randint(2, 3)
        else:  # 70% regular faculty
            num_courses_for_instructor = random.randint(3, 5)
        
        # Assign courses
        assigned_courses = []
        
        # Assign from unassigned courses first
        if unassigned_courses:
            courses_from_unassigned = min(num_courses_for_instructor, len(unassigned_courses))
            selected_unassigned = random.sample(unassigned_courses, courses_from_unassigned)
            assigned_courses.extend(selected_unassigned)
            
            for course in selected_unassigned:
                unassigned_courses.remove(course)
                course_instructor_mapping[course].append(instructor_name)
        
        # Fill remaining slots for backup instructors
        remaining_slots = num_courses_for_instructor - len(assigned_courses)
        if remaining_slots > 0:
            available_courses = [c for c in all_course_names if c not in assigned_courses]
            if available_courses:
                additional_courses = random.sample(
                    available_courses, 
                    min(remaining_slots, len(available_courses))
                )
                assigned_courses.extend(additional_courses)
                
                for course in additional_courses:
                    course_instructor_mapping[course].append(instructor_name)
        
        # FOOLPROOF: Generate verified non-overlapping time slots (back-to-back allowed)
        availability = {}
        
        # Define all possible foolproof combinations (manually verified)
        # Now including back-to-back combinations
        foolproof_combinations = [
            # =================== SINGLE BLOCKS (8 combinations) ===================
            [(8, 11)], [(9, 12)], [(10, 13)], [(11, 14)],
            [(12, 15)], [(13, 16)], [(14, 17)], [(15, 18)],
            
            # =================== TWO BLOCKS WITH GAPS (10 combinations) ===================
            [(8, 11), (12, 15)],   # Gap: 1 hour (11-12)
            [(8, 11), (13, 16)],   # Gap: 2 hours (11-13)
            [(8, 11), (14, 17)],   # Gap: 3 hours (11-14)  
            [(8, 11), (15, 18)],   # Gap: 4 hours (11-15)
            [(9, 12), (13, 16)],   # Gap: 1 hour (12-13)
            [(9, 12), (14, 17)],   # Gap: 2 hours (12-14)
            [(9, 12), (15, 18)],   # Gap: 3 hours (12-15)
            [(10, 13), (14, 17)],  # Gap: 1 hour (13-14)
            [(10, 13), (15, 18)],  # Gap: 2 hours (13-15)
            [(11, 14), (15, 18)],  # Gap: 1 hour (14-15)
            
            # =================== BACK-TO-BACK COMBINATIONS (5 combinations) ===================
            [(8, 11), (11, 14)],   # 08.00-11.00 | 11.00-14.00 (back-to-back)
            [(9, 12), (12, 15)],   # 09.00-12.00 | 12.00-15.00 (back-to-back)
            [(10, 13), (13, 16)],  # 10.00-13.00 | 13.00-16.00 (back-to-back)
            [(11, 14), (14, 17)],  # 11.00-14.00 | 14.00-17.00 (back-to-back)
            [(12, 15), (15, 18)],  # 12.00-15.00 | 15.00-18.00 (back-to-back)
            
            # =================== THREE BACK-TO-BACK COMBINATIONS (3 combinations) ===================
            [(8, 11), (11, 14), (14, 17)],   # 08.00-11.00 | 11.00-14.00 | 14.00-17.00 (9 hours straight!)
            [(9, 12), (12, 15), (15, 18)],   # 09.00-12.00 | 12.00-15.00 | 15.00-18.00 (9 hours straight!)
            [(10, 13), (13, 16)],             # Already added above (duplicate - will be removed)
            
            # =================== MIXED COMBINATIONS (back-to-back + gap) ===================
            [(8, 11), (11, 14), (15, 18)],   # Back-to-back morning + gap + evening
            [(8, 11), (12, 15), (15, 18)],   # Morning + back-to-back afternoon-evening
        ]
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        for combo in foolproof_combinations:
            combo_tuple = tuple(sorted(combo))  # Sort to catch duplicates
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
        
        # Add weight preferences for realistic distribution
        weighted_combinations = []
        
        for combo in unique_combinations:
            # Single blocks: High probability (realistic for most instructors)
            if len(combo) == 1:
                weight = 5
                
            # Two blocks: Medium probability  
            elif len(combo) == 2:
                # Back-to-back gets medium weight (some instructors like continuous teaching)
                start1, end1 = combo[0]
                start2, end2 = combo[1]
                if end1 == start2:  # Back-to-back
                    weight = 3
                else:  # With gap
                    weight = 4
                    
            # Three blocks: Low probability (only very dedicated instructors)
            else:
                weight = 1
            
            # Add this combination multiple times based on weight
            for _ in range(weight):
                weighted_combinations.append(combo)
        
        for day in days:
            # 80% chance available each day
            if random.random() < 0.8:
                # Choose instructor preference
                if random.random() < 0.3:  # Morning person
                    morning_combos = [combo for combo in weighted_combinations 
                                    if any(start <= 10 for start, end in combo)]
                    chosen_blocks = random.choice(morning_combos) if morning_combos else random.choice(weighted_combinations)
                elif random.random() < 0.3:  # Afternoon person
                    afternoon_combos = [combo for combo in weighted_combinations 
                                      if any(start >= 12 for start, end in combo)]
                    chosen_blocks = random.choice(afternoon_combos) if afternoon_combos else random.choice(weighted_combinations)
                else:  # Flexible
                    chosen_blocks = random.choice(weighted_combinations)
                
                # Format as time strings
                formatted_slots = []
                for start, end in chosen_blocks:
                    formatted_slots.append(f"{start:02d}.00-{end:02d}.00")
                
                availability[f"{day} Slots"] = " | ".join(formatted_slots)
            else:
                availability[f"{day} Slots"] = ""  # Not available
        
        # Create instructor record
        instructor_data = {
            "Name": instructor_name,
            "Courses": ", ".join(assigned_courses)
        }
        instructor_data.update(availability)
        instructors.append(instructor_data)
    
    # Handle remaining unassigned courses
    if unassigned_courses:
        print(f"   ⚠️  Assigning {len(unassigned_courses)} remaining courses...")
        for course in unassigned_courses:
            random_instructor_idx = random.randint(0, len(instructors) - 1)
            current_courses = instructors[random_instructor_idx]["Courses"]
            if current_courses:
                instructors[random_instructor_idx]["Courses"] += f", {course}"
            else:
                instructors[random_instructor_idx]["Courses"] = course
            course_instructor_mapping[course].append(instructors[random_instructor_idx]["Name"])
    
    print("👥 Generating realistic student enrollments...")
    # Generate students with realistic course loads
    students = []
    
    for i in range(num_students):
        # Realistic course load distribution
        if i < num_students * 0.1:  # 10% part-time
            num_courses_per_student = random.randint(2, 3)
        elif i < num_students * 0.2:  # 10% heavy load
            num_courses_per_student = random.randint(7, 8)
        else:  # 80% normal load
            num_courses_per_student = random.randint(4, 6)
        
        # Select courses with academic logic
        if random.random() < 0.7:  # 70% follow level preferences
            if random.random() < 0.4:  # Freshman
                preferred_courses = [c["Name"] for c in courses if int(c["Name"].split()[1]) < 250]
            elif random.random() < 0.7:  # Sophomore/Junior
                preferred_courses = [c["Name"] for c in courses if 200 <= int(c["Name"].split()[1]) < 400]
            else:  # Senior/Graduate
                preferred_courses = [c["Name"] for c in courses if int(c["Name"].split()[1]) >= 300]
            
            if len(preferred_courses) >= num_courses_per_student:
                enrolled_courses = random.sample(preferred_courses, num_courses_per_student)
            else:
                enrolled_courses = random.sample(all_course_names, num_courses_per_student)
        else:
            enrolled_courses = random.sample(all_course_names, num_courses_per_student)
        
        students.append({
            "Name": f"Student {i+1}",
            "Courses": ", ".join(enrolled_courses)
        })
        
        if i % 250 == 0:
            progress = (i / num_students) * 100
            print(f"   📊 Student generation: {progress:.1f}%")
    
    # Verification
    print("\n🔍 Verifying data consistency...")
    courses_without_instructors = []
    for course_name in all_course_names:
        if not course_instructor_mapping[course_name]:
            courses_without_instructors.append(course_name)
    
    if courses_without_instructors:
        print(f"❌ ERROR: {len(courses_without_instructors)} courses have no instructors!")
    else:
        print("✅ All courses have instructors assigned!")
    
    # Statistics
    print(f"\n📊 REALISTIC DATA STATISTICS:")
    print(f"   📚 Total courses: {len(courses)}")
    print(f"   👨‍🏫 Total instructors: {len(instructors)}")
    print(f"   🏢 Total classrooms: {len(classrooms)}")
    print(f"   👥 Total students: {len(students)}")
    print(f"   📅 Time slots: {len(unique_combinations)} foolproof combinations (including back-to-back)")
    
    # Export to Excel
    if export_excel:
        print(f"\n💾 Exporting to {excel_filename}...")
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            pd.DataFrame(classrooms).to_excel(writer, sheet_name="Classrooms", index=False)
            pd.DataFrame(courses).to_excel(writer, sheet_name="Courses", index=False)
            pd.DataFrame(instructors).to_excel(writer, sheet_name="Instructors", index=False)
            pd.DataFrame(students).to_excel(writer, sheet_name="Students", index=False)
        
        print(f"✅ Realistic data exported successfully!")
    
    return {
        "classrooms": classrooms,
        "courses": courses,
        "instructors": instructors,
        "students": students,
        "course_instructor_mapping": course_instructor_mapping
    }

def validate_realistic_data(excel_filename='realistic_school_data.xlsx'):
    """Validate the generated data for overlaps"""
    print(f"\n🔍 Validating data in {excel_filename}...")
    
    try:
        instructors_df = pd.read_excel(excel_filename, sheet_name='Instructors')
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        overlap_count = 0
        total_instructors = len(instructors_df)
        
        for _, instructor in instructors_df.iterrows():
            instructor_name = instructor['Name']
            
            for day in days:
                day_slots = str(instructor[f'{day} Slots'])
                if day_slots and day_slots != 'nan' and day_slots.strip():
                    # Parse time slots
                    time_blocks = []
                    for slot in day_slots.split(' | '):
                        if '-' in slot and '.' in slot:
                            try:
                                start_str, end_str = slot.split('-')
                                start_hour = int(float(start_str))
                                end_hour = int(float(end_str))
                                time_blocks.append((start_hour, end_hour))
                            except ValueError:
                                continue
                    
                    # Check for overlaps within this day
                    for i in range(len(time_blocks)):
                        for j in range(i + 1, len(time_blocks)):
                            start1, end1 = time_blocks[i]
                            start2, end2 = time_blocks[j]
                            
                            # Check if blocks overlap
                            if not (end1 <= start2 or start1 >= end2):
                                print(f"❌ OVERLAP FOUND: {instructor_name} on {day}")
                                print(f"   Block 1: {start1:02d}.00-{end1:02d}.00")
                                print(f"   Block 2: {start2:02d}.00-{end2:02d}.00")
                                overlap_count += 1
        
        print(f"\n📊 Validation Results:")
        print(f"   Total instructors: {total_instructors}")
        print(f"   Overlaps found: {overlap_count}")
        
        if overlap_count == 0:
            print(f"   ✅ PERFECT! No overlapping time slots found!")
            return True
        else:
            print(f"   ❌ FAILED! Found {overlap_count} overlapping time slots!")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    print("🏛️  FOOLPROOF UNIVERSITY DATA GENERATOR")
    print("100% Guaranteed No Overlapping Time Slots")
    print("=" * 60)
    
    # Generate realistic data
    data = create_realistic_schedule_data(
        num_classrooms=20,     # 25 sınıf (çeşitli türlerde)
        num_courses=50,        # 60 ders (3-4 bölümün dersleri)
        num_instructors=13,    # 35 eğitmen 
        num_students=250,      # 800 öğrenci
        export_excel=True,
        excel_filename='realistic_school_data.xlsx'
    )
    
    # Validate the data
    print("\n" + "=" * 60)
    print("🔍 VALIDATION")
    print("=" * 60)
    
    is_valid = validate_realistic_data('realistic_school_data.xlsx')
    
    if is_valid:
        print("\n🎉 SUCCESS! Foolproof data ready!")
        print("📝 Key features:")
        print("   ✅ Zero overlapping time slots (guaranteed)")
        print("   ✅ 18 pre-verified time combinations")
        print("   ✅ Realistic instructor patterns")
        print("   ✅ Academic course level groupings")
        print("   ✅ Proper classroom assignments")
        print("\n🚀 Ready for breakthrough optimization!")
    else:
        print("\n⚠️  Validation failed - please check the code.")