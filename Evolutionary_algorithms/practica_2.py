import random

def generate_schedule(num_classes, subjects):
    # Define days and hours
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(8, 15))

    # Initialize the schedule as an empty dictionary
    schedules = []

    for class_num in range(1, num_classes + 1):
        schedule = {}
        for day in days:
            for hour in hours:
                # Assign a subject to the class
                subject = random.choice(subjects)
                # Store the subject in the schedule
                schedule[(day, hour)] = subject
        # Append the schedule for the current class to the list
        schedules.append(schedule)

    return schedules

def print_schedule(schedule):
    # Print the schedule in a readable format
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'))
    for hour in range(8, 15):
        print("{:<10}".format(f'{hour}:00-{hour + 1}:00'), end=' ')
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            print("{:<10}".format(schedule.get((day, hour), '')), end=' ')
        print()

def parent_selection(schedules):
    for schedule in schedules:

if __name__ == "__main__":
    # Set the number of classes
    num_classes = int(input("Enter the number of classes: "))
    num_teachers = int(input("Enter the number of classes: "))
    num_hours = int(input("Enter the number of classes: "))

    # Define a list of subjects
    subjects = ['Math', 'English', 'Chemistry', 'History', 'Spanish', 'Art', 'Music', 'Physics']

    # Generate and print the schedule for each class
    schedules = generate_schedule(num_classes, subjects)
    for class_num, schedule in enumerate(schedules, start=1):
        print(f"\nSchedule for Class {class_num}:\n")
        print_schedule(schedule)
        print('\n' + '-'*50)  # Separate schedules with a line for better readability




import random

class Schedule:
    def __init__(self, classes, professors, time_slots):
        self.classes = classes
        self.professors = professors
        self.time_slots = time_slots
        self.schedule = {}  # Key: (class, time_slot), Value: (course, professor)

    def generate_random_schedule(self):
        """
        Generates a random schedule based on the available classes, professors, and time slots.
        This function ensures that the schedule respects hard constraints.
        """
        for class_ in self.classes:
            for course in class_['courses']:
                assigned_hours = 0
                while assigned_hours < course['hours']:
                    # Randomly select a time slot and a professor
                    time_slot = random.choice(self.time_slots)
                    professor = random.choice([prof for prof in self.professors if course['name'] in prof['courses']])
                    
                    # Check if the time slot and professor are available
                    if (class_['name'], time_slot) not in self.schedule and time_slot not in professor['unavailable_times']:
                        self.schedule[(class_['name'], time_slot)] = (course['name'], professor['name'])
                        assigned_hours += 1

        return self.schedule

class EvolutionaryAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, classes, professors, time_slots):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.classes = classes
        self.professors = professors
        self.time_slots = time_slots
        self.population = []

    def initialize_population(self):
        """
        Initializes the population with random schedules.
        """
        for _ in range(self.population_size):
            schedule = Schedule(self.classes, self.professors, self.time_slots)
            random_schedule = schedule.generate_random_schedule()
            self.population.append(random_schedule)

# Placeholder data for classes, professors, and time slots
classes = [{'name': 'Class1', 'courses': [{'name': 'Math', 'hours': 5}, {'name': 'Science', 'hours': 4}]}]
professors = [{'name': 'Prof1', 'courses': ['Math'], 'unavailable_times': ['Monday 9AM']}, 
              {'name': 'Prof2', 'courses': ['Science'], 'unavailable_times': []}]
time_slots = ['Monday 9AM', 'Monday 10AM', 'Tuesday 9AM', 'Tuesday 10AM']

# Create an instance of the evolutionary algorithm
ea = EvolutionaryAlgorithm(population_size=10, mutation_rate=0.1, crossover_rate=0.8, classes=classes, professors=professors, time_slots=time_slots)

# Initialize the population
ea.initialize_population
