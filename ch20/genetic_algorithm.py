import random
import string
import math

# Our optimization problem - create a matching string!
perfect_string = "Krakens are Majestic Sea Creatures!"
perfect_list = list(perfect_string)
possible_chars = string.ascii_letters + string.whitespace + string.punctuation
string_length = len(perfect_string)


class individual:
    def __init__(self):
        self.string_list = []
        self.create_random_string()
        self.run_fitness_function()

    def create_random_string(self):
        # Initializez a random string
        for i in range(0, string_length):
            char = random.choice(possible_chars)
            self.string_list.append(char)

    def print_self(self):
        print(f"Fitness: {self.fitness}")
        print("".join(self.string_list))

    def run_fitness_function(self):
        # Set difference value to 0
        overall_diff = 0
        # Compare the characters
        for char_1, char_2 in zip(self.string_list, perfect_list):
            if char_1 != char_2:
                overall_diff += 1
        self.fitness = overall_diff
        return self.fitness

    def mutate(self):
        # Pick a random index and mutate the character there
        index_to_mutate = random.randrange(0, string_length)
        new_char = random.choice(possible_chars)
        self.string_list[index_to_mutate] = new_char

    def __lt__(self, other):
        # Overloaded operator so we can sort this in python easily
        return self.fitness < other.fitness


class Population:
    def __init__(self, size, mutation_rate, tournament_size, selection_method="tournament"):
        # Set the population information
        self.population_size = size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        if selection_method == "tournament_with_crossover":
            self.selection_method = self.tournament_selection_with_crossover
        else:
            self.selection_method = self.tournament_selection
        # Initialize the population with individuals
        self.population_list = []
        for i in range(0, self.population_size):
            new_member = individual()
            self.population_list.append(new_member)
        self.population_list.sort()

    def print_population(self, num_members=False):
        if not num_members:
            num_members = self.population_size
        for i in range(0, num_members):
            self.population_list[i].print_self()

    def copy_member(self, og, new_string=False):
        # Initialize a new individual
        new_member = individual()
        # Copy the relevant parameters to the new individual
        if new_string:
            new_member.string_list = new_string.copy()
        else:
            new_member.string_list = og.string_list.copy()
        new_member.fitness = og.fitness
        # Return new individual
        return new_member

    def tournament_selection(self):
        # Get a sample for the tournament
        selection_list = random.sample(
            self.population_list, self.tournament_size)
        # Sort by fitness
        selection_list.sort()
        # The winner the the most fit (lowest fitness)
        winner = selection_list[0]
        # Return a copy of the winner
        return self.copy_member(winner)

    def tournament_selection_with_crossover(self):
        # Get a sample for the tournament
        selection_list = random.sample(
            self.population_list, self.tournament_size)
        # Sort by fitness
        selection_list.sort()
        # The winner the the most fit (lowest fitness)
        winner_1 = selection_list[0]
        winner_2 = selection_list[1]
        random_point = random.randrange(0, len(winner_1.string_list))
        half_1 = winner_1.string_list[0:random_point]
        half_2 = winner_2.string_list[random_point:len(winner_2.string_list)]
        new_string = half_1 + half_2
        # Return a copy of the winner with a new string
        return self.copy_member(winner_1, new_string=new_string)

    def create_new_generation(self):
        new_generation = []
        for i in range(0, self.population_size):
            new_generation.append(self.selection_method())
        self.population_list = new_generation

    def mutate_population(self):
        # Number of population members to mutate based on our mutation rate
        mutation_number = math.floor(self.population_size*self.mutation_rate)
        # Sample our mutants
        to_mutate = random.sample(self.population_list, mutation_number)
        # Perform mutation
        for member in to_mutate:
            member.mutate()

    def calc_population_fitness(self):
        for member in self.population_list:
            member.run_fitness_function()

    def run_generation(self):
        self.create_new_generation()
        self.mutate_population()
        self.calc_population_fitness()
        self.population_list.sort()


# Actually run the experiment
population_size = 1000
mutation_rate = 0.3
tournament_size = 4
selection_method = "tournament"
generations = 300

pop = Population(
    population_size,
    mutation_rate,
    tournament_size,
    selection_method=selection_method
)
done = False
i = 0
while not done:
    i += 1
    print(f"Generation {i}")
    pop.run_generation()
    pop.print_population(num_members=1)
    if pop.population_list[0].fitness == 0 or i >= 300:
        done = True
