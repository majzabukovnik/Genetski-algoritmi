import ast
import random
import pygad

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Globalni parametri GA
GREEDY_RATIO = 0.7             # delež požrešnih osebkov v začetni populaciji
MUTATION_RATE = 0.15           # verjetnost mutacije konfliktnega predavanja
HEALTHY_MUTATION_FACTOR = 0.2  # zdrava predavanja se mutirajo z MUTATION_RATE * HEALTHY_MUTATION_FACTOR

ASSETS_PATH = 'assets/large/'


def parse_classes(filename):
    classes = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            subject = parts[0]
            durations = ast.literal_eval(" ".join(parts[1:]))
            for d in durations:
                classes.append({'subject': subject, 'duration': d})
    return classes


def parse_classrooms(filename):
    classrooms = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            name = parts[0]
            times = ast.literal_eval(" ".join(parts[1:]))
            classrooms[name] = {'start': times[0], 'end': times[1]}
    return classrooms


def parse_teachers(filename):
    teachers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            name = parts[0]
            raw_str = " ".join(parts[1:]).strip('[]')
            subjects = [s.strip() for s in raw_str.split(',') if s.strip()]
            teachers[name] = subjects
    return teachers


classes = parse_classes(ASSETS_PATH + 'Classes.txt')
classrooms = parse_classrooms(ASSETS_PATH + 'Classrooms.txt')
teachers = parse_teachers(ASSETS_PATH + 'Teachers.txt')

valid_terms_by_duration = {}
all_durations = set(c['duration'] for c in classes)

for d in all_durations:
    terms = []
    for cr_name, cr_range in classrooms.items():
        op = cr_range['start']
        cl = cr_range['end']
        for start in range(op, cl - d + 1):
            terms.append((cr_name, start))
    valid_terms_by_duration[d] = terms

gene_space = []
eligible_teachers_per_lecture = []

for c in classes:
    d = c['duration']
    subj = c['subject']

    num_terms = len(valid_terms_by_duration[d])
    gene_space.append(list(range(num_terms)))

    elig_teach = [t for t, subs in teachers.items() if subj in subs]
    eligible_teachers_per_lecture.append(elig_teach)

    gene_space.append(list(range(len(elig_teach))))


def decode_chromosome(solution):
    schedule = []
    for i, c in enumerate(classes):
        term_idx = int(solution[2 * i])
        teacher_idx = int(solution[2 * i + 1])

        d = c['duration']
        subj = c['subject']

        cr_name, start = valid_terms_by_duration[d][term_idx]
        teacher_name = eligible_teachers_per_lecture[i][teacher_idx]

        schedule.append({
            'subject': subj,
            'duration': d,
            'classroom': cr_name,
            'start': start,
            'end': start + d,
            'teacher': teacher_name
        })
    return schedule


total_hours = sum(c['duration'] for c in classes)
max_fitness = 2 * total_hours


def get_lecture_conflicts(schedule):
    conflicts_per_lecture = []
    for i, lec in enumerate(schedule):
        l_start = lec['start']
        l_end = lec['end']
        l_room = lec['classroom']
        l_teacher = lec['teacher']

        room_conflicts = 0
        teacher_conflicts = 0

        for hour in range(l_start, l_end):
            room_hit = False
            teacher_hit = False

            for j, other in enumerate(schedule):
                if i == j:
                    continue
                if other['start'] <= hour < other['end']:
                    if other['classroom'] == l_room:
                        room_hit = True
                    if other['teacher'] == l_teacher:
                        teacher_hit = True

            if room_hit:
                room_conflicts += 1
            if teacher_hit:
                teacher_conflicts += 1

        conflicts_per_lecture.append({
            'room': room_conflicts,
            'teacher': teacher_conflicts,
            'total': room_conflicts + teacher_conflicts
        })
    return conflicts_per_lecture


def fitness_func(ga_instance, solution, solution_idx):
    schedule = decode_chromosome(solution)
    lect_confs = get_lecture_conflicts(schedule)
    total_conflicts = sum(c['total'] for c in lect_confs)
    score = max_fitness - total_conflicts
    return float(score)


def evaluate_new_lecture(temp_lec, assigned_schedule):
    conf = 0
    for hour in range(temp_lec['start'], temp_lec['end']):
        r_hit = False
        t_hit = False
        for other in assigned_schedule:
            if other['start'] <= hour < other['end']:
                if other['classroom'] == temp_lec['classroom']:
                    r_hit = True
                if other['teacher'] == temp_lec['teacher']:
                    t_hit = True
        if r_hit:
            conf += 1
        if t_hit:
            conf += 1
    return conf


def generate_greedy_chromosome(greedy_ratio=GREEDY_RATIO):
    if random.random() > greedy_ratio:
        return [random.choice(space) for space in gene_space]

    indexed_classes = list(enumerate(classes))
    grouped = {}
    for idx, c in indexed_classes:
        grouped.setdefault(c['duration'], []).append((idx, c))

    sorted_indices = []
    for d in sorted(grouped.keys(), reverse=True):
        group = grouped[d]
        random.shuffle(group)
        sorted_indices.extend(group)

    solution = [0] * (len(classes) * 2)
    assigned_schedule = []

    for i, c in sorted_indices:
        d = c['duration']
        possible_terms = list(enumerate(valid_terms_by_duration[d]))
        random.shuffle(possible_terms)

        elig_teach = eligible_teachers_per_lecture[i]
        teach_indices = list(range(len(elig_teach)))
        random.shuffle(teach_indices)

        best_term_idx = possible_terms[0][0]
        best_teach_idx = teach_indices[0]
        min_conflicts = float('inf')

        for t_idx, (cr_name, start) in possible_terms:
            for th_idx in teach_indices:
                temp_lec = {
                    'subject': c['subject'],
                    'duration': d,
                    'classroom': cr_name,
                    'start': start,
                    'end': start + d,
                    'teacher': elig_teach[th_idx]
                }
                conf = evaluate_new_lecture(temp_lec, assigned_schedule)
                if conf < min_conflicts:
                    min_conflicts = conf
                    best_term_idx = t_idx
                    best_teach_idx = th_idx
                if min_conflicts == 0:
                    break
            if min_conflicts == 0:
                break

        solution[2 * i] = best_term_idx
        solution[2 * i + 1] = best_teach_idx

        cr_name, start = valid_terms_by_duration[d][best_term_idx]
        assigned_schedule.append({
            'subject': c['subject'],
            'duration': d,
            'classroom': cr_name,
            'start': start,
            'end': start + d,
            'teacher': elig_teach[best_teach_idx]
        })

    return solution


sol_per_pop = 60
initial_population = [generate_greedy_chromosome(greedy_ratio=GREEDY_RATIO) for _ in range(sol_per_pop)]


def custom_mutation(offspring, ga_instance):
    mutation_rate = MUTATION_RATE
    for idx in range(offspring.shape[0]):
        chromosome = offspring[idx]
        schedule = decode_chromosome(chromosome)
        lect_confs = get_lecture_conflicts(schedule)

        for i, conf in enumerate(lect_confs):
            # Če ima predavanje konflikt, ciljano napademo napačen gen
            if conf['room'] > 0 or conf['teacher'] > 0:
                if random.random() < mutation_rate:
                    if conf['room'] > 0 and conf['teacher'] > 0:
                        gene_to_mutate = random.choice([2 * i, 2 * i + 1])  # izberemo naključno enega od obeh
                    elif conf['room'] > 0:
                        gene_to_mutate = 2 * i  # napaka je v učilnici/terminu
                    else:
                        gene_to_mutate = 2 * i + 1  # napaka je pri učitelju

                    chromosome[gene_to_mutate] = random.choice(gene_space[gene_to_mutate])
            else:
                # Za "zdrave" gene ohranimo nizko naključno raziskovanje (npr. 20% od mutation_rate)
                if random.random() < (mutation_rate * HEALTHY_MUTATION_FACTOR):
                    gene_to_mutate = random.choice([2 * i, 2 * i + 1])
                    chromosome[gene_to_mutate] = random.choice(gene_space[gene_to_mutate])

    return offspring


def on_generation(ga_instance):
    best_fit = ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
    print(f"Generacija {ga_instance.generations_completed}: Najboljši fitness = {best_fit}")
    if best_fit >= max_fitness:
        print("Najden popoln urnik!")
        return "stop"


ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    initial_population=initial_population,
    gene_space=gene_space,
    num_genes=len(classes) * 2,
    gene_type=int,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type=custom_mutation,
    random_seed=RANDOM_SEED,
    on_generation=on_generation
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"\nKončni fitness: {solution_fitness} / {max_fitness}")
final_schedule = decode_chromosome(solution)

print("\nIzris končnega urnika:")
for lec in final_schedule:
    print(
        f"Predmet: {lec['subject']:<16} | Trajanje: {lec['duration']}h | Učilnica: {lec['classroom']} ({lec['start']}:00 - {lec['end']}:00) | Učitelj: {lec['teacher']}")
