import ast
import pygad

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
            # Združi ostanek vrstice, odstrani oklepaje in razdeli po vejici
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


def fitness_func(ga_instance, solution, solution_idx):
    schedule = decode_chromosome(solution)
    score = 0

    for i, lec in enumerate(schedule):
        l_start = lec['start']
        l_end = lec['end']
        l_room = lec['classroom']
        l_teacher = lec['teacher']

        for hour in range(l_start, l_end):
            room_conflict = False
            teacher_conflict = False

            for j, other in enumerate(schedule):
                if i == j:
                    continue
                if other['start'] <= hour < other['end']:
                    if other['classroom'] == l_room:
                        room_conflict = True
                    if other['teacher'] == l_teacher:
                        teacher_conflict = True

            if not room_conflict:
                score += 1
            if not teacher_conflict:
                score += 1

    return float(score)


def on_generation(ga_instance):
    best_fit = ga_instance.best_solution()[1]
    print(f"Generacija {ga_instance.generations_completed}: Najboljši fitness = {best_fit}")
    if best_fit >= max_fitness:
        print("Najden popoln urnik!")
        return "stop"


ga_instance = pygad.GA(
    num_generations=2000,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=60,
    num_genes=len(classes) * 2,
    gene_space=gene_space,
    gene_type=int,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=15,
    on_generation=on_generation
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"\nKončni fitness: {solution_fitness} / {max_fitness}")
final_schedule = decode_chromosome(solution)

print("\nIzris končnega urnika:")
for lec in final_schedule:
    print(
        f"Predmet: {lec['subject']:<16} | Trajanje: {lec['duration']}h | Učilnica: {lec['classroom']} ({lec['start']}:00 - {lec['end']}:00) | Učitelj: {lec['teacher']}")