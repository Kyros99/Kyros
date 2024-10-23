import numpy as np
import random
import time
import multiprocessing
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


tic = time.time()


class Portal:
    def __init__(self, onoma, mikos, ipsos, x, y):
        self.onoma = onoma
        self.mikos = mikos
        self.ipsos = ipsos
        self.x = x
        self.y = y


class Terminal:
    def __init__(self, onoma, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.onoma = onoma


class Individual:
    def __init__(self, generation, prototype):

        self.chromosome = np.full(2 * X + 2 * Y, -1)
        self.generation = generation
        self.prototype = prototype
        self.Fitness_Score = 0

        if self.prototype is None:
            for terminal in Terminals:
                if terminal.y == Y + 33:
                    self.chromosome[terminal.x] = terminal.id
                    continue
                elif terminal.y == -33:
                    self.chromosome[terminal.x + X] = terminal.id
                    continue
                elif terminal.x == -33:
                    self.chromosome[terminal.y + 2 * X] = terminal.id
                    continue
                elif terminal.x == X + 32:
                    self.chromosome[terminal.y + 2 * X + Y] = terminal.id
                    continue

        elif type(self.prototype) is np.ndarray:

            self.chromosome = np.array(self.prototype)
            np.random.shuffle(self.chromosome)

    def visualize(self):
        Xs = []
        Ys = []
        for i in range(len(self.chromosome)):
            if self.chromosome[i] != -1:
                xx, yy = self.getXY(i)
                Xs.append(xx)
                Ys.append(yy)

        fig, ax = plt.subplots()

        ax.plot([0, 1757], [0, 1744], "None")

        ax.add_patch(
            Rectangle((0, 0), 1757, 1744, edgecolor="red", facecolor="None", lw=1)
        )

        plt.scatter(Xs, Ys, marker=".", s=2)

        plt.show()

    def getXY(self, index):
        if index < X:
            coordinateX = index
            coordinateY = Y + 33
        elif index < 2 * X:
            coordinateX = index - X
            coordinateY = -33
        elif index < 2 * X + Y:
            coordinateY = index - 2 * X
            coordinateX = -33
        elif index < 2 * X + 2 * Y:
            coordinateY = index - (2 * X + Y)
            coordinateX = X + 32
        return coordinateX, coordinateY

    def PinsInSamePlaceCheck(self):
        i = 0
        for pin in self.chromosome:
            if pin != -1:
                i = i + 1
        print(i)

    def Fitness(self):

        FitnessScore = 0

        for i in range(len(self.chromosome)):
            if self.chromosome[i] != -1:
                a = self.chromosome[i] - 1
                sortedArr[a][2] = i

        for i in range(len(sortedArr)):
            Xmax, Xmin, Ymax, Ymin = -500, 20000, -500, 20000

            port = Portals[sortedArr[i][0]]
            Xi, Yi = self.getXY(sortedArr[i][2])

            if Xi < Xmin:
                Xmin = Xi
            if Xi > Xmax:
                Xmax = Xi
            if Yi < Ymin:
                Ymin = Yi
            if Yi > Ymax:
                Ymax = Yi

            if port.x < Xmin:
                Xmin = port.x
            if port.x + port.mikos > Xmax:
                Xmax = port.x + port.mikos
            if port.y < Ymin:
                Ymin = port.y
            if port.y + port.ipsos > Ymax:
                Ymax = port.y + port.ipsos

            FitnessScore = FitnessScore + (Xmax - Xmin) + (Ymax - Ymin)
        self.Fitness_Score = FitnessScore
        return FitnessScore


class GeneticAlgorithm:
    def __init__(self, SizeOfPopulation):

        self.Population = []

        for i in range(SizeOfPopulation):
            if len(self.Population) == 0:
                self.Population.append(Individual(0, None))
            else:
                self.Population.append(Individual(0, self.Population[0].chromosome))

    def Stats(self):
        for i in self.Population:
            print(
                "Fitness value is:", i.Fitness_Score, "  Generation is ", i.generation
            )

    def PortalsWireLength(self):

        Length = 0
        for row in Nets:
            Xmax, Xmin, Ymax, Ymin = -50, 5000, -50, 5000
            if not any("p" in word for word in row):
                for name in row:
                    port = Portals[int(name[1:])]
                    if port.x < Xmin:
                        Xmin = port.x
                    if port.x + port.mikos > Xmax:
                        Xmax = port.x + port.mikos
                    if port.y < Ymin:
                        Ymin = port.y
                    if port.y + port.ipsos > Ymax:
                        Ymax = port.y + port.ipsos

                Length = Length + ((Xmax - Xmin) + (Ymax - Ymin))
        return Length

    def SumOfFitness(self):

        summ = 0
        for sol in self.Population:
            summ = summ + sol.Fitness_Score
        return summ

    def RouletteWheel(self):

        summ = self.SumOfFitness()
        probabilities = [a.Fitness_Score / summ for a in self.Population]
        return np.random.choice(self.Population, p=probabilities)
    
    @profile
    def Crossover(self, generation):

        x1 = random.randrange(len(self.Population[0].chromosome))
        x2 = random.randrange(len(self.Population[0].chromosome))

        if x1 > x2:
            x1, x2 = x2, x1

        child1 = Individual(0, 0)
        child2 = Individual(0, 0)

        parent1 = self.RouletteWheel()
        parent2 = self.RouletteWheel()

        child1.chromosome[x1:x2] = parent2.chromosome[x1:x2]
        child2.chromosome[x1:x2] = parent1.chromosome[x1:x2]

        for i in range(0, x1):

            if parent1.chromosome[i] != -1:
                if parent1.chromosome[i] not in child1.chromosome:
                    child1.chromosome[i] = parent1.chromosome[i]

            if parent2.chromosome[i] != -1:
                if parent2.chromosome[i] not in child2.chromosome:
                    child2.chromosome[i] = parent2.chromosome[i]

        for i in range(x2, len(parent1.chromosome)):

            if parent1.chromosome[i] != -1:
                if parent1.chromosome[i] not in child1.chromosome:
                    child1.chromosome[i] = parent1.chromosome[i]

            if parent2.chromosome[i] != -1:
                if parent2.chromosome[i] not in child2.chromosome:
                    child2.chromosome[i] = parent2.chromosome[i]

        for i in range(x1, x2):

            if parent1.chromosome[i] != -1:
                if parent1.chromosome[i] not in child1.chromosome:
                    indexx = np.where(child1.chromosome == -1)
                    k = random.randrange(len(indexx[0]))
                    child1.chromosome[indexx[0][k]] = parent1.chromosome[i]

            if parent2.chromosome[i] != -1:
                if parent2.chromosome[i] not in child2.chromosome:
                    index = np.where(child2.chromosome == -1)
                    k = random.randrange(len(index[0]))
                    child2.chromosome[index[0][k]] = parent2.chromosome[i]

        self.Mutation(child1)
        self.Mutation(child2)

        return child1, child2

    def Mutation(self, child):

        if random.random() < 0.5:
            lista = np.where(child.chromosome != -1)

            x1 = random.choice(lista[0])
            x2 = random.choice(lista[0])

            child.chromosome[x1], child.chromosome[x2] = (
                child.chromosome[x2],
                child.chromosome[x1],
            )

        child.Fitness()

    def SortPopulation(self):
        self.Population = sorted(
            self.Population,
            key=lambda Population: Population.Fitness_Score,
            reverse=False,
        )

    def One_Point_Crossover(self,generation):
        
        Size = len(self.Population[0].chromosome)
        x1 = random.randrange(Size)
        child1 = Individual(0, 0)
        child2 = Individual(0, 0)

        parent1 = self.RouletteWheel()

        parent2 = self.RouletteWheel()

        child1.chromosome[:x1] = parent1.chromosome[:x1]
        child2.chromosome[:x1] = parent2.chromosome[:x1]
        
        j1 = j2 = 0
        
        index = np.arange(x1,Size)
        np.random.shuffle(index)
        
        index2 = np.arange(x1,Size)
        np.random.shuffle(index2)
        
        Pins = [x for x in parent2.chromosome if x != -1 if x not in child1.chromosome[:x1] ]
        for pin in Pins:
            
            child1.chromosome[index[j1]] = pin
            j1 = j1 + 1

        Pins = [x for x in parent1.chromosome if x != -1 if x not in child2.chromosome[:x1] ]
        for pin in Pins:
            
            child2.chromosome[index2[j2]] = pin
            j2 = j2 + 1

    
        self.Mutation(child1)
        self.Mutation(child2)
        
        
        return child1,child2

Portals = []
Terminals = []

fp = open("ibm02.pl")
pinakas = np.array(fp.readlines())

i = 1
for row in pinakas[4:]:
    if row[1] == "a":
        Portals.append(
            Portal(row.split()[0], None, None, int(row.split()[1]), int(row.split()[2]))
        )
    elif row[1] == "p":
        Terminals.append(
            Terminal(row.split()[0], i, int(row.split()[1]), int(row.split()[2]))
        )
        i = i + 1

fp.close()

fp = open("ibm02.nodes")
pinakas = np.array(fp.readlines())
i = 0

for row in pinakas[7:]:
    if int(row.split()[1]) == 1:
        break
    else:
        Portals[i].mikos = int(row.split()[1])
        Portals[i].ipsos = int(row.split()[2])
        i = i + 1

fp.close()

fp = open("ibm02.scl")
pinakas = np.array(fp.readlines())

for row in pinakas:
    if len(row) > 1:
        
        if row.split()[0] == "NumRows":
            Numrows = int(row.split()[2])
        elif row.split()[0] == "Height":
            YpsosGrammhs = int(row.split()[2])
        elif row.split()[0] == "SubrowOrigin":
            Mhkos = int(row.split()[5])
            break

fp.close()

fp = open("ibm02.nets")
pinakas2 = np.array(fp.readlines())
pinakas = pinakas2[7:]

Nets = []

for i in range(len(pinakas)):
    if pinakas[i].split()[0] == "NetDegree":
        NetDegree = int(pinakas[i].split()[2])
        Nets.append([pinakas[k + 1 + i].split()[0] for k in range(NetDegree)])

count = 0
for row in Nets:
    for pin in row:
        if "p" in pin:
            count = count + 1
k = 0
PinsId = np.zeros((count, 3))
for i in range(len(Nets)):
    if "p" in Nets[i][0]:
        PinsId[k][0] = Nets[i][1][1:]
        PinsId[k][1] = Nets[i][0][1:]
        k = k + 1
    elif "p" in Nets[i][1]:
        PinsId[k][0] = Nets[i][0][1:]
        PinsId[k][1] = Nets[i][1][1:]
        k = k + 1

sortedArr = PinsId[np.argsort(PinsId[:, 1])]
sortedArr = sortedArr.astype(int)

X = Mhkos
Y = YpsosGrammhs * Numrows

GA = GeneticAlgorithm(100)
Wire = GA.PortalsWireLength()

for i in GA.Population:
    i.Fitness()
GA.SortPopulation()

kappa = []
kappa2 = []
for i in GA.Population:
    kappa.append(i.Fitness_Score)
    kappa2.append(i.chromosome)

counter = 0
MaxNumberOfGenerations = 100
NumberOfGenerations = 100000
children = []


for j in range(10):
    
    max_score = GA.Population[0].Fitness_Score
    
    if j % 500 == 0:
        print("GA Without Multiproccesing in Generation ",j," Has fitness ",GA.Population[0].Fitness_Score)
        GA.Population[0].visualize()
        GA.Population[0].PinsInSamePlaceCheck()
        for term in Terminals:
            if term.id not in GA.Population[0].chromosome:
                print("terminal with id ",term.id," doestn exist ")


    if counter == 500:
        print("Stopped at Generation", j)
        print("Fitness Score is", GA.Population[0].Fitness_Score)
        MaxNumberOfGenerations = j
        break

    for i in range(int(len(GA.Population)/2)):
        children.append(GA.Crossover(j))

    for child in children:
        child[0].generation = j + 1
        child[1].generation = j + 1

        GA.Population.append(child[0])
        GA.Population.append(child[1])

    children.clear()

    GA.SortPopulation()
    if max_score == GA.Population[0].Fitness_Score:
        counter = counter + 1
    else:
        counter = 0

    half = int(len(GA.Population) / 2)
    GA.Population = GA.Population[:half]
toc = time.time()
print("Fitness Score is", GA.Population[0].Fitness_Score)
# print(GA.Stats())
print("Done in {:.4f} seconds".format(toc - tic))


"""
for i in range(len(GA.Population)):
    GA.Population[i].Fitness_Score = kappa[i]
    GA.Population[i].chromosome = kappa2[i]

tic = time.time()
for j in range(2000):
    
    if j % 500 == 0:
        print("GA With Multiproccesing in Generation ",j," Has fitness ",GA.Population[0].Fitness_Score)

    pool = multiprocessing.Pool()
    Children = pool.map(GA.Crossover, range(int(len(GA.Population) / 2)))
    pool.close()

    for child in Children:
        child[0].generation = j + 1
        child[1].generation = j + 1

        GA.Population.append(child[0])
        GA.Population.append(child[1])

        # GA.Mutation(child[0])
        # GA.Mutation(child[1])

    GA.SortPopulation()
    half = int(len(GA.Population) / 2)
    GA.Population = GA.Population[:half]
# print(GA.Stats())
print("Fitness Score is", GA.Population[0].Fitness_Score)

toc = time.time()
print("Done in {:.4f} seconds".format(toc - tic))
"""
