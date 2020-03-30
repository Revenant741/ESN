
import random

from deap import base
from deap import creator
from deap import tools

class Evolve(object):
    def __init__(self, bit_size=100, pop_size=300):
        self.bit_size = bit_size
        self.pop_size = pop_size
        self.CXPB = 0.5
        self.MUTPB = 0.2
        self.NGEN = 40

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        self.toolbox = toolbox
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.bit_size)
        toolbox.register("populatioin", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        self.pop = tools.population(self.pop_size)

    def evaluate(self):
        return sum(self.pop)

    def __call__(self):
        
        toolbox = self.toolbox

        pop = self.pop
        
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  %i の個体を評価" % len(pop))

        for g in range(NGEN):
            print("-- %i 世代 --" % g)

            ##############
            # 選択
            ##############
            # 次世代の個体群を選択
            offspring = toolbox.select(pop, len(pop))
            # 個体群のクローンを生成
            offspring = list(map(toolbox.clone, offspring))

            # 選択した個体群に交差と突然変異を適応する

            ##############
            # 交叉
            ##############
            # 偶数番目と奇数番目の個体を取り出して交差
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    toolbox.mate(child1, child2)
                    # 交叉された個体の適合度を削除する
                    del child1.fitness.values
                    del child2.fitness.values

            ##############
            # 変異
            ##############
            for mutant in offspring:
                if random.random() < self.MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 適合度が計算されていない個体を集めて適合度を計算
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  %i の個体を評価" % len(invalid_ind))

            # 次世代群をoffspringにする
            pop[:] = offspring

            # すべての個体の適合度を配列にする
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        print("-- 進化終了 --")

        best_ind = tools.selBest(pop, 1)[0]
        print("最も優れていた個体: %s, %s" % (best_ind, best_ind.fitness.values))

evolve = Evolve()
evolve()
