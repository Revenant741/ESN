
import reservoir as res
import reservoir_test as res_test

import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import argparse

import torch

def add_arguments(parser):
    parser.add_argument('--pop_size', type=int, default=300, help='size of population')
    parser.add_argument('--cxpb', type=float, default=0.5, help='pass')
    parser.add_argument('--mutpb', type=float, default=0.2, help='pass')
    parser.add_argument('--ngen', type=int, default=40, help='number of generation')

def evaluate(model, res_size, train_loader, test_x, test_y):
    def evaluate_(individual):
        model.weight_res = torch.Tensor(individual).view(res_size, res_size)
        _, accuracys = res_test.train(args, model, train_loader, test_x, test_y)
        return accuracys[-1]
    return evaluate_
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    res_test.add_arguments(parser)
    add_arguments(parser)
    args = parser.parse_args()

    model = res_test.RNN(args)
    train_loader, test_x, test_y = res_test.prepare_dataset(args)

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_bool', random.randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, args.res_size**2)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', evaluate(model, args.res_size, train_loader, test_x, test_y))
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(args.pop_size)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit 

    print("  %i の個体を評価" % len(pop))

    generations = []
    fitnesses = []
    for g in range(args.ngen):
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
            if random.random() < args.CXPB:
                toolbox.mate(child1, child2)
                # 交叉された個体の適合度を削除する
                del child1.fitness.values
                del child2.fitness.values

        ##############
        # 変異
        ##############
        for mutant in offspring:
            if random.random() < args.MUTPB:
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

        best_ind = tools.selBest(pop, 1)[0]
        generations.append(g)
        fitnesses.append(best_ind.fitness.values)

    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.plot(generations, fitnesses)
    plt.savefig('image/ga-esn-mnist.png')

    
