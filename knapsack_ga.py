import random
import numpy as np
from deap import base, creator, tools

# 设置随机种子，保证结果可重复
random.seed(42)
np.random.seed(42)

# ================== 1. 问题定义 ==================
# 随机生成20个货物（重量5-20kg，价值10-50元）
num_items = 20
max_weight = 150

# 生成随机货物
items_weight = [random.randint(5, 20) for _ in range(num_items)]
items_value = [random.randint(10, 50) for _ in range(num_items)]

# 为了与参考结果对比，也可以使用固定数据（取消下面注释）
# items_weight = [14, 20, 20, 5, 14, 11, 7, 6, 15, 10, 15, 20, 17, 10, 14, 19, 15, 16, 13, 15]
# items_value = [17, 26, 41, 33, 23, 46, 44, 12, 11, 20, 36, 26, 31, 16, 17, 22, 43, 10, 32, 26]

print("="*70)
print("货物清单（重量, 价值）:")
print("="*70)
for i, (w, v) in enumerate(zip(items_weight, items_value), 1):
    print(f"货物{i:2d}: 重量={w:2d}kg, 价值={v:2d}元")
print(f"\n货车最大载重: {max_weight}kg")
print(f"货物总数: {num_items}个")

# ================== 2. 遗传算法参数 ==================
POPULATION_SIZE = 100    # 种群大小
GENERATIONS = 10         # 迭代代数
CX_PROB = 0.7           # 交叉概率
MUT_PROB = 0.2          # 变异概率
TOURNAMENT_SIZE = 3      # 锦标赛大小

# ================== 3. 定义适应度函数 ==================
def evalKnapsack(individual):
    """适应度函数：计算总价值，超重则适应度为0"""
    total_weight = sum(items_weight[i] for i in range(num_items) if individual[i] == 1)
    total_value = sum(items_value[i] for i in range(num_items) if individual[i] == 1)

    if total_weight > max_weight:
        return 0,  # 超重惩罚：适应度为0
    else:
        return total_value,  # 最大化总价值

# ================== 4. 创建遗传算法框架 ==================
# 创建适应度类（最大化，权重为正）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 创建个体类（二进制列表）
creator.create("Individual", list, fitness=creator.FitnessMax)

# 注册遗传操作
toolbox = base.Toolbox()
# 属性生成器：0或1
toolbox.register("attr_bool", random.randint, 0, 1)
# 个体生成器：长度为num_items的二进制列表
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, num_items)
# 种群生成器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传算子
toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", tools.cxOnePoint)  # 单点交叉
toolbox.register("mutate", tools.mutFlipBit, indpb=MUT_PROB)  # 位翻转变异
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)  # 锦标赛选择

# ================== 5. 运行遗传算法 ==================
def run_ga():
    # 创建初始种群
    population = toolbox.population(n=POPULATION_SIZE)

    # 统计每代的最佳适应度
    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    best_individual_per_gen = []

    print("\n" + "="*70)
    print("运行遗传算法...")
    print("="*70)

    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 记录初始代
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]
    best_fitness_per_gen.append(best_fitness)
    avg_fitness = sum(ind.fitness.values[0] for ind in population) / len(population)
    avg_fitness_per_gen.append(avg_fitness)
    best_individual_per_gen.append(best_individual[:])

    print(f"第0代: 最佳适应度={best_fitness:.0f}, 平均适应度={avg_fitness:.2f}, 最佳个体重量={sum(items_weight[i] for i in range(num_items) if best_individual[i]==1)}kg")

    # 迭代进化
    for gen in range(1, GENERATIONS + 1):
        # 选择下一代
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 替换种群
        population[:] = offspring

        # 记录统计信息
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        avg_fitness = sum(ind.fitness.values[0] for ind in population) / len(population)

        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(avg_fitness)
        best_individual_per_gen.append(best_individual[:])

        best_weight = sum(items_weight[i] for i in range(num_items) if best_individual[i]==1)
        print(f"第{gen:2d}代: 最佳适应度={best_fitness:.0f}, 平均适应度={avg_fitness:.2f}, 最佳个体重量={best_weight}kg")

    # 获取最终最优解
    best_individual = tools.selBest(population, 1)[0]
    best_value = best_individual.fitness.values[0]
    best_weight = sum(items_weight[i] for i in range(num_items) if best_individual[i] == 1)
    selected_items = [i+1 for i in range(num_items) if best_individual[i] == 1]

    return best_individual, best_weight, best_value, selected_items, best_fitness_per_gen, avg_fitness_per_gen

# 运行算法
best_solution, best_weight, best_value, selected_items, best_fitness, avg_fitness = run_ga()

# ================== 6. 输出结果 ==================
print("\n" + "="*70)
print("最优装载方案:")
print("="*70)
print(f"总重量: {best_weight}kg / {max_weight}kg")
print(f"总价值: {best_value}元")
print(f"所选货物编号: {selected_items}")
print(f"所选货物数量: {len(selected_items)}个")

# 输出详细货物列表
print("\n详细货物清单:")
print("-"*70)
print(f"{'货物编号':<8} {'重量(kg)':<10} {'价值(元)':<10} {'价值/重量':<10}")
print("-"*70)
total_w = 0
total_v = 0
for idx in selected_items:
    w = items_weight[idx-1]
    v = items_value[idx-1]
    total_w += w
    total_v += v
    print(f"货物{idx:<7d} {w:<10d} {v:<10d} {v/w:<10.2f}")
print("-"*70)
print(f"{'总计':<8} {total_w:<10d} {total_v:<10d}")

# ================== 7. 参数影响分析 ==================
print("\n" + "="*70)
print("参数影响分析")
print("="*70)

# 测试不同参数组合
param_tests = [
    {"name": "默认参数", "pop": 100, "gen": 10, "cx": 0.7, "mut": 0.2, "tour": 3},
    {"name": "大种群", "pop": 200, "gen": 10, "cx": 0.7, "mut": 0.2, "tour": 3},
    {"name": "更多代数", "pop": 100, "gen": 20, "cx": 0.7, "mut": 0.2, "tour": 3},
    {"name": "高交叉率", "pop": 100, "gen": 10, "cx": 0.9, "mut": 0.2, "tour": 3},
    {"name": "低交叉率", "pop": 100, "gen": 10, "cx": 0.5, "mut": 0.2, "tour": 3},
    {"name": "高变异率", "pop": 100, "gen": 10, "cx": 0.7, "mut": 0.4, "tour": 3},
    {"name": "低变异率", "pop": 100, "gen": 10, "cx": 0.7, "mut": 0.1, "tour": 3},
    {"name": "大锦标赛", "pop": 100, "gen": 10, "cx": 0.7, "mut": 0.2, "tour": 5},
    {"name": "小锦标赛", "pop": 100, "gen": 10, "cx": 0.7, "mut": 0.2, "tour": 2},
]

print("\n不同参数下的最优解对比:")
print("-"*100)
print(f"{'参数配置':<12} {'种群':<6} {'代数':<6} {'交叉率':<6} {'变异率':<6} {'锦标赛':<6} {'最优价值':<10} {'最优重量':<10}")
print("-"*100)

results = []
for test in param_tests:
    # 重新创建遗传框架
    creator.FitnessMax = None
    creator.Individual = None
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=test["mut"])
    toolbox.register("select", tools.selTournament, tournsize=test["tour"])

    # 运行算法
    population = toolbox.population(n=test["pop"])
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(test["gen"]):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < test["cx"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < test["mut"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    best = tools.selBest(population, 1)[0]
    best_val = best.fitness.values[0]
    best_w = sum(items_weight[i] for i in range(num_items) if best[i]==1)

    results.append((test["name"], best_val, best_w))
    print(f"{test['name']:<12} {test['pop']:<6} {test['gen']:<6} {test['cx']:<6.1f} {test['mut']:<6.1f} {test['tour']:<6} {best_val:<10} {best_w:<10}")

print("-"*100)

# ================== 8. 输出分析结论 ==================
print("\n" + "="*70)
print("参数影响分析结论")
print("="*70)
print("""
1. 种群大小影响:
   - 种群越大(200)，搜索空间更广，更容易找到全局最优解
   - 但计算时间增加，收敛速度可能变慢

2. 迭代代数影响:
   - 代数越多(20)，算法有更多时间优化，结果通常更好
   - 但可能过早收敛到局部最优

3. 交叉概率影响:
   - 高交叉率(0.9)有助于探索新解空间
   - 低交叉率(0.5)保留更多原有优良基因

4. 变异概率影响:
   - 高变异率(0.4)增加多样性，避免早熟收敛
   - 低变异率(0.1)收敛更快，但可能陷入局部最优

5. 锦标赛大小影响:
   - 大锦标赛(5)选择压力大，收敛快但可能丢失多样性
   - 小锦标赛(2)保持多样性，但收敛慢

6. 最优参数建议:
   - 种群: 100-200
   - 代数: 10-20
   - 交叉率: 0.7-0.9
   - 变异率: 0.1-0.3
   - 锦标赛大小: 3
""")

# 输出遗传算法流程说明
print("\n" + "="*70)
print("遗传算法流程说明")
print("="*70)
print("""
1. 初始化:
   - 随机生成100个个体（二进制串，长度20）
   - 每个个体代表一种货物选择方案（1=选，0=不选）

2. 适应度评估:
   - 计算每个个体的总重量和总价值
   - 若总重量 > 150kg，适应度设为0（惩罚）
   - 否则适应度 = 总价值（最大化目标）

3. 选择（锦标赛选择）:
   - 随机选择3个个体
   - 保留适应度最高的个体进入下一代
   - 重复直到填满种群

4. 交叉（单点交叉，概率0.7）:
   - 随机选择交叉点
   - 交换两个个体交叉点后的基因
   - 产生两个新个体

5. 变异（位翻转变异，概率0.2）:
   - 每个基因位有20%的概率翻转（0变1，1变0）
   - 引入新基因，增加多样性

6. 迭代:
   - 重复选择、交叉、变异10代
   - 每代保留最优解
   - 最终输出全局最优解
""")
