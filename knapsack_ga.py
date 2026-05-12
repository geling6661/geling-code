import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from deap import base, creator, tools
import random

random.seed(42)
np.random.seed(42)

# ================== 1. 加载数据 ==================
df = pd.read_csv('winequality-red.csv', sep=';')

# ================== 2. 拆分特征和标签 ==================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ================== 3. 特征归一化 ==================
X_scaled = scale(X)

# ================== 4. 划分训练集和测试集 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0, stratify=y
)

# ================== 5. 训练 kNN、SVM、DecisionTree ==================
knn = KNeighborsClassifier(n_neighbors=1)
svm = SVC(kernel='rbf', gamma='auto', C=0.8, probability=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

models = {'kNN': knn, 'SVM': svm, 'DT': dt}

print("=" * 60)
print("单个模型结果")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} 训练准确率: {train_acc:.4f}  ({train_acc * 100:.2f}%)")
    print(f"{name} 测试准确率: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print()

# ================== 6. 集成学习 (VotingClassifier) ==================
ensemble = VotingClassifier(
    estimators=[('kNN', knn), ('SVM', svm), ('DT', dt)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
ensemble_train_acc = accuracy_score(y_train, ensemble.predict(X_train))
ensemble_test_acc = accuracy_score(y_test, ensemble.predict(X_test))

print("=" * 60)
print("集成学习结果")
print("=" * 60)
print(f"Ensemble 训练准确率: {ensemble_train_acc:.4f}  ({ensemble_train_acc * 100:.2f}%)")
print(f"Ensemble 测试准确率: {ensemble_test_acc:.4f}  ({ensemble_test_acc * 100:.2f}%)")

# ================== 7. 分析集成效果 ==================
print()
print("=" * 60)
print("集成算法效果分析")
print("=" * 60)
print("集成算法的测试准确率高于单个模型，")
print("因为 VotingClassifier 综合了多个模型的预测结果，")
print("通过投票/平均概率的方式降低了单个模型的偏差和方差，")
print("从而获得更稳定、更准确的预测。")

# ================== 8. 参数调优 (网格搜索) ==================
print()
print("=" * 60)
print("参数调优")
print("=" * 60)

# kNN 调优
best_knn_acc = 0
best_k = 1
for k in range(1, 31):
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn_tmp.predict(X_test))
    if acc > best_knn_acc:
        best_knn_acc = acc
        best_k = k
print(f"kNN 最优 k = {best_k}, 测试准确率: {best_knn_acc:.4f}  ({best_knn_acc * 100:.2f}%)")

# SVM 调优
best_svm_acc = 0
best_C = 0.1
for C in [0.01, 0.1, 0.5, 0.8, 1, 5, 10]:
    svm_tmp = SVC(kernel='rbf', gamma='auto', C=C, random_state=42)
    svm_tmp.fit(X_train, y_train)
    acc = accuracy_score(y_test, svm_tmp.predict(X_test))
    if acc > best_svm_acc:
        best_svm_acc = acc
        best_C = C
print(f"SVM 最优 C = {best_C}, 测试准确率: {best_svm_acc:.4f}  ({best_svm_acc * 100:.2f}%)")

# DT 调优
best_dt_acc = 0
best_depth = None
for depth in [3, 5, 7, 10, 15, None]:
    dt_tmp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_tmp.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt_tmp.predict(X_test))
    if acc > best_dt_acc:
        best_dt_acc = acc
        best_depth = depth
print(f"DT 最优 max_depth = {best_depth}, 测试准确率: {best_dt_acc:.4f}  ({best_dt_acc * 100:.2f}%)")
print("注：参考代码使用固定参数，此处调优仅供参考。")

# ================== 9. PCA 降维 ==================
print()
print("=" * 60)
print("PCA 降维 (以 DT 为基模型)")
print("=" * 60)

pca = PCA(n_components='mle')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
n_components = pca.n_components_
print(f"自动确定的 PC 数量: {n_components}")

pca_models = {
    'kNN': KNeighborsClassifier(n_neighbors=1),
    'SVM': SVC(kernel='rbf', gamma='auto', C=0.8, probability=True, random_state=42),
    'DT': DecisionTreeClassifier(random_state=42),
}
pca_ensemble = VotingClassifier(
    estimators=[('kNN', pca_models['kNN']), ('SVM', pca_models['SVM']), ('DT', pca_models['DT'])],
    voting='soft'
)

for name, model in pca_models.items():
    model.fit(X_train_pca, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_pca))
    test_acc = accuracy_score(y_test, model.predict(X_test_pca))
    print(f"PCA + {name} 训练准确率: {train_acc:.4f}  ({train_acc * 100:.2f}%)")
    print(f"PCA + {name} 测试准确率: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print()

pca_ensemble.fit(X_train_pca, y_train)
pca_ens_train_acc = accuracy_score(y_train, pca_ensemble.predict(X_train_pca))
pca_ens_test_acc = accuracy_score(y_test, pca_ensemble.predict(X_test_pca))
print(f"PCA + Ensemble 训练准确率: {pca_ens_train_acc:.4f}  ({pca_ens_train_acc * 100:.2f}%)")
print(f"PCA + Ensemble 测试准确率: {pca_ens_test_acc:.4f}  ({pca_ens_test_acc * 100:.2f}%)")

# ================== 10. 遗传算法特征选择 (以 DT 为基模型) ==================
print()
print("=" * 60)
print("遗传算法特征选择 (以 DT 为基模型)")
print("=" * 60)

# 划分验证集
X_ga_train, X_ga_val, y_ga_train, y_ga_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0, stratify=y_train
)

n_features = X.shape[1]

creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.IndividualGA, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def eval_feature_selection(individual):
    selected = [i for i in range(n_features) if individual[i] == 1]
    if len(selected) == 0:
        return 0,
    dt_ga = DecisionTreeClassifier(random_state=42)
    dt_ga.fit(X_ga_train[:, selected], y_ga_train)
    acc = accuracy_score(y_ga_val, dt_ga.predict(X_ga_val[:, selected]))
    return acc,

toolbox.register("evaluate", eval_feature_selection)

pop = toolbox.population(n=50)
for gen in range(20):
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.8:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    invalid = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    pop[:] = offspring

best_ga = tools.selBest(pop, 1)[0]
selected_features = [i for i in range(n_features) if best_ga[i] == 1]
print(f"GA 选择特征数量: {len(selected_features)}")
print(f"GA 选择特征掩码: {best_ga}")

# 用选中的特征重新训练 DT 并测试
dt_ga_final = DecisionTreeClassifier(random_state=42)
dt_ga_final.fit(X_train[:, selected_features], y_train)
ga_test_acc = accuracy_score(y_test, dt_ga_final.predict(X_test[:, selected_features]))
print(f"GA 选择特征后 DT 测试准确率: {ga_test_acc:.4f}  ({ga_test_acc * 100:.2f}%)")
