import networkx as nx
from ACO_for_RL import run_ant_colony_dynamic
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple
from SMA_algorithm import SlimeMouldAlgorithm
import copy
import time
import pandas as pd


GRID_SIZE = (50, 50)
START = (0, 0)
GOAL = (49, 49)
OBSTACLES = {
    (1,2), (1,3), (6,5),
    (7,2), (8,2), (8,3),
    (0,7), (1,7), (2,7), (3,7), (4,7),
    (6,8), (15,15), (20,15), (18,30), (28,35)
}
CYCLE_OBSTACLES = [(10, 5), (10, 7), (12, 6), (14, 8), (16, 10),
                   (16, 11), (16, 12), (20, 20), (20, 30), (20, 40),
                   (15, 20), (15, 30), (15, 40), (15, 41), (15, 42),
                   (22, 15), (22, 16), (12, 35), (13, 20), (16, 17),
                   (16, 18), (15, 13), (25, 35), (25, 38), (31, 45),
                   (31, 46), (31, 47), (31, 48), (44, 10), (44, 11),
                   (44, 12), (18, 33), (19, 40), (15, 27), (38, 44),
                   (38, 45), (21, 49), (18, 17), (5, 30), (27, 10)]
DYNAMIC_CHANGE_EP = 500
for i in range(35, 41):
    OBSTACLES.add((i, 20))
    OBSTACLES.add((15, i))
for i in range(15, 30):
    OBSTACLES.add((30, i))
    OBSTACLES.add((i, 10))
for i in range(0, 15):
    OBSTACLES.add((i, 40))
    OBSTACLES.add((45, i))
for i in range(30, 40):
    OBSTACLES.add((i, 48))
for i in range(25, 35):
    OBSTACLES.add((i, 42))
    OBSTACLES.add((20, i))
for j in range(5, 45):
    if j not in (24, 25):
        OBSTACLES.add((25, j))
for i in range(10, 40):
    if i not in (20, 21):
        OBSTACLES.add((i, 15))
for i in range(40, 48):
    OBSTACLES.add((i, 40))


ALPHA = 0.1
GAMMA = 0.995
EPS_START = 0.98
EPS_END = 0.05
EPS_DECAY = 0.99995
EPISODES = 15000
MAX_STEPS = 1000

ACO_COOLDOWN = 200
ACO_THRESHOLD = 0.25
ACO_INFLUENCE = 10
SMA_ONLINE_COOLDOWN = 1200
SMA_ONLINE_THRESHOLD = 0.4

BETA = 0.3
REWARD_GOAL = 150.0
REWARD_STEP = -1.0
REWARD_WALL = -3.0

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def grid_to_graph(grid_size, obstacles):
    """
    Преобразует прямоугольную сетку в неориентированный граф.
    Каждая свободная клетка сетки становится вершиной графа,
    а рёбра соединяют соседние клетки (вверх, вниз, влево, вправо).

    :param grid_size: Размер сетки (rows, cols)
    :param obstacles: Множество координат клеток-препятствий
    :return: Граф NetworkX с весами рёбер
    """
    G = nx.Graph()
    rows, cols = grid_size
    for x in range(rows):
        for y in range(cols):
            if (x, y) in obstacles:
                continue
            node = pos_to_index((x, y))
            G.add_node(node, pos=(x, y))
    for x in range(rows):
        for y in range(cols):
            if (x, y) in obstacles:
                continue
            u = pos_to_index((x, y))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx_pos = (x + dx, y + dy)
                if 0 <= nx_pos[0] < rows and 0 <= nx_pos[1] < cols and nx_pos not in obstacles:
                    v = pos_to_index(nx_pos)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=1.0)
    return G


def in_bounds(pos):
    """
    Проверяет, находится ли позиция внутри границ сетки.

    :param pos: Кортеж (x, y)
    :return: True, если позиция внутри сетки
    """
    x, y = pos
    return 0 <= x < GRID_SIZE[0] and 0 <= y < GRID_SIZE[1]


def neighbors(pos):
    """
    Возвращает допустимых соседей клетки (без выхода за границы и препятствий).

    :param pos: Текущая позиция агента
    :return: Список допустимых соседних позиций
    """
    x, y = pos
    cand = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return [p for p in cand if in_bounds(p) and p not in OBSTACLES]


def update_dynamic_obstacles(episode: int):
    """
    Обновляет динамические препятствия по номеру эпизода.
    Через каждые DYNAMIC_CHANGE_EP эпизодов препятствия из
    CYCLE_OBSTACLES добавляются или удаляются из среды.

    :param episode: Текущий номер эпизода
    :return: True, если препятствия были изменены
    """
    if DYNAMIC_CHANGE_EP <= 0:
        return False
    if episode % DYNAMIC_CHANGE_EP == 0:
        for cell in CYCLE_OBSTACLES:
            if cell in OBSTACLES:
                OBSTACLES.remove(cell)
            else:
                OBSTACLES.add(cell)
            print(f"Динамика: добавлено препятствие {cell} на эпизоде {episode}")
        return True
    return False


def pos_to_index(pos):
    """
    Преобразует координаты клетки в линейный индекс состояния.

    :param pos: Позиция (x, y)
    :return: Индекс состояния
    """
    return pos[0]*GRID_SIZE[1] + pos[1]


def index_to_pos(idx):
    """
    Преобразует линейный индекс состояния обратно в координаты клетки.

    :param idx: Индекс состояния
    :return: Позиция (x, y)
    """
    return idx // GRID_SIZE[1], idx % GRID_SIZE[1]


def manhattan_distance(pos1, pos2):
    """
    Вычисляет манхэттенское расстояние между двумя клетками.

    :param pos1: Первая позиция
    :param pos2: Вторая позиция
    :return: Манхэттенское расстояние
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class GridEnv:
    """
    Класс среды GridWorld для обучения агента с подкреплением.
    Среда представляет собой сетку с препятствиями, стартом и целью.
    """
    def __init__(self):
        self.start = START
        self.goal = GOAL
        self.reset()

    def reset(self):
        """
        Сбрасывает среду в начальное состояние.

        :return: Начальная позиция агента
        """
        self.agent = self.start
        return self.agent

    def step(self, action_pos):
        """
        Выполняет шаг агента в среде.
        Возвращает новое состояние, награду и флаг завершения эпизода.
        Используется shaping-награда на основе изменения расстояния до цели.

        :param action_pos: Следующая позиция агента
        :return: (next_state, reward, done, info)
        """
        state_prev = self.agent
        if action_pos in OBSTACLES or not in_bounds(action_pos):
            reward_base = REWARD_WALL
            done = False
            next_state = self.agent
        else:
            self.agent = action_pos
            next_state = self.agent
            if self.agent == self.goal:
                reward_base = REWARD_GOAL
                done = True
            else:
                reward_base = REWARD_STEP
                done = False

        D_t = manhattan_distance(state_prev, self.goal)
        D_t_plus_1 = manhattan_distance(next_state, self.goal)
        progress_change = D_t - D_t_plus_1
        progress_bonus = BETA * progress_change

        if not done:
            reward_final = reward_base + progress_bonus
        else:
            reward_final = reward_base

        return next_state, reward_final, done, {}

    def get_actions(self, state):
        """
        Возвращает допустимые действия из данного состояния.

        :param state: Текущая позиция
        :return: Список допустимых соседних клеток
        """
        return neighbors(state)


class QLearner:
    """
    Q-learning агент с табличным представлением Q-функции.
    """
    def __init__(self):
        n_states = GRID_SIZE[0]*GRID_SIZE[1]
        n_actions = 4
        self.Q = {s: np.zeros(n_actions) for s in range(n_states)}
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.alpha = ALPHA
        self.gamma = GAMMA

    def get_valid_action_indices(self, state):
        """
        Возвращает индексы допустимых действий для данного состояния.

        :param state: Позиция агента
        :return: Список индексов действий
        """
        valid_idx = []
        s_pos = state
        for i, a in enumerate(self.actions):
            a: Tuple[int, int]
            nxt = (s_pos[0] + a[0], s_pos[1] + a[1])
            if in_bounds(nxt) and nxt not in OBSTACLES:
                valid_idx.append(i)
        return valid_idx

    def select_action(self, state, eps):
        """
        Выбирает действие по ε-жадной стратегии.

        :param state: Текущее состояние
        :param eps: Вероятность случайного действия
        :return: Индекс выбранного действия
        """
        s_idx = pos_to_index(state)
        valid = self.get_valid_action_indices(state)
        if random.random() < eps:
            ai = random.choice(valid)
            return ai
        else:
            qvals = self.Q[s_idx]
            q_valid = {i: qvals[i] for i in valid}
            max_q = max(q_valid.values())
            best_actions = [i for i, q in q_valid.items() if q == max_q]
            return random.choice(best_actions)

    def update(self, s, a_idx, r, s_next):
        """
        Обновляет Q-значение по формуле Q-learning.

        :param s: Текущее состояние
        :param a_idx: Индекс действия
        :param r: Полученная награда
        :param s_next: Следующее состояние
        """
        s_i = pos_to_index(s)
        snext_i = pos_to_index(s_next)
        max_next = np.max(self.Q[snext_i])
        td = r + self.gamma * max_next - self.Q[s_i][a_idx]
        self.Q[s_i][a_idx] += self.alpha * td


def path_nodes_to_positions(path_nodes):
    """
    Преобразует путь в виде индексов графа в координаты сетки.

    :param path_nodes: Список индексов вершин
    :return: Список координат (x, y)
    """
    return [index_to_pos(n) for n in path_nodes]


def prefill_q_with_aco(agent, graph, start_pos, goal_pos, num_iterations=50, num_ants_start=400, influence=ACO_INFLUENCE):
    """
    Использует ACO для предварительного заполнения Q-таблицы.
    Лучший найденный путь усиливает Q-значения агента,
    ускоряя дальнейшее обучение RL.

    :return: Найденный путь в координатах сетки
    """
    start_node = pos_to_index(start_pos)
    end_node = pos_to_index(goal_pos)

    best_paths, best_len = run_ant_colony_dynamic(
        graph,
        num_ants_start=num_ants_start,
        num_iterations=num_iterations,
        start_node=start_node,
        end_node=end_node,
    )

    if not best_paths:
        print("ACO не нашёл путей")
        return

    best_path_nodes = best_paths[0]
    path_positions = path_nodes_to_positions(best_path_nodes)

    path_len = len(best_path_nodes) - 1 if len(best_path_nodes) > 1 else 1

    for i in range(len(path_positions) - 1):
        s = path_positions[i]
        s_idx = pos_to_index(s)
        nxt = path_positions[i + 1]

        dx = nxt[0] - s[0]
        dy = nxt[1] - s[1]
        try:
            a_idx = agent.actions.index((dx, dy))
        except ValueError:
            continue

        steps_to_goal = path_len - i
        heuristic_value = influence * (1 + 10.0 / steps_to_goal)

        current_q = agent.Q[s_idx][a_idx]
        agent.Q[s_idx][a_idx] = max(current_q, heuristic_value)

    print(f"ACO путь применен к Q-таблице (длина {path_len}). Influence={influence}")
    return path_positions


def evaluate_rl_params(params, env_template, agent_template, current_eps):
    """
    Оценивает параметры RL (alpha, gamma) на серии тестовых эпизодов.
    Используется как целевая функция для SMA.

    :param params: [alpha, gamma]
    :return: Отрицательная оценка качества
    """
    alpha, gamma = params

    runs = 3
    total_score = 0
    test_episodes = 25
    test_eps = min(current_eps, 0.2)

    for _ in range(runs):
        env = copy.deepcopy(env_template)
        agent = copy.deepcopy(agent_template)

        agent.alpha = alpha
        agent.gamma = gamma

        accumulated_reward = 0
        dist_penalty = 0
        success_count = 0

        for ep in range(test_episodes):
            state = env.reset()
            done = False
            steps = 0
            ep_reward = 0

            while not done and steps < 600:
                a_idx = agent.select_action(state, test_eps)
                a = agent.actions[a_idx]
                next_pos = (state[0] + a[0], state[1] + a[1])
                next_state, r, done, _ = env.step(next_pos)
                agent.update(state, a_idx, r, next_state)

                state = next_state
                ep_reward += r
                steps += 1

            if done:
                success_count += 1

            dist = abs(state[0] - env.goal[0]) + abs(state[1] - env.goal[1])
            dist_penalty += dist
            accumulated_reward += ep_reward

        score = (success_count * 100) + (accumulated_reward / test_episodes) - (dist_penalty / test_episodes * 0.5)
        total_score += score

    return -(total_score / runs)


def optimize_rl_params_with_sma(env, agent, current_eps, pop_size=10, iter_max=10, seed=42):
    """
    Оптимизирует параметры Q-learning с помощью SMA.

    :return: Лучшие параметры и их оценка
    """
    print(f"\nSMA: Запуск оптимизации (Base Eps={current_eps:.3f})...")

    sma = SlimeMouldAlgorithm(pop_size=pop_size, iter_max=iter_max, seed=seed)

    lb = [0.09, 0.990]
    ub = [0.15, 0.999]

    eval_fn = lambda params: evaluate_rl_params(params, env, agent, current_eps)
    best_x, best_f, _ = sma.optimize(eval_fn, lb, ub, dim=2)

    print(f"SMA Результат: ALPHA={best_x[0]:.3f}, GAMMA={best_x[1]:.3f} (Score: {-best_f:.2f})")
    return best_x, best_f


def train_hybrid(enable_aco=False, enable_sma=False, enable_sma_online=False):
    """
    Обучает агента в гибридной схеме RL + ACO + SMA.

    :return: Обученный агент и история обучения
    """
    global ALPHA, GAMMA, EPS_DECAY
    last_aco_ep = -ACO_COOLDOWN
    last_sma_ep = -SMA_ONLINE_COOLDOWN

    env = GridEnv()
    agent = QLearner()
    graph = grid_to_graph(GRID_SIZE, OBSTACLES)

    if enable_sma:
        best_params, _ = optimize_rl_params_with_sma(
            copy.deepcopy(env),
            copy.deepcopy(agent),
            current_eps=EPS_START,
            pop_size=10,
            iter_max=10
        )
        ALPHA, GAMMA = best_params
        agent.alpha = ALPHA
        agent.gamma = GAMMA

    if enable_aco:
        aco_path_initial = prefill_q_with_aco(agent, graph, START, GOAL)
    else:
        aco_path_initial = None

    eps = EPS_START
    rewards_history = []
    success_history = []

    print("\nНачало обучения")

    for ep in range(1, EPISODES + 1):
        update_dynamic_obstacles(ep)
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            a_idx = agent.select_action(state, eps)
            a = agent.actions[a_idx]
            next_pos = (state[0] + a[0], state[1] + a[1])
            next_state, r, done, _ = env.step(next_pos)
            agent.update(state, a_idx, r, next_state)
            state = next_state
            total_reward += r
            steps += 1

        rewards_history.append(total_reward)
        success_history.append(1 if state == env.goal else 0)

        if (enable_aco and (ep - last_aco_ep >= ACO_COOLDOWN) and (len(success_history) >= 50) and
                (np.mean(success_history[-50:]) < ACO_THRESHOLD)):
            print(f"Ep {ep}: Низкий успех, вызов ACO для перестроения маршрута")
            graph = grid_to_graph(GRID_SIZE, OBSTACLES)
            prefill_q_with_aco(agent, graph, START, GOAL, influence=ACO_INFLUENCE)
            last_aco_ep = ep

        if enable_sma_online and (ep - last_sma_ep >= SMA_ONLINE_COOLDOWN) and (
                len(success_history) >= 100 and np.mean(success_history[-100:]) < SMA_ONLINE_THRESHOLD):
            print(f"\nEp {ep}: Запуск SMA оптимизации")
            current_params = [agent.alpha, agent.gamma]
            current_score = evaluate_rl_params(current_params, env, agent, current_eps=eps)
            print(f"Текущая оценка параметров ({agent.alpha:.3f}, {agent.gamma:.3f}): {-current_score:.2f}")
            best_params, best_score = optimize_rl_params_with_sma(
                copy.deepcopy(env),
                copy.deepcopy(agent),
                current_eps=eps,
                pop_size=10,
                iter_max=10
            )
            if best_score < current_score:
                ALPHA, GAMMA = best_params
                agent.alpha = ALPHA
                agent.gamma = GAMMA
                print(f"Новые параметры: Alpha: {ALPHA:.3f}, Gamma: {GAMMA:.3f}")
            else:
                print(f"SMA не нашел параметров лучше текущих.")

            last_sma_ep = ep

        eps = max(EPS_END, eps * EPS_DECAY)

        if ep % 50 == 0 or ep == 1:
            succ_rate = np.mean(success_history[-50:]) * 100
            print(
                f"Ep {ep:4d} | reward {np.mean(rewards_history[-50:]):6.1f} | eps {eps:.3f} | succ(last50) {succ_rate:.1f}%")

    return agent, rewards_history, success_history, aco_path_initial


def extract_greedy_policy(agent):
    """
    Извлекает жадную политику из обученной Q-таблицы.

    :return: Словарь {позиция: действие}
    """
    policy = {}
    for idx in range(GRID_SIZE[0] * GRID_SIZE[1]):
        pos = index_to_pos(idx)
        if pos in OBSTACLES:
            policy[pos] = None
            continue

        valid_indices = agent.get_valid_action_indices(pos)
        if not valid_indices:
            policy[pos] = None
            continue

        qvals = agent.Q[idx]
        q_valid = {i: qvals[i] for i in valid_indices}
        max_q = max(q_valid.values())
        best_actions_indices = [i for i, q in q_valid.items() if q == max_q]
        best_action_index = random.choice(best_actions_indices)

        policy[pos] = agent.actions[best_action_index]
    return policy


def follow_policy(env, policy, max_steps=100):
    """
    Следует заданной политике в среде и возвращает путь.

    :return: Список позиций пути
    """
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        action = policy.get(state)
        if action is None: break
        next_state, _, done, _ = env.step((state[0] + action[0], state[1] + action[1]))
        path.append(next_state)
        state = next_state
        if done:
            break
    return path


def plot_results(rewards, successes, path=None):
    """
    Строит графики награды и успешности обучения.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    axs[0].plot(rewards)
    axs[0].set_title("Reward per episode")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total reward")

    axs[1].plot(np.convolve(successes, np.ones(50)/50, mode='valid'))
    axs[1].set_title("Success rate (rolling 50 eps)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Success rate")

    plt.tight_layout()
    plt.show()

    if path:
        grid = np.zeros(GRID_SIZE)
        for (x,y) in OBSTACLES:
            grid[x,y] = -1
        for (x,y) in path:
            grid[x,y] = 0.5
        grid[START] = 0.2
        grid[GOAL] = 1.0

        plt.figure(figsize=(4,4))
        plt.imshow(grid.T, origin='lower', cmap='viridis', vmin=-1, vmax=1)
        plt.title("Path")
        plt.show()


def plot_paths_rl_aco(rl_path=None, aco_path=None):
    """
    Отображает пути, найденные RL и ACO, на одной карте.
    """
    grid = np.zeros(GRID_SIZE)
    for (x, y) in OBSTACLES:
        grid[x, y] = -1
    if rl_path:
        for (x, y) in rl_path:
            grid[x, y] = 0.5
    if aco_path:
        for (x, y) in aco_path:
            grid[x, y] = 0.7
    grid[START] = 0.2
    grid[GOAL] = 1.0
    plt.figure(figsize=(4,4))
    plt.imshow(grid.T, origin='lower', cmap='viridis', vmin=-1, vmax=1)
    plt.scatter([START[1]], [START[0]], c="green", s=100, label="Start")
    plt.scatter([GOAL[1]], [GOAL[0]], c="red", s=100, label="Goal")
    plt.legend()
    plt.title("Paths RL & ACO")
    plt.show()


def plot_results_comparison(rewards_data, successes_data, run_name):
    """
    Сравнивает несколько алгоритмов по наградам и успешности.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for name, rewards in rewards_data.items():
        smoothed_rewards = np.convolve(rewards, np.ones(50) / 50, mode='valid')
        axs[0].plot(smoothed_rewards, label=f'{name}')
    axs[0].set_title(f"Сравнение наград ({run_name})")
    axs[0].set_xlabel("Эпизод")
    axs[0].set_ylabel("Сглаженная суммарная награда")
    axs[0].legend()

    for name, successes in successes_data.items():
        smoothed_successes = np.convolve(successes, np.ones(50) / 50, mode='valid')
        axs[1].plot(smoothed_successes, label=f'{name}')
    axs[1].set_title(f"Сравнение успешности (скользящее среднее 50 эпизодов) ({run_name})")
    axs[1].set_xlabel("Эпизод")
    axs[1].set_ylabel("Успешность")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(rewards_data):
    """
    Строит сглаженные кривые награды с доверительным интервалом.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    window = 100

    colors = {'1. Базовый RL': 'gray', '2. RL + ACO': 'blue', '3. Полный Гибрид': 'red'}

    for name, data in rewards_data.items():
        series = pd.Series(data)
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()

        x = range(len(mean))
        ax.plot(x, mean, label=name, color=colors.get(name), linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=colors.get(name), alpha=0.15)

    ax.set_title("Динамика Награды (Mean ± Std)", fontsize=14)
    ax.set_xlabel("Эпизод", fontsize=12)
    ax.set_ylabel("Суммарная награда", fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5, label='Threshold 0.9')

    plt.tight_layout()
    plt.show()


def first_100_success(success_history, window=50):
    """
    Определяет эпизод достижения 100% успешности (по скользящему среднему).

    :return: Номер эпизода или None
    """
    series = pd.Series(success_history)
    smoothed = series.rolling(window=window).mean()

    for i in range(len(smoothed)):
        if smoothed.iloc[i] == 1.0:
            return i + window
    return None


def print_100_percent_convergence(successes_data):
    """
    Печатает таблицу с эпизодом первой полной сходимости для каждого алгоритма.
    """
    print("\n" + "=" * 80)
    print(f"{'Алгоритм':<30} | {'Эпизод первого 100% успеха':<50}")
    print("-" * 80)

    for name, data in successes_data.items():
        ep_100 = first_100_success(data)

        if ep_100 is None:
            conv_str = "Не достигнуто (за 15000)"
        else:
            conv_str = f"{ep_100} эпизод"

        print(f"{name:<30} | {conv_str:<50}")
    print("=" * 80)


def greedy_test(agent, env, n_runs=100, max_steps=300, verbose=True):
    """
    Тестирует агента в жадном режиме без исследования.

    :return: Метрики успешности, шагов и награды
    """
    succ = 0
    steps_list = []
    rewards_list = []
    start_time = time.time()

    for _ in range(n_runs):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            s_idx = pos_to_index(state)
            valid_indices = agent.get_valid_action_indices(state)

            if not valid_indices:
                break

            qvals = agent.Q[s_idx]
            q_valid = {i: qvals[i] for i in valid_indices}

            max_q = max(q_valid.values())
            best_actions_indices = [i for i, q in q_valid.items() if q == max_q]
            best_action_index = random.choice(best_actions_indices)

            a = agent.actions[best_action_index]
            next_state, r, done, _ = env.step((state[0] + a[0], state[1] + a[1]))

            total_reward += r
            steps += 1
            state = next_state
            if done: break

        rewards_list.append(total_reward)
        steps_list.append(steps)
        if state == env.goal:
            succ += 1

    duration = time.time() - start_time
    succ_rate = succ / n_runs

    if verbose:
        print(
            f"Greedy test: runs={n_runs} | succ_rate={succ_rate:.3f} | mean_steps={np.mean(steps_list):.1f} | mean_reward={np.mean(rewards_list):.1f} | time={duration:.2f}s")

    return {"succ_rate": succ_rate, "mean_steps": np.mean(steps_list), "mean_reward": np.mean(rewards_list),
            "duration": duration}


def robustness_test(agent, base_obstacles, modify_fn, env_ctor, n_runs=100):
    """
    Проверяет устойчивость агента к изменению препятствий.

    :return: Метрики greedy-теста
    """
    new_obs = set(base_obstacles)
    modify_fn(new_obs)
    global OBSTACLES
    old_obs = OBSTACLES
    OBSTACLES = new_obs
    env = env_ctor()
    res = greedy_test(agent, env, n_runs=n_runs, max_steps=MAX_STEPS)
    OBSTACLES = old_obs
    return res


def run_experiment(title, enable_aco=False, enable_sma=False, enable_sma_online=False, max_steps=300):
    """
    Запускает полный эксперимент обучения и тестирования алгоритма.

    :return: Словарь с результатами эксперимента
    """
    print(f"\n{title}")

    agent, rewards, successes, aco_path = train_hybrid(
        enable_aco=enable_aco,
        enable_sma=enable_sma,
        enable_sma_online=enable_sma_online
    )

    policy = extract_greedy_policy(agent)
    env = GridEnv()

    path = follow_policy(env, policy, max_steps=max_steps)
    print("\nНайденный путь:", path)

    greedy_metrics = greedy_test(
        agent,
        GridEnv(),
        n_runs=200,
        max_steps=MAX_STEPS
    )

    base_obs = set(OBSTACLES)

    def add_obstacle(obs):
        obs.add((10, 10))
        obs.add((18, 15))
        obs.add((15, 15))
        obs.add((20, 20))

    robustness_metrics = robustness_test(
        agent,
        base_obs,
        add_obstacle,
        GridEnv,
        n_runs=500
    )

    return {
        "agent": agent,
        "rewards": rewards,
        "successes": successes,
        "path": path,
        "greedy": greedy_metrics,
        "robustness": robustness_metrics,
        "aco_path": aco_path
    }


if __name__ == "__main__":

    base_results = run_experiment(
        title="1. Обучение базового RL",
        enable_aco=False,
        enable_sma=False,
        enable_sma_online=False
    )

    aco_results = run_experiment(
        title="2. Обучение RL + ACO",
        enable_aco=True,
        enable_sma=False,
        enable_sma_online=False
    )

    hybrid_results = run_experiment(
        title="3. Полный гибрид (RL + ACO + SMA)",
        enable_aco=True,
        enable_sma=False, # Для корректности сравнения алгоритмы начинают с одинаковыми параметрами
        enable_sma_online=True
    )

    rewards_data = {
        '1. Базовый RL': base_results["rewards"],
        '2. RL + ACO': aco_results["rewards"],
        '3. Полный Гибрид': hybrid_results["rewards"]
    }

    successes_data = {
        '1. Базовый RL': base_results["successes"],
        '2. RL + ACO': aco_results["successes"],
        '3. Полный Гибрид': hybrid_results["successes"]
    }

    plot_results_comparison(
        rewards_data,
        successes_data,
        run_name="Граф 50x50"
    )
    plot_comparison(
        rewards_data
    )

    plot_paths_rl_aco(hybrid_results["path"], hybrid_results["aco_path"])
    print_100_percent_convergence(successes_data)
