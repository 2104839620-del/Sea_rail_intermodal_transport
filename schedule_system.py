import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import webbrowser
import threading
import socket

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面
st.set_page_config(
    page_title="海铁联运智能调度系统",
    page_icon="🚢",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .url-box {
        background-color: #e7f3ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .url-link {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: none;
    }
    .good-rating {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .medium-rating {
        background-color: #ffc107;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .poor-rating {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


def open_browser():
    """在后台线程中打开浏览器"""
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


# 获取本机IP并显示网址
local_ip = get_local_ip()
local_url = "http://localhost:8501"
network_url = f"http://{local_ip}:8501"



# 显示网址信息
st.markdown('<div class="main-header">🚢 海铁联运智能调度系统</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="url-box">
    <h3>🌐 系统访问地址</h3>
    <p><strong>本地访问:</strong> <a class="url-link" href="{local_url}" target="_blank">{local_url}</a></p>
    <p><strong>网络访问:</strong> <a class="url-link" href="{network_url}" target="_blank">{network_url}</a></p>
    <p><em>💡 浏览器已自动打开，如果未打开请点击上方链接</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


class SmartScheduler:
    def __init__(self):
        # 物理参数
        self.V_big_g = 4
        self.V_small_g = 2
        self.V_big_s = 4
        self.V_small_s = 2
        self.V_load_a = 3
        self.V_empty_a = 5
        self.t_lift = 40
        self.l_bay = 6.1
        self.l_y = 2.44
        self.W = 2.44
        self.dist_yard_rail = 1200

        # 算法参数
        self.pop_size = 50
        self.max_iter = 100

    def generate_task_locations(self, container_count):
        """生成任务位置"""
        max_stacks_per_bay = 6
        layers = 2

        min_bays_needed = math.ceil(container_count / (max_stacks_per_bay * layers))
        bays = max(min_bays_needed + 2, 10)
        stacks = max_stacks_per_bay

        total_capacity = bays * stacks * layers
        while total_capacity < container_count:
            bays += 1
            total_capacity = bays * stacks * layers

        np.random.seed(42)
        bi_list = np.random.randint(0, bays, container_count)
        yi_list = np.random.randint(0, stacks, container_count)

        return bi_list, yi_list, bays, stacks

    def calculate_makespan(self, chromosome, bi_list, yi_list, G, A, S):
        """计算完工时间"""
        task_order = chromosome[0].astype(int)
        g_assign = chromosome[1].astype(int)
        a_assign = chromosome[2].astype(int)
        s_assign = chromosome[3].astype(int)

        g_time = {g: (0, bi_list[0], yi_list[0]) for g in range(G)}
        a_time = {a: (0, bi_list[0], yi_list[0]) for a in range(A)}
        s_time = {s: (0, bi_list[0]) for s in range(S)}

        max_s_finish = 0

        for idx in task_order:
            bi, yi = bi_list[idx], yi_list[idx]
            g, a, s = g_assign[idx], a_assign[idx], s_assign[idx]

            # 场桥时间计算
            prev_g_finish, prev_g_bi, prev_g_yi = g_time[g]
            t_g_big = abs(bi - prev_g_bi) * self.l_bay / self.V_big_g
            t_g_small = abs(yi - prev_g_yi) * self.l_y / self.V_small_g
            t_g_work = t_g_big + t_g_small + 2 * self.t_lift
            g_start = max(prev_g_finish, a_time[a][0])
            g_finish = g_start + t_g_work
            g_time[g] = (g_finish, bi, yi)

            # ART时间计算
            prev_a_finish, prev_a_bi, prev_a_yi = a_time[a]
            t_a_load = self.dist_yard_rail / self.V_load_a
            t_a_empty = self.dist_yard_rail / self.V_empty_a if prev_a_finish != 0 else 0
            a_start = max(g_finish, prev_a_finish + t_a_empty)
            a_finish = a_start + t_a_load
            a_time[a] = (a_finish, bi, yi)

            # 轨道吊时间计算
            prev_s_finish, prev_s_bi = s_time[s]
            t_s_big = abs(bi - prev_s_bi) * self.l_bay / self.V_big_s
            t_s_work = t_s_big + 2 * self.t_lift + 2 * self.W / self.V_small_s
            s_start = max(a_finish, prev_s_finish)
            s_finish = s_start + t_s_work
            s_time[s] = (s_finish, bi)

            if s_finish > max_s_finish:
                max_s_finish = s_finish

        return max_s_finish

    def create_chromosome(self, container_count, G, A, S):
        """创建染色体"""
        task_order = np.random.permutation(container_count)
        g_assign = np.random.randint(0, G, size=container_count)
        a_assign = np.random.randint(0, A, size=container_count)
        s_assign = np.random.randint(0, S, size=container_count)
        return np.vstack([task_order, g_assign, a_assign, s_assign])

    def quick_evaluate_config(self, container_count, G, A, S, bi_list, yi_list, num_samples=10):
        """快速评估配置"""
        total_makespan = 0
        for _ in range(num_samples):
            chromosome = self.create_chromosome(container_count, G, A, S)
            makespan = self.calculate_makespan(chromosome, bi_list, yi_list, G, A, S)
            total_makespan += makespan
        return total_makespan / num_samples

    def improved_discrete_pso(self, container_count, G, A, S, bi_list, yi_list,
                              pop_size=None, max_iter=None, progress_bar=None, status_text=None):
        """改进的离散粒子群算法"""
        if pop_size is None:
            pop_size = self.pop_size
        if max_iter is None:
            max_iter = self.max_iter

        if status_text:
            status_text.text(f"🔄 正在优化配置: 场桥{G}台, ART{A}台, 轨道吊{S}台 (种群:{pop_size}, 迭代:{max_iter})")

        population = [self.create_chromosome(container_count, G, A, S) for _ in range(pop_size)]
        personal_best = [p.copy() for p in population]

        personal_best_makespan = [self.calculate_makespan(p, bi_list, yi_list, G, A, S) for p in population]

        global_best_idx = np.argmin(personal_best_makespan)
        global_best = population[global_best_idx].copy()
        global_best_makespan = personal_best_makespan[global_best_idx]

        makespan_history = [global_best_makespan]

        w_init, w_final = 0.9, 0.4
        c1, c2 = 2.0, 2.0

        for iter in range(max_iter):
            w = w_init - (w_init - w_final) * (iter / max_iter)

            for i in range(pop_size):
                new_position = self.discrete_pso_velocity(
                    population[i], personal_best[i], global_best, w, c1, c2, container_count
                )
                mutation_rate = 0.1 * (1 - iter / max_iter)
                new_position = self.mutate_particle(new_position, container_count, G, A, S, mutation_rate)

                new_makespan = self.calculate_makespan(new_position, bi_list, yi_list, G, A, S)

                if new_makespan < personal_best_makespan[i]:
                    personal_best[i] = new_position.copy()
                    personal_best_makespan[i] = new_makespan

                    if new_makespan < global_best_makespan:
                        global_best = new_position.copy()
                        global_best_makespan = new_makespan

                population[i] = new_position

            makespan_history.append(global_best_makespan)

            if progress_bar:
                progress_bar.progress((iter + 1) / max_iter)

        final_makespan = self.calculate_makespan(global_best, bi_list, yi_list, G, A, S)
        workload = self.get_equipment_workload(global_best, G, A, S)

        return {
            'best_solution': global_best,
            'best_makespan': final_makespan,
            'makespan_history': makespan_history,
            'equipment_workload': workload,
            'pop_size': pop_size,
            'max_iter': max_iter
        }

    def discrete_pso_velocity(self, position, personal_best, global_best, w, c1, c2, container_count):
        """速度更新"""
        new_position = position.copy()
        task_order = position[0].copy()
        pbest_order = personal_best[0]
        gbest_order = global_best[0]

        for i in range(container_count):
            if random.random() < c1 * random.random():
                if task_order[i] != pbest_order[i]:
                    j = np.where(task_order == pbest_order[i])[0][0]
                    task_order[i], task_order[j] = task_order[j], task_order[i]

            if random.random() < c2 * random.random():
                if task_order[i] != gbest_order[i]:
                    j = np.where(task_order == gbest_order[i])[0][0]
                    task_order[i], task_order[j] = task_order[j], task_order[i]

        new_position[0] = task_order

        for i in range(1, 4):
            current_assign = position[i].copy()
            pbest_assign = personal_best[i]
            gbest_assign = global_best[i]

            for j in range(container_count):
                if random.random() < w:
                    continue
                if random.random() < c1:
                    current_assign[j] = pbest_assign[j]
                if random.random() < c2:
                    current_assign[j] = gbest_assign[j]

            new_position[i] = current_assign

        return new_position

    def mutate_particle(self, particle, container_count, G, A, S, mutation_rate=0.05):
        """变异操作"""
        mutated = particle.copy()

        if random.random() < mutation_rate:
            i, j = random.sample(range(container_count), 2)
            mutated[0, i], mutated[0, j] = mutated[0, j], mutated[0, i]

        for k in range(1, 4):
            if random.random() < mutation_rate:
                num_mutations = random.randint(1, max(1, container_count // 10))
                for _ in range(num_mutations):
                    task_idx = random.randint(0, container_count - 1)
                    if k == 1:
                        mutated[k, task_idx] = random.randint(0, G - 1)
                    elif k == 2:
                        mutated[k, task_idx] = random.randint(0, A - 1)
                    else:
                        mutated[k, task_idx] = random.randint(0, S - 1)

        return mutated

    def get_equipment_workload(self, chromosome, G, A, S):
        """获取设备工作量"""
        task_order = chromosome[0].astype(int)
        g_assign = chromosome[1].astype(int)
        a_assign = chromosome[2].astype(int)
        s_assign = chromosome[3].astype(int)

        g_workload = [np.sum(g_assign == g) for g in range(G)]
        a_workload = [np.sum(a_assign == a) for a in range(A)]
        s_workload = [np.sum(s_assign == s) for s in range(S)]

        return {
            'g_workload': g_workload,
            'a_workload': a_workload,
            's_workload': s_workload
        }

    def evaluate_performance_rating(self, makespan_hours, container_count, workload_balance):
        """评估性能评级"""
        if container_count <= 50:
            time_thresholds = [3, 5, 8]
        elif container_count <= 100:
            time_thresholds = [5, 8, 12]
        elif container_count <= 200:
            time_thresholds = [8, 12, 16]
        else:
            time_thresholds = [12, 18, 24]

        if makespan_hours <= time_thresholds[0]:
            time_rating = "优秀"
        elif makespan_hours <= time_thresholds[1]:
            time_rating = "良好"
        elif makespan_hours <= time_thresholds[2]:
            time_rating = "一般"
        else:
            time_rating = "较差"

        if workload_balance <= 3:
            balance_rating = "优秀"
        elif workload_balance <= 6:
            balance_rating = "良好"
        elif workload_balance <= 10:
            balance_rating = "一般"
        else:
            balance_rating = "较差"

        if time_rating == "优秀" and balance_rating in ["优秀", "良好"]:
            overall_rating = "优秀"
        elif time_rating in ["优秀", "良好"] and balance_rating in ["优秀", "良好", "一般"]:
            overall_rating = "良好"
        elif time_rating == "较差" or balance_rating == "较差":
            overall_rating = "较差"
        else:
            overall_rating = "一般"

        return overall_rating, time_rating, balance_rating

    def find_optimal_equipment_config(self, container_count, bi_list, yi_list, progress_callback=None):
        """动态寻找最优设备配置"""
        if container_count <= 50:
            G_range, A_range, S_range = range(1, 5), range(2, 7), range(1, 5)
        elif container_count <= 100:
            G_range, A_range, S_range = range(1, 6), range(3, 9), range(1, 6)
        else:
            G_range, A_range, S_range = range(2, 8), range(4, 12), range(2, 7)

        best_config = None
        best_makespan = float('inf')
        tested_configs = []

        config_strategies = []

        if container_count <= 50:
            base_configs = [(2, 4, 2), (2, 5, 2), (3, 4, 2), (3, 5, 3)]
        elif container_count <= 100:
            base_configs = [(2, 5, 2), (3, 5, 3), (3, 6, 3), (4, 6, 3)]
        else:
            base_configs = [(3, 6, 3), (4, 7, 4), (4, 8, 4), (5, 8, 4)]

        for G, A, S in base_configs:
            if G in G_range and A in A_range and S in S_range:
                config_strategies.append(('基础配置', G, A, S))

        for strategy_name, G, A, S in config_strategies:
            if progress_callback:
                progress_callback(f"测试{strategy_name}: 场桥{G}台, ART{A}台, 轨道吊{S}台")

            makespan = self.quick_evaluate_config(container_count, G, A, S, bi_list, yi_list, 12)

            config_data = {
                'G': G, 'A': A, 'S': S,
                'makespan': makespan,
                'strategy': strategy_name
            }
            tested_configs.append(config_data)

            if makespan < best_makespan:
                best_config = config_data
                best_makespan = makespan

        if best_config:
            base_G, base_A, base_S = best_config['G'], best_config['A'], best_config['S']

            expansion_configs = []
            for dG in [-1, 0, 1]:
                for dA in [-1, 0, 1]:
                    for dS in [-1, 0, 1]:
                        if dG == 0 and dA == 0 and dS == 0:
                            continue
                        G_new, A_new, S_new = base_G + dG, base_A + dA, base_S + dS
                        if (G_new in G_range and A_new in A_range and S_new in S_range and
                                A_new >= max(G_new, S_new)):
                            expansion_configs.append((G_new, A_new, S_new))

            expansion_configs = list(set(expansion_configs))

            for G, A, S in expansion_configs:
                if not any(c['G'] == G and c['A'] == A and c['S'] == S for c in tested_configs):
                    if progress_callback:
                        progress_callback(f"扩展搜索: 场桥{G}台, ART{A}台, 轨道吊{S}台")

                    makespan = self.quick_evaluate_config(container_count, G, A, S, bi_list, yi_list, 10)

                    config_data = {
                        'G': G, 'A': A, 'S': S,
                        'makespan': makespan,
                        'strategy': '扩展搜索'
                    }
                    tested_configs.append(config_data)

                    if makespan < best_makespan:
                        best_config = config_data
                        best_makespan = makespan

        tested_configs.sort(key=lambda x: x['makespan'])

        return best_config, tested_configs


# 初始化调度器
scheduler = SmartScheduler()

# 侧边栏
with st.sidebar:
    st.header("📋 输入参数")

    container_count = st.slider(
        "集装箱数量",
        min_value=10,
        max_value=500,
        value=100,
        help="选择需要调度的集装箱数量"
    )

    st.header("⚙️ 算法参数")
    col1, col2 = st.columns(2)
    with col1:
        pop_size = st.slider("种群规模", 30, 150, 50)
    with col2:
        max_iter = st.slider("迭代次数", 50, 500, 100)

    st.header("🎯 优化目标")
    auto_optimize = st.checkbox("自动持续优化直到获得良好评级", value=True)

    max_optimization_rounds = st.slider("最大优化轮次", 1, 10, 3)

    generate_btn = st.button("🚀 开始优化调度", type="primary", use_container_width=True)

# 主内容区
if generate_btn:
    with st.spinner('正在初始化调度系统...'):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        optimization_history = []

        progress_text.text("📦 生成任务位置...")
        bi_list, yi_list, bays, stacks = scheduler.generate_task_locations(container_count)

        progress_text.text("🔍 搜索最优设备配置...")


        def update_progress(message):
            status_text.text(message)


        config_start_time = time.time()
        best_config, tested_configs = scheduler.find_optimal_equipment_config(
            container_count, bi_list, yi_list, update_progress
        )
        config_time = time.time() - config_start_time

        if best_config is None:
            st.error("❌ 未找到合适的设备配置")
            st.stop()

        G, A, S = best_config['G'], best_config['A'], best_config['S']

        best_overall_results = None
        best_rating = "较差"
        optimization_round = 0

        while optimization_round < max_optimization_rounds:
            optimization_round += 1

            progress_text.text(f"🔄 正在进行第 {optimization_round} 轮优化...")
            progress_bar.progress(0)

            current_pop_size = min(pop_size + optimization_round * 20, 150)
            current_max_iter = min(max_iter + optimization_round * 50, 500)

            optimization_start_time = time.time()
            results = scheduler.improved_discrete_pso(
                container_count, G, A, S, bi_list, yi_list,
                pop_size=current_pop_size,
                max_iter=current_max_iter,
                progress_bar=progress_bar,
                status_text=status_text
            )
            optimization_time = time.time() - optimization_start_time

            workload_balance = np.std(list(results['equipment_workload']['g_workload']) +
                                      list(results['equipment_workload']['a_workload']) +
                                      list(results['equipment_workload']['s_workload']))

            makespan_hours = results['best_makespan'] / 3600
            overall_rating, time_rating, balance_rating = scheduler.evaluate_performance_rating(
                makespan_hours, container_count, workload_balance
            )

            round_info = {
                'round': optimization_round,
                'pop_size': current_pop_size,
                'max_iter': current_max_iter,
                'makespan_hours': makespan_hours,
                'workload_balance': workload_balance,
                'overall_rating': overall_rating,
                'time_rating': time_rating,
                'balance_rating': balance_rating,
                'optimization_time': optimization_time
            }
            optimization_history.append(round_info)

            if best_overall_results is None or overall_rating in ["优秀", "良好"]:
                best_overall_results = results
                best_overall_results.update({
                    'overall_rating': overall_rating,
                    'time_rating': time_rating,
                    'balance_rating': balance_rating,
                    'workload_balance': workload_balance,
                    'optimization_round': optimization_round
                })
                best_rating = overall_rating

            status_text.text(f"第 {optimization_round} 轮完成 - 评级: {overall_rating}")

            if auto_optimize and overall_rating in ["优秀", "良好"]:
                break

        total_time = config_time + sum([h['optimization_time'] for h in optimization_history])

        progress_bar.empty()
        progress_text.text("✅ 优化完成！")
        status_text.text("")

    if best_overall_results:
        results = best_overall_results
        efficiency = container_count / results['best_makespan'] * 3600

        rating_color = {
            "优秀": "good-rating",
            "良好": "good-rating",
            "一般": "medium-rating",
            "较差": "poor-rating"
        }

        st.markdown(f"""
        <div class="{rating_color[results['overall_rating']]}">
            <h2>🎉 优化完成！最终评级: {results['overall_rating']}</h2>
            <p>经过 {results['optimization_round']} 轮优化，获得 {results['overall_rating']} 评级</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("完工时间", f"{results['best_makespan'] / 3600:.2f}小时", f"时间评级: {results['time_rating']}")
        with col2:
            st.metric("作业效率", f"{efficiency:.1f}箱/小时")
        with col3:
            st.metric("最优配置", f"场桥{G}/ART{A}/轨道吊{S}")
        with col4:
            st.metric("均衡度", f"{results['workload_balance']:.2f}", f"均衡评级: {results['balance_rating']}")

        st.markdown("---")

        st.subheader("📈 优化历史记录")
        history_data = []
        for hist in optimization_history:
            history_data.append({
                "轮次": hist['round'],
                "种群规模": hist['pop_size'],
                "迭代次数": hist['max_iter'],
                "完工时间(小时)": f"{hist['makespan_hours']:.2f}",
                "均衡度": f"{hist['workload_balance']:.2f}",
                "综合评级": hist['overall_rating'],
                "计算时间(秒)": f"{hist['optimization_time']:.2f}"
            })

        st.dataframe(history_data, use_container_width=True)

else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🎯 智能持续优化系统

        本系统采用**多轮持续优化策略**，确保获得最佳的调度方案和均衡的设备工作量。

        ### ✨ 核心功能：

        **🌐 便捷访问**
        - 自动浏览器打开
        - 支持多设备访问
        - 一键点击进入

        **🔄 多轮优化机制**
        - 自动进行多轮优化尝试
        - 每轮增加算法参数强度
        - 动态选择最佳结果

        **📊 智能性能评估**
        - 基于作业时间的评级系统
        - 设备工作量均衡度评估
        - 综合性能评级

        **⚡ 强大算法参数**
        - 种群规模最大可调至150
        - 迭代次数最大可调至500
        - 根据作业规模自动调整
        """)

    with col2:
        st.info("""
        **💡 使用指南**

        **访问方式：**
        - 点击上方网址直接访问
        - 支持手机、平板等多设备

        **优化设置：**
        - 种群规模: 50-150
        - 迭代次数: 100-500  
        - 优化轮次: 3-5轮

        **性能目标：**
        - 优秀: 时间短 + 均衡好
        - 良好: 满足作业要求
        """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "海铁联运智能调度系统 © 2024 | 完整功能版本"
    "</div>",
    unsafe_allow_html=True
)
