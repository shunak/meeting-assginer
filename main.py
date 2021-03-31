# 会議のスケジュール    
schedules = {
    "meeting1": ["10:00", "13:00"],
    "meeting2": ["10:00", "12:00"],
    "meeting3": ["10:00", "11:00"],
    "meeting4": ["11:00", "13:00"],
    "meeting5": ["11:00", "12:00"],
    "meeting6": ["11:00", "15:00"],
    "meeting7": ["12:00", "16:00"],
    "meeting8": ["12:00", "15:00"],
    "meeting9": ["13:00", "15:00"],
    "meeting10": ["13:00", "14:00"],
    "meeting11": ["14:00", "17:00"],
    "meeting12": ["15:00", "19:00"],
    "meeting13": ["15:00", "17:00"],
    "meeting14": ["15:00", "16:00"],
    "meeting15": ["16:00", "18:00"],
    "meeting16": ["16:00", "18:00"],
    "meeting17": ["17:00", "19:00"],
    "meeting18": ["17:00", "18:00"],
    "meeting19": ["18:00", "19:00"],
} 

# 会議の数
Nm = len(schedules)    

# 会議室の数
Nr = 8


# 時刻を時間単位の数値に変換
def time2num(time: str):
    h, m = map(float, time.split(":"))
    return h + m / 60

# 2つの会議時間に重なりがあるかをチェック
def check_overlap(time_slot1, time_slot2):
    start1, end1 = map(time2num, time_slot1)
    start2, end2 = map(time2num, time_slot2)
    
    return start1 < end2 and start2 < end1



import itertools

# 会議名のリストを取得
mtg_names = list(schedules.keys())

# 会議室名とインデックスの辞書を作成
mtg_name2idx = {mtg_names[i]: i for i in range(Nm)}

# スケジュールの重なりがある会議のインデックスをタプルで格納
overlaps = []
for mtg1, mtg2 in itertools.combinations(mtg_names, 2):
    if check_overlap(schedules[mtg1], schedules[mtg2]):
        overlaps.append(tuple(sorted([mtg_name2idx[mtg1], mtg_name2idx[mtg2]])))




from amplify import gen_symbols, BinaryPoly

# 決定変数を (Nm x Nr) の配列の形式で定義
q = gen_symbols(BinaryPoly, Nm, Nr)



from amplify import sum_poly
from amplify.constraint import equal_to

# 一つの会議に一つの会議室を割り当てるための one-hot 制約
# equal_to 関数と sum_poly 関数を用いて書けます
room_constraints = sum([equal_to(sum_poly(Nr, lambda r: q[i][r]), 1) for i in range(Nm)])




from amplify import sum_poly
from amplify.constraint import penalty

# overlaps内の全ての (i, j) で、q[i][r] * q[j][r] = 0 の制約条件を課す 
overlap_constraints = sum([penalty(q[i][r] * q[j][r]) for (i, j) in overlaps for r in range(Nr)])




model = room_constraints + overlap_constraints



from amplify import Solver
from amplify.client import FixstarsClient

# クライアントを設定
client = FixstarsClient()
#client.token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ローカル環境で使用する場合は、Amplify AEのアクセストークンを入力してください
client.parameters.timeout = 1000  # タイムアウト1秒

# ソルバーを設定
solver = Solver(client)
# 問題を解く
result = solver.solve(model)

# result が空の場合、制約条件を満たす解が得られなかったことを示す
if len(result) == 0:
    raise RuntimeError("Given constraint conditions are not satisfied")




from amplify import decode_solution
import numpy as np 

# 求めた解を元の変数に代入
values = result[0].values
solution = np.array(decode_solution(q, values))

# 各会議がどの会議室に割り当てられるかを読み取る
room_assignment = list(zip(*np.where(solution == 1)))




# # 
# 会議室割り当てを可視化
#
%matplotlib inline
def plot_mtg_schedule(num_rooms, room_assignment):
    import matplotlib.pyplot as plt

    room_names = ['Room ' + chr(65 + i) for i in range(num_rooms)]

    cmap = plt.get_cmap("coolwarm", num_rooms)
    colors = [cmap(i) for i in range(num_rooms)]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    for mtg_idx, room in room_assignment:
        mtg_name = mtg_names[mtg_idx]
        start = time2num(schedules[mtg_name][0])
        end = time2num(schedules[mtg_name][1])

        plt.fill_between([room + 0.55, room + 1.45], [start, start], [end, end], 
                        edgecolor='black', 
                        linewidth=3.0,
                        facecolor=colors[room] 
                        )
        plt.text(room + 0.6, start + 0.1, f'{schedules[mtg_name][0]}', va='top', fontsize=10)
        plt.text(room + 1.0, (start + end) * 0.5, mtg_name, ha='center', va='center', fontsize=15)

# Set First Axis
ax1.yaxis.grid()
ax1.set_xlim(0.5, len(room_names) + 0.5)
ax1.set_ylim(19.1, 7.9)
ax1.set_xticks(range(1, len(room_names) + 1))
ax1.set_xticklabels(room_names)
ax1.set_ylabel('Time')

# Set Second Axis
ax2 = ax1.twiny().twinx()
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(room_names)
ax2.set_ylabel('Time')

plt.show()


plot_mtg_schedule(num_rooms=Nr, room_assignment=room_assignment)


