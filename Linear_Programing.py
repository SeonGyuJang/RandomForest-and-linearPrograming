import pandas as pd
from pulp import *

# 데이터 로드
data = pd.read_csv('Pre_data_Marketing.csv')

# 변수 생성
variables = LpVariable.dicts('Segment', (i+1 for i in data.index), cat='Binary')

# 최적화 모델 생성
model = LpProblem('Maximize_Pre', LpMaximize)

# 목적함수 생성
model += lpSum([data['Pre'][i-1] * variables[i] for i in variables])

# 제약조건 생성
model += lpSum([data['MK'][i-1] * variables[i] for i in variables]) <= 30000

# 최적화 모델 풀기
model.solve()
model.writeLP("LP.txt")

# 결과 출력
print('Status:', LpStatus[model.status])
print('Optimal Solution:')
for v in model.variables():
    if v.varValue == 1:
        segment_index = int(v.name.split('_')[1])
        segment_number = segment_index + 1
        print('Index:', segment_number, ', Variable:', v.name, ', Pre:', data['Pre'][segment_index-1], ', MK:', data['MK'][segment_index-1])


selected_segments = []
total_mk_cost = 0
total_expected_profit = 0
for v in model.variables():
    if v.varValue == 1:
        selected_segments.append(v.name)
        segment_index = int(v.name.split('_')[1])
        total_mk_cost += data['MK'][segment_index-1]
        total_expected_profit += data['Pre'][segment_index-1]

print('총 마케팅 비용:', total_mk_cost)
print('총 예상 수익:', total_expected_profit)
