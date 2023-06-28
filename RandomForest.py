import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import math
import joblib

# CSV 파일 불러오기
data = pd.read_csv('data_OG.csv')

# 필요한 열 선택하기
selected_features = ['Education', 'Marital_Status', 'Income','MntPurchases']
data = data[selected_features].dropna()  # 결측치 제거

# 데이터 타입 설정
data['Education'] = data['Education'].astype('category')
data['Marital_Status'] = data['Marital_Status'].astype('category')
data['Income'] = data['Income'].astype(float)
data['MntPurchases'] = data['MntPurchases'].astype(float)
print(data.dtypes)

# 로그 변환 // Feature Engineering
data['Income'] = np.log1p(data['Income'])
data['MntPurchases'] = np.log1p(data['MntPurchases'])

# 특성과 타겟 데이터로 나누기
X = data[['Education', 'Marital_Status', 'Income']]
y = data['MntPurchases']

# 범주형 변수를 더미 변수로 변환
X = pd.get_dummies(X, columns=['Education', 'Marital_Status'])

# 하이퍼 파라메타 찾기
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=7,scoring='neg_mean_squared_error')

grid_search.fit(X, y)


best_params = grid_search.best_params_
best_score = grid_search.best_score_

model = RandomForestRegressor(**best_params)

# 모델 학습
model.fit(X, y)

# 예측 수행
y_pred = model.predict(X).astype(int)  # 예측값을 정수로 변환

# Pre_data에 예측값을 추가하여 저장
Pre_data = X.copy()
Pre_data['Pre'] = y_pred.astype(int)

# Pre_data.csv로 저장
Pre_data.to_csv('Pre_data.csv', index=False)

# MSE 계산
mse = mean_squared_error(y, y_pred)
print("MSE:", mse)

# 최적의 모델 저장
joblib.dump(model, 'OA_AI_model.pkl')

loaded_model = joblib.load('OA_AI_model.pkl')
loaded_model.predict(X) 
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)


# CSV 파일 불러오기
data = pd.read_csv('Pre_data.csv')

# 로그 변환 역으로 되돌리기
data['Pre'] = np.expm1(data['Pre'])
data['Income'] = np.expm1(data['Income'])
# data['MntPurchases'] = np.expm1(data['MntPurchases'])


# 대체된 데이터로 CSV 파일 저장
data.to_csv('Pre_data.csv', index=False)


# CSV 파일 불러오기
data = pd.read_csv('Pre_data.csv')

# 마케팅 비용 계산 함수 정의
def calculate_marketing_cost(row):
    income = row['Income']
    education_2n_cycle = row['Education_2n Cycle']
    education_basic = row['Education_Basic']
    education_master = row['Education_Master']
    education_phd = row['Education_PhD']
    education_graduation = row['Education_Graduation']
    marital_status_absurd = row['Marital_Status_Absurd']
    marital_status_alone = row['Marital_Status_Alone']
    marital_status_divorced = row['Marital_Status_Divorced']
    marital_status_married = row['Marital_Status_Married']
    marital_status_single = row['Marital_Status_Single']
    marital_status_together = row['Marital_Status_Together']
    marital_status_widow = row['Marital_Status_Widow']
    marital_status_yolo = row['Marital_Status_YOLO']

    marketing_cost = (math.ceil(income / 10000) * 100) + 10 * education_2n_cycle + 20 * education_basic + 40 * education_master + 50 * education_phd + \
        30 * education_graduation + 10 * marital_status_absurd + 10 * marital_status_alone + 10 * marital_status_divorced + \
        50 * marital_status_married + 30 * marital_status_single + 10 * marital_status_together + 10 * marital_status_widow + \
        10 * marital_status_yolo 

    return marketing_cost

# "MK" 열에 마케팅 비용 계산 결과 할당
data['MK'] = data.apply(calculate_marketing_cost, axis=1)

# Pre_data_MK_All.csv로 저장
data.to_csv('Pre_data_Marketing.csv', index=False)

# 특성의 중요도 출력
feature_importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'변수': feature_names, '중요도': feature_importances})
importance_df = importance_df.sort_values('중요도', ascending=False)

print("변수의 중요도:")
print(importance_df)