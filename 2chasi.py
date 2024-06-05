import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

data = {
    '주소': ['서울', '서울', '서울', '서울', '서울', '서울', '서울', '서울', '서울', '서울'],
    '가격': [900, 550, 600, 650, 700, 750, 800, 850, 900, 600]
}

df = pd.DataFrame(data)

def visualize_trend_with_prediction(df, region, future_price):
    region_data = df[df['주소'] == region]

    plt.figure(figsize=(10, 6))
    plt.plot(region_data.index, region_data['가격'], marker='o', linestyle='-', label='실제 가격')
    plt.title(f'{region} house price')
    plt.xlabel('year')
    plt.ylabel('price')
    plt.grid(True)

    # 미래의 가격 예측을 그래프에 추가
    future_year = len(region_data) + 1
    plt.scatter(future_year, future_price, color='red', label='미래 가격 예측')
    
    plt.legend()
    plt.show()

def predict_future_price(df, region):
    region_data = df[df['주소'] == region]

    X = torch.tensor(region_data.index.values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(region_data['가격'].values, dtype=torch.float32).view(-1, 1)

    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 미래의 가격 예측
    future_year = len(region_data) + 1
    future_price = model(torch.tensor([[future_year]], dtype=torch.float32)).item()
    return future_price

region = input("지역을 입력하세요: ")

# 미래의 집값 예측
future_price = predict_future_price(df, region)
print(f"미래의 집값 예측 (다음 해): {future_price:.2f}만원")

# 시각화
visualize_trend_with_prediction(df, region, future_price)
