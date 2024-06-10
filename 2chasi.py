import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_trend_with_prediction(region, period, price, future_price):
    years = list(range(-period+1, 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, price, marker='o', linestyle='-', label='Current Price')
    plt.title(f'{region} House Price')
    plt.xlabel('Year')
    plt.ylabel('Price (in 만원)')
    plt.grid(True)

    future_year = 1
    plt.scatter(future_year, future_price, color='red', label='Future Price')
    
    plt.legend()
    plt.show()

def predict_future_price(region, period, price):
    years = list(range(-period+1, 1))
    X = torch.tensor(years, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(price, dtype=torch.float32).view(-1, 1)

    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 10000  
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0: 
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    future_year = torch.tensor([[1]], dtype=torch.float32)
    future_price = model(future_year).item()
    return future_price

region = input("지역을 입력하세요: ")
period = int(input("과거 몇년간의 기간을 입력하시겠습니까?: "))
price = [float(input(f"{period - i}년 전 집값을 입력하세요: ")) for i in range(period)]

future_price = predict_future_price(region, period, price)
print(f"미래의 집값 예측 (1년 후): {future_price:.2f}만원")

visualize_trend_with_prediction(region, period, price, future_price)
