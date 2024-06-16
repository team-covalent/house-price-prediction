import asyncio

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import inquirer
import zigbag


def visualize_trend_with_prediction(period, price, future_price):
    years = list(range(-period + 1, 1))  # 과거 기간 리스트

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    plt.plot(years, price, marker="o", linestyle="-", label="Current Price")
    plt.title(f"House Price")
    plt.xlabel("Month")
    plt.ylabel("Price (in 만원)")
    plt.grid(True)

    # 예측된 가격
    future_year = 1
    plt.scatter(future_year, future_price, color="red", label="Future Price")

    plt.legend()
    plt.show()


def predict_future_price(period, price):
    months = list(range(-period + 1, 1))  # 과거 기간 리스트 생성
    X = torch.tensor(months, dtype=torch.float32).view(-1, 1)  # 월을 텐서로 변환
    y = torch.tensor(price, dtype=torch.float32).view(-1, 1)  # 집값을 텐서로 변환

    input_size = 1  # 입력 크기
    output_size = 1  # 출력 크기
    model = nn.Linear(input_size, output_size)  # 선형 회귀 모델 정의

    criterion = nn.MSELoss()  # 손실 함수 정의
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 옵티마이저 정의

    num_epochs = 10000  # 학습 수
    for epoch in range(num_epochs):
        model.train()  # 학습 모드 설정
        outputs = model(X)  # 예측 값 생성
        loss = criterion(outputs, y)  # 손실 계산

        optimizer.zero_grad()  # 옵티마이저 기울기 초기화
        loss.backward()  # 역전파 수행
        optimizer.step()  # 옵티마이저 스텝 진행

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()  # 평가 모드로 설정
    future_year = torch.tensor([[1]], dtype=torch.float32)  # 미래 년도 설정
    future_price = model(future_year).item()  # 미래 가격 예측
    return future_price


async def main():
    print("집값 예측 프로그램")
    client = zigbag.ZigBagClient()
    answers = inquirer.prompt(
        [
            inquirer.List(
                "service_type",
                message="무슨 종류의 집을 검색하시겠습니까?",
                choices=["아파트"],
            ),
        ]
    )
    answers["query"] = input("[?] 원하시는 지역명, 지하철, 아파트명을 입력해주세요: ")
    data = await client.search(answers["query"], answers["service_type"])
    print("검색 결과 (주변 시설물)")
    for not_apt, number in zip(data, range(len(data))):
        if not_apt["type"] != "apartment":
            print(f"[{number}] <{not_apt['type']}> {not_apt['name']}, {not_apt['hint']}")

    answers = inquirer.prompt(
        [
            inquirer.List(
                "room",
                message="원하시는 아파트를 선택해주세요",
                choices=[apt["name"] + ", " + apt["description"] for apt in data if apt["type"] == "apartment"],
            )
        ]
    )
    room_id, region = None, None
    for apt, number in zip(data, range(len(data))):
        if apt["type"] == "apartment" and apt["name"] == answers["room"].split(",")[0]:
            room_id, region = apt["id"], apt["_source"]["local2"]
            break

    prices_data = await client.get_apart_info(room_id)
    price = [int(each_price_data["price"]) for each_price_data in prices_data]
    future_price = predict_future_price(len(price), price)
    print(f"미래의 집값 예측 (1년 후): {future_price:.2f}만원")
    visualize_trend_with_prediction(len(price), price, future_price)


asyncio.run(main())

