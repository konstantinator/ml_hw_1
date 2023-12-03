# ml_hw_1

В ходе работы над ДЗ было получена масса плохих ощущений от плохой подготовительной работы авторами
Неточности, разночтения ошибки, всё что мог и на что было желание пометил в ноутбуке ресёрча

Что сделано:
1) проведено EDA
2) обучено несколько моделей
3) проведены самые разнообразные преобразования над фичами
4) построен пайплайн для быстрого использования модели в инференсе
5) построен веб сервис

В результате были получены хорошие метрики R2 0.9195 на тесте

Однако показатели бизнес метрики оставляют желать лучшего - всего 0.6

Возможно это связано с тем, что я логарифмировал целевую переменную, чтобы получить распределение 
более похожее на нормальное и линейной регрессии было бы легче учиться.

Мл модель была завёрнута в веб сервис

/predict_item
![0](https://github.com/konstantinator/ml_hw_1/blob/main/img/0.png)

/predict_items
![1](https://github.com/konstantinator/ml_hw_1/blob/main/img/1.png)

