# Bitcoin price prediction with LSTM Network

This project was developed Winter 2020

In this project, I take a dataset of bitcoin prices, preprocess the data, and implement two LSTM models. The first model predicts bitcoin's day-to-day price. The second model predicts day-to-day price differences. 

Results showed that the first modeled predictions were able to follow the trend and direction of pricing correctly. However, there is a considerable mismatch between the predicted values and the actual values. The second model's predictions were more accurate, but not significantly.

## Libraries

```
Python
tensorflow
sklearn
matplotlib
pandas
numpy
```

## Results

To the left you can see the prediction of the day-to-day pricing of Bitcoin, to the right you can find the prediction of day-to-day Bitcoin price changes.
<p align="center">
	<img width="250" src="https://jonaac.github.io/img/lstmprediction.png" />
	<img width="250" src="https://jonaac.github.io/img/lstmchangeprediction.png" />
</p>

## Further Work

Some work that I would like to expand on:
1. Explore other possible prediction targets.
2. Use a more extensive dataset.
3. Define a strategy to buy, sell, or keep bitcoin based on the predictions made by the LSTM models.
4. Compare both models. Which one is better for trading purposes, which one will return the highest profit.
5. Explore the idea of having a Reinforcement Learning agent whose actions are Buy, Sell or Keep learn how to maximize profit based on the predictions.
