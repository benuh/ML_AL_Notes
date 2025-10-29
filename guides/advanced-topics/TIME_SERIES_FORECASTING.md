# Time Series & Forecasting Guide

Master time series analysis and forecasting from classical methods to deep learning.

## Table of Contents
1. [Time Series Fundamentals](#time-series-fundamentals)
2. [Classical Methods](#classical-methods)
3. [Deep Learning for Time Series](#deep-learning-for-time-series)
4. [Advanced Architectures](#advanced-architectures)
5. [Multivariate & Probabilistic Forecasting](#multivariate-probabilistic-forecasting)
6. [Production Deployment](#production-deployment)

---

## Time Series Fundamentals

### Understanding Time Series Data

**Components of Time Series:**
1. **Trend** - Long-term increase/decrease
2. **Seasonality** - Regular periodic patterns
3. **Cyclical** - Long-term oscillations (not fixed period)
4. **Irregular/Noise** - Random variations

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesAnalyzer:
    """Analyze time series components"""

    def __init__(self, data, freq='D'):
        self.data = data
        self.freq = freq

    def decompose(self, model='additive'):
        """Decompose time series into components"""

        # Additive: Y = Trend + Seasonal + Residual
        # Multiplicative: Y = Trend * Seasonal * Residual

        result = seasonal_decompose(
            self.data,
            model=model,
            period=365 if self.freq == 'D' else 12
        )

        # Plot components
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        result.observed.plot(ax=axes[0], title='Original')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residual')

        plt.tight_layout()
        plt.show()

        return result

    def check_stationarity(self):
        """Test for stationarity using ADF test"""
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(self.data.dropna())

        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'  {key}: {value}')

        # p-value < 0.05 => stationary
        is_stationary = result[1] < 0.05
        print(f'\nStationary: {is_stationary}')

        return is_stationary

    def make_stationary(self):
        """Transform to stationary series"""

        # Differencing
        diff1 = self.data.diff().dropna()

        # Log transform + differencing
        log_diff = np.log(self.data).diff().dropna()

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        self.data.plot(ax=axes[0], title='Original')
        diff1.plot(ax=axes[1], title='First Difference')
        log_diff.plot(ax=axes[2], title='Log + Difference')

        plt.tight_layout()
        plt.show()

        return diff1, log_diff

# Usage
ts_analyzer = TimeSeriesAnalyzer(sales_data, freq='D')
decomposition = ts_analyzer.decompose(model='additive')
is_stationary = ts_analyzer.check_stationarity()
stationary_series = ts_analyzer.make_stationary()
```

---

## Classical Methods

### ARIMA (AutoRegressive Integrated Moving Average)

**Mathematical Foundation:**

**ARIMA(p, d, q) Model:**
```
Definition: ARIMA combines three components

1. AR(p) - AutoRegressive of order p:
   y_t = c + φ_1·y_{t-1} + φ_2·y_{t-2} + ... + φ_p·y_{t-p} + ε_t

2. I(d) - Integrated of order d:
   Apply differencing d times to achieve stationarity
   ∇y_t = y_t - y_{t-1}  (first difference)
   ∇²y_t = ∇y_t - ∇y_{t-1}  (second difference)

3. MA(q) - Moving Average of order q:
   y_t = μ + ε_t + θ_1·ε_{t-1} + θ_2·ε_{t-2} + ... + θ_q·ε_{t-q}

Combined ARIMA(p,d,q):
∇^d y_t = c + Σ^p_{i=1} φ_i·∇^d y_{t-i} + Σ^q_{j=1} θ_j·ε_{t-j} + ε_t

where:
- y_t: Observed value at time t
- ∇^d: d-th order difference operator
- φ_i: AR coefficients (autoregressive parameters)
- θ_j: MA coefficients (moving average parameters)
- ε_t ~ N(0, σ²): White noise error term
- c: Constant term (drift)
```

**Stationarity Requirements:**

**Weak Stationarity (Required for ARIMA):**
```
A time series {y_t} is weakly stationary if:

1. Constant Mean: E[y_t] = μ for all t
2. Constant Variance: Var(y_t) = σ² for all t
3. Autocovariance depends only on lag:
   Cov(y_t, y_{t+k}) = γ_k for all t

If non-stationary, apply differencing until stationary.
```

**Augmented Dickey-Fuller (ADF) Test:**
```
Test for unit root (non-stationarity):

Null Hypothesis H_0: Unit root exists (non-stationary)
Alternative H_1: No unit root (stationary)

Test Statistic:
Δy_t = α + βt + γy_{t-1} + Σ^p_{i=1} δ_i·Δy_{t-i} + ε_t

If γ = 0 → unit root → non-stationary
Test: t-statistic for γ

Decision Rule:
- p-value < 0.05: Reject H_0, series is stationary
- p-value ≥ 0.05: Fail to reject H_0, series is non-stationary

Critical Values (at 95% confidence):
- -3.43 (1% significance)
- -2.86 (5% significance)
- -2.57 (10% significance)
```

**Model Selection Criteria:**

**Akaike Information Criterion (AIC):**
```
AIC = -2·log(L) + 2k

where:
- L: Maximum likelihood of the model
- k: Number of parameters (p + q + 1)

Lower AIC → Better model
Penalizes complexity to prevent overfitting
```

**Bayesian Information Criterion (BIC):**
```
BIC = -2·log(L) + k·log(n)

where:
- n: Number of observations
- k: Number of parameters

BIC penalizes complexity more heavily than AIC
Preferred for larger datasets
```

**Box-Jenkins Methodology:**
```
1. Identification: Use ACF/PACF to determine p, q
   - ACF cuts off at lag q → MA(q)
   - PACF cuts off at lag p → AR(p)
   - Both decay → ARMA(p,q)

2. Estimation: Fit model parameters using MLE
   Maximize: L(φ, θ, σ²) = Π^n_{t=1} f(y_t | y_{t-1}, ..., y_1)

3. Diagnostic Checking:
   - Residuals should be white noise
   - Ljung-Box test: H_0: residuals are uncorrelated
   - Q = n(n+2) Σ^h_{k=1} ρ²_k/(n-k) ~ χ²_h

4. Forecasting:
   ŷ_{t+h|t} = E[y_{t+h} | y_t, y_{t-1}, ...]

   Forecast intervals (95%):
   ŷ_{t+h|t} ± 1.96·σ_h

   where σ²_h = Var(y_{t+h} - ŷ_{t+h|t})
```

**Theoretical Properties:**

**Wold Decomposition Theorem:**
```
Any covariance-stationary process can be written as:

y_t = μ + Σ^∞_{j=0} ψ_j·ε_{t-j}

where Σ^∞_{j=0} ψ²_j < ∞

This justifies using ARMA models for stationary time series.
```

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class ARIMAForecaster:
    """ARIMA forecasting"""

    def __init__(self, data):
        self.data = data
        self.model = None
        self.fitted_model = None

    def find_order(self):
        """Find optimal ARIMA order using ACF/PACF plots"""

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # ACF plot (for MA order q)
        plot_acf(self.data.dropna(), lags=40, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')

        # PACF plot (for AR order p)
        plot_pacf(self.data.dropna(), lags=40, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        plt.show()

    def auto_arima(self):
        """Automatically find best ARIMA parameters"""
        from pmdarima import auto_arima

        model = auto_arima(
            self.data,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,  # Let algorithm determine
            seasonal=True,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            m=12,  # Seasonal period
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        print(model.summary())
        return model.order, model.seasonal_order

    def fit(self, order=(1, 1, 1)):
        """Fit ARIMA model"""

        self.model = ARIMA(self.data, order=order)
        self.fitted_model = self.model.fit()

        print(self.fitted_model.summary())
        return self.fitted_model

    def forecast(self, steps=30):
        """Generate forecasts"""

        forecast = self.fitted_model.forecast(steps=steps)

        # Confidence intervals
        forecast_df = self.fitted_model.get_forecast(steps=steps).summary_frame()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data.values, label='Historical')
        plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
        plt.fill_between(
            forecast_df.index,
            forecast_df['mean_ci_lower'],
            forecast_df['mean_ci_upper'],
            alpha=0.3,
            color='red'
        )
        plt.legend()
        plt.title('ARIMA Forecast')
        plt.show()

        return forecast

    def evaluate(self, test_data):
        """Evaluate forecast accuracy"""

        predictions = self.fitted_model.forecast(steps=len(test_data))

        # Metrics
        mae = np.mean(np.abs(predictions - test_data))
        rmse = np.sqrt(np.mean((predictions - test_data)**2))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

        print(f'MAE: {mae:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAPE: {mape:.2f}%')

        return {'mae': mae, 'rmse': rmse, 'mape': mape}

# Usage
arima = ARIMAForecaster(sales_data)
arima.find_order()

# Auto-select order
order, seasonal_order = arima.auto_arima()

# Fit and forecast
arima.fit(order=order)
forecast = arima.forecast(steps=30)
metrics = arima.evaluate(test_data)
```

---

### Prophet (Facebook)

**Best for:** Business time series with strong seasonal patterns.

```python
from prophet import Prophet

class ProphetForecaster:
    """Facebook Prophet forecasting"""

    def __init__(self, data):
        # Prophet requires 'ds' (date) and 'y' (value) columns
        self.data = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        self.model = None

    def fit(self, seasonality_mode='additive', changepoint_prior_scale=0.05):
        """Fit Prophet model"""

        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        # Add custom seasonality
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Add holidays
        self.model.add_country_holidays(country_name='US')

        self.model.fit(self.data)

        return self.model

    def forecast(self, periods=30, freq='D'):
        """Generate forecast"""

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Predict
        forecast = self.model.predict(future)

        # Plot
        fig = self.model.plot(forecast)
        plt.title('Prophet Forecast')
        plt.show()

        # Components plot
        fig2 = self.model.plot_components(forecast)
        plt.show()

        return forecast

    def add_regressor(self, regressor_name, regressor_data):
        """Add external regressor"""

        self.data[regressor_name] = regressor_data
        self.model.add_regressor(regressor_name)

    def cross_validate(self):
        """Cross-validation for Prophet"""
        from prophet.diagnostics import cross_validation, performance_metrics

        df_cv = cross_validation(
            self.model,
            initial='730 days',
            period='180 days',
            horizon='90 days'
        )

        df_p = performance_metrics(df_cv)
        print(df_p.head())

        return df_cv, df_p

# Usage
prophet = ProphetForecaster(sales_data)
prophet.fit(seasonality_mode='multiplicative')

# Add external regressors
prophet.add_regressor('temperature', temperature_data)
prophet.add_regressor('promotion', promotion_data)

forecast = prophet.forecast(periods=90, freq='D')
cv_results = prophet.cross_validate()
```

---

## Deep Learning for Time Series

### LSTM for Time Series

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """LSTM for time series forecasting"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))

        # Take last output
        out = self.fc(out[:, -1, :])

        return out

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Time series dataset with sliding window"""

    def __init__(self, data, seq_length=30, forecast_horizon=1):
        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon]

        return torch.FloatTensor(x), torch.FloatTensor(y)

def train_lstm_forecaster(model, train_loader, val_loader, epochs=50):
    """Train LSTM forecaster"""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch.unsqueeze(-1))
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch.unsqueeze(-1))
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')

    return model

# Usage
dataset = TimeSeriesDataset(normalized_data, seq_length=30, forecast_horizon=7)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2)
trained_model = train_lstm_forecaster(model, train_loader, val_loader, epochs=100)
```

---

### Temporal Convolutional Network (TCN)

**Advantages over RNN:** Parallel processing, stable gradients, flexible receptive field.

```python
class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions"""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)  # Causal padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    dilation=dilation_size, dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # TCN forward
        y = self.network(x)

        # Take last timestep
        y = y[:, :, -1]

        # Output
        return self.fc(y)

# Usage
model = TCN(
    num_inputs=1,
    num_channels=[64, 64, 64, 64],  # 4 levels
    kernel_size=3,
    dropout=0.2
)

# Receptive field = 2^num_levels * (kernel_size - 1) + 1
# = 2^4 * (3-1) + 1 = 33 timesteps
```

---

## Advanced Architectures

### Transformer for Time Series

```python
class TimeSeriesTransformer(nn.Module):
    """Transformer for time series forecasting"""

    def __init__(self, d_model=64, nhead=4, num_layers=3, seq_len=30, pred_len=7):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = self._generate_pos_encoding(seq_len + pred_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

    def _generate_pos_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch, seq_len, 1)

        # Embed input
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Project to output
        output = self.output_projection(x[:, -self.pred_len:, :])

        return output

# Training
model = TimeSeriesTransformer(d_model=64, nhead=4, num_layers=3, seq_len=30, pred_len=7)
```

---

### N-BEATS (Neural Basis Expansion Analysis)

**State-of-the-art for univariate forecasting.**

```python
class NBeatsBlock(nn.Module):
    """N-BEATS block"""

    def __init__(self, input_size, theta_size, basis_function, num_layers=4, hidden_size=256):
        super().__init__()

        # Fully connected stack
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.fc_stack = nn.Sequential(*layers)

        # Basis parameters
        self.theta_b_fc = nn.Linear(hidden_size, theta_size)
        self.theta_f_fc = nn.Linear(hidden_size, theta_size)

        self.basis_function = basis_function

    def forward(self, x):
        # FC stack
        h = self.fc_stack(x)

        # Basis parameters
        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)

        # Backcast and forecast
        backcast = self.basis_function(theta_b, x.size(-1))
        forecast = self.basis_function(theta_f, x.size(-1))

        return backcast, forecast

class TrendBasis(nn.Module):
    """Polynomial trend basis"""

    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, theta, length):
        # Polynomial basis
        T = torch.pow(
            torch.arange(length, dtype=torch.float).unsqueeze(0) / length,
            torch.arange(self.degree, dtype=torch.float).unsqueeze(1)
        ).to(theta.device)

        return torch.matmul(theta, T)

class NBeats(nn.Module):
    """N-BEATS architecture"""

    def __init__(self, input_size, output_size, num_stacks=2, num_blocks=3):
        super().__init__()

        self.stacks = nn.ModuleList()

        for _ in range(num_stacks):
            stack = nn.ModuleList()
            for _ in range(num_blocks):
                stack.append(
                    NBeatsBlock(
                        input_size=input_size,
                        theta_size=4,
                        basis_function=TrendBasis(degree=4)
                    )
                )
            self.stacks.append(stack)

    def forward(self, x):
        forecast = torch.zeros_like(x)

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(x)
                x = x - backcast
                forecast = forecast + block_forecast

        return forecast
```

---

## Multivariate & Probabilistic Forecasting

### Multivariate LSTM with Attention

```python
class MultivariateLSTMAttention(nn.Module):
    """Multivariate time series with attention"""

    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            num_features, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )

        # Attention
        self.attention = nn.Linear(hidden_size, 1)

        # Output
        self.fc = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # x: (batch, seq_len, num_features)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        output = self.fc(context)

        return output, attention_weights
```

---

### Probabilistic Forecasting with DeepAR

```python
class DeepAR(nn.Module):
    """DeepAR for probabilistic forecasting"""

    def __init__(self, num_features, hidden_size=40, num_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(
            num_features + 1,  # +1 for lag
            hidden_size, num_layers,
            batch_first=True, dropout=0.1
        )

        # Distribution parameters
        self.mu = nn.Linear(hidden_size, 1)
        self.sigma = nn.Linear(hidden_size, 1)

    def forward(self, x, z_prev):
        # Concatenate input with previous prediction
        lstm_input = torch.cat([x, z_prev], dim=-1)

        # LSTM
        lstm_out, hidden = self.lstm(lstm_input)

        # Distribution parameters
        mu = self.mu(lstm_out[:, -1, :])
        sigma = torch.exp(self.sigma(lstm_out[:, -1, :]))  # Ensure positive

        return mu, sigma, hidden

    def sample(self, mu, sigma, num_samples=100):
        """Sample from predicted distribution"""
        dist = torch.distributions.Normal(mu, sigma)
        samples = dist.sample((num_samples,))
        return samples

    def loss(self, mu, sigma, target):
        """Negative log-likelihood loss"""
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(target).mean()

# Training with teacher forcing
def train_deepar(model, train_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            # Initialize with last observed value
            z_prev = x[:, -1:, -1:]

            mu, sigma, _ = model(x, z_prev)
            loss = model.loss(mu, sigma, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}')
```

---

## Production Deployment

### Real-Time Forecasting API

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Time Series Forecasting API")

# Load model
model = joblib.load('forecaster_model.pkl')
scaler = joblib.load('scaler.pkl')

class ForecastRequest(BaseModel):
    historical_data: list
    horizon: int = 7

class ForecastResponse(BaseModel):
    predictions: list
    confidence_intervals: dict

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate forecast"""

    # Preprocess
    data = np.array(request.historical_data).reshape(-1, 1)
    data_scaled = scaler.transform(data)

    # Generate forecast
    predictions = model.predict(data_scaled, horizon=request.horizon)

    # Inverse transform
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Confidence intervals (example)
    lower_bound = predictions * 0.95
    upper_bound = predictions * 1.05

    return ForecastResponse(
        predictions=predictions.tolist(),
        confidence_intervals={
            'lower': lower_bound.tolist(),
            'upper': upper_bound.tolist()
        }
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

### Streaming Forecasts

```python
class StreamingForecaster:
    """Online learning for streaming time series"""

    def __init__(self, model, window_size=100):
        self.model = model
        self.window_size = window_size
        self.buffer = []

    def update(self, new_data):
        """Update model with new data"""

        self.buffer.append(new_data)

        # Keep only recent data
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Retrain periodically
        if len(self.buffer) == self.window_size:
            X = np.array(self.buffer[:-1])
            y = np.array(self.buffer[-1])

            # Online update
            self.model.partial_fit(X.reshape(1, -1), [y])

    def predict(self, horizon=1):
        """Predict next values"""

        if len(self.buffer) < self.window_size:
            return None

        X = np.array(self.buffer[-self.window_size:])
        return self.model.predict(X.reshape(1, -1), horizon=horizon)
```

---

## Summary

| Method | Type | Best For | Accuracy | Speed |
|--------|------|----------|----------|-------|
| **ARIMA** | Statistical | Univariate, linear | Medium | Fast |
| **Prophet** | Statistical | Business data, seasonality | Medium | Fast |
| **LSTM** | Deep Learning | Nonlinear, multivariate | High | Slow |
| **TCN** | Deep Learning | Long sequences | High | Medium |
| **Transformer** | Deep Learning | Complex patterns | Very High | Slow |
| **N-BEATS** | Deep Learning | Univariate forecasting | Very High | Medium |
| **DeepAR** | Probabilistic | Uncertainty quantification | High | Slow |

---

## Key Takeaways

1. **Start with Prophet** for business time series (easy, interpretable)
2. **ARIMA** for stationary univariate series
3. **LSTM/TCN** for nonlinear multivariate forecasting
4. **Transformer** for complex long-range dependencies
5. **Always check stationarity** and decompose series
6. **Use proper train/val/test splits** (no random shuffling!)
7. **Evaluate with domain-appropriate metrics** (MAPE, SMAPE, WAPE)
8. **Consider probabilistic forecasts** for uncertainty

**Best Practices:**
- Use walk-forward validation, not random CV
- Handle missing data appropriately
- Feature engineer: lags, rolling stats, seasonality
- Ensemble multiple models for robustness
- Monitor for concept drift in production
- Retrain periodically with recent data

**Next Steps:**
- Implement LSTM forecaster for your data
- Try Prophet for business metrics
- Experiment with Transformer architectures
- Deploy real-time forecasting API
- Build ensemble of classical + DL methods


---

## 📚 References

**Time Series Foundations:**

1. **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.** (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
   - Classic textbook on ARIMA methodology

2. **Hamilton, J. D.** (1994). *Time Series Analysis*. Princeton University Press.
   - Rigorous mathematical treatment

3. **Hyndman, R. J., & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - Free online: https://otexts.com/fpp3/

**ARIMA & Statistical Methods:**

4. **Dickey, D. A., & Fuller, W. A.** (1979). "Distribution of the estimators for autoregressive time series with a unit root." *Journal of the American Statistical Association*, 74(366a), 427-431.
   - ADF test for stationarity

5. **Ljung, G. M., & Box, G. E. P.** (1978). "On a measure of lack of fit in time series models." *Biometrika*, 65(2), 297-303.
   - Ljung-Box test for residuals

6. **Akaike, H.** (1974). "A new look at the statistical model identification." *IEEE Transactions on Automatic Control*, 19(6), 716-723.
   - AIC for model selection

**Deep Learning for Time Series:**

7. **Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T.** (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *International Journal of Forecasting*, 36(3), 1181-1191.
   - DeepAR for probabilistic forecasting

8. **Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T.** (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 37(4), 1748-1764.
   - TFT architecture

9. **Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y.** (2020). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." *ICLR 2020*.
   - N-BEATS architecture

**Prophet & Practical Methods:**

10. **Taylor, S. J., & Letham, B.** (2018). "Forecasting at scale." *The American Statistician*, 72(1), 37-45.
    - Facebook Prophet algorithm

**Online Resources:**
- statsmodels documentation: https://www.statsmodels.org/
- Facebook Prophet: https://facebook.github.io/prophet/
- GluonTS (Amazon): https://ts.gluon.ai/

---

*Time series forecasting combines classical statistics with modern deep learning for robust predictions.*
