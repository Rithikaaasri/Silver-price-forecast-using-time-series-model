
#--------------------------------------------IMPORT FILE : STEP 1 -----------------------------
library(readxl)

# Load dataset 
df <- read.csv("C://Users/hp/OneDrive/Desktop/FIN/Silver weekly Historical Data.csv")
df

# Load essential libraries
library(tseries)
library(forecast)
library(TSA)
library(TSstudio)
library(strucchange)
library(plotly)
library(psych)
library(DataExplorer)
library(lubridate)
library(ggplot2)

# -----------------------UNDERSTAND THE DATA: STEP 2 [EDA]--------------------------
str(df)
summary(df)
head(df, 10)
tail(df, 10)

# Check for missing values
sum(is.na(df))
plot_missing(df)
plot_intro(df)
plot_qq(df$Price)

# Price line plot
plot(df$Price, type = "l", main = "Silver Price Over Time",
     xlab = "Time Index", ylab = "Price (USD)")
hist(df$Price)

#-------------------------CLEAN AND REMOVE : STEP 3 -----------------------------------
colnames(df) <- c("Date", "Price", "Open", "High", "Low", "Volume", "ChangePercent")

# Fix date
df$Date <- gsub("-", "/", df$Date)
df$Date <- parse_date_time(df$Date, orders = c("mdy", "dmy"))

# Volume column: remove K
df$Volume <- as.numeric(gsub("K", "", df$Volume)) * 1000

# Remove '%' in Change column
df$ChangePercent <- as.numeric(gsub("%", "", df$ChangePercent))

# Convert prices to numeric
df$Price <- as.numeric(gsub(",", "", df$Price))
df$Open  <- as.numeric(gsub(",", "", df$Open))
df$High  <- as.numeric(gsub(",", "", df$High))
df$Low   <- as.numeric(gsub(",", "", df$Low))

# Ensure sorted by date
df <- df[order(df$Date), ]

head(df, 10)
tail(df, 10)
str(df, 10)

#--------------------------------------KPI VISUALIZATIONS (UNIT 1) --------------------------
ggplot(df, aes(x = Date, y = Price)) +
  geom_line(color = "steelblue") +
  labs(title = "Silver Price Over Time", x = "Date", y = "Price (USD)") +
  theme_minimal()

ggplot(df, aes(x = Date, y = Volume)) +
  geom_line(color = "darkgreen") +
  labs(title = "Trading Volume", x = "Date", y = "Volume (Contracts)") +
  theme_minimal()

ggplot(df, aes(x = Date, y = ChangePercent)) +
  geom_line(color = "darkred") +
  labs(title = "Daily % Change", x = "Date", y = "Change %") +
  theme_minimal()

#------------------------------------UNIT 2 : TIME SERIES CONVERSION -------------------------
min(df$Date)
max(df$Date)

# Convert to weekly time series
start_year <- year(min(df$Date))
start_week <- isoweek(min(df$Date))
end_year <- year(max(df$Date))
end_week <- isoweek(max(df$Date))

df_ts <- ts(df$Price,
            start = c(start_year, start_week),
            end = c(end_year, end_week),
            frequency = 52)

str(df_ts)

#-------------------------RANDOM WALK CHECK -------------------------------
lagged_price <- stats::lag(df_ts, k = -1)

plot(lagged_price, df_ts,
     main = "Price(t) vs Price(t-1)",
     xlab = "t-1", ylab = "t",
     col = "blue", pch = 20)
abline(lm(df_ts ~ lagged_price), col = "red")


#------------------ STATIONARITY TESTS -----------------------------
kpss.test(df_ts)
adf.test(df_ts) # both test confirms non-stationary

plot(df_ts) #looks multiplicative

#-------------------------- DECOMPOSITION ------------------------------

decomp_mul <- decompose(df_ts, type = "multiplicative")
plot(decomp_mul)

#------------------------ FORECASTING ----------------------------------
h <- 60  # forecast 20 weeks

#---------NAIVE FORECAST -----------

naive_model <- naive(df_ts, h) 
autoplot(naive_model) +
  ggtitle("Naive Forecast - Silver Prices") +
  xlab("Time") + ylab("Price (USD)") +
  theme_minimal()

#---------NAIVE WITH DRIFT FORECAST------------

drift_model <- rwf(df_ts, drift = TRUE, h = 20)
autoplot(drift_model) +
  ggtitle("Naive Forecast with Drift") +
  xlab("Time") + ylab("Price (USD)") +
  theme_minimal()

#---------Exponential Smoothing FORECASTING -----------
# SES
model_ses <- ets(df_ts, model = "ANN")
forecast_ses <- forecast(model_ses, h = h)

# DES
model_des <- ets(df_ts, model = "AAN")
forecast_des <- forecast(model_des, h = h)

# TES - Skip if error due to frequency
# model_tes <- ets(df_ts, model = "AAA")  
# forecast_tes <- forecast(model_tes, h = h)

# Plot SES & DES
par(mfrow = c(2, 1))
plot(forecast_ses, main = "SES", col = "blue")
plot(forecast_des, main = "DES", col = "green")
par(mfrow = c(1, 1))


#------------------------------ACF AND PACF TEST-----------------------
# PACF
pacf(df_ts) #AR(1)
#------------------------------ARIMA--------------------------------------

model_arima <- auto.arima(df_ts)
summary(model_arima)
forecast_arima <- forecast(model_arima, h )
autoplot(forecast_arima) +
  ggtitle("ARIMA(0,1,0) Forecast - Silver Price") +
  xlab("Time") + ylab("Price (USD)") +
  theme_minimal()
checkresiduals(model_arima)



#------------------------------ SARIMA --------------------------------------
#------------------------------ USING ACF AND PACF --------------------------------------

# ACF and PACF after seasonal differencing to guide (P,D,Q)
Acf(diff(df_ts, 52), main = "ACF - Seasonal Differencing")
Pacf(diff(df_ts, 52), main = "PACF - Seasonal Differencing")

# ACF AND PACF for (p, d, q)
# First-order differencing (non-seasonal)
diff_non_seasonal <- diff(df_ts, differences = 1)

# ACF and PACF for non-seasonal pdq
Acf(diff_non_seasonal, main = "ACF - Non-seasonal First Differencing")
Pacf(diff_non_seasonal, main = "PACF - Non-seasonal First Differencing")

# Fit a manual SARIMA model based on observed ACF/PACF
manual_sarima <- Arima(df_ts, 
                       order = c(1, 1, 2),                     # <- updated (p,d,q)
                       seasonal = list(order = c(1, 1, 1),     # <- updated (P,D,Q)
                                       period = 52))           # <- seasonal period

# Print summary
summary(manual_sarima)

# Forecast next 60 weeks
forecast_manual_sarima <- forecast(manual_sarima, h = 60)

# Plot with model details in title
autoplot(forecast_manual_sarima) +
  ggtitle("SARIMA(1,1,2)(1,1,1)[52] Forecast - Silver Price") +
  xlab("Time") +
  ylab("Price (USD)") +
  theme_minimal()

# Residual diagnostics
checkresiduals(manual_sarima)

# Ljung-Box test
Box.test(manual_sarima$residuals, type = "Ljung-Box")



#-------------------------------IMPROVING------------------------------------
