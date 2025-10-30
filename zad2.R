library(tidymodels) 
library(skimr) 
library(GGally) 
library(openair) 
library(dplyr)
library(yardstick)
tidymodels_prefer()

air <- mydata |> selectByDate(year = 2002) 
air |> skim()
head(air)

set.seed(222)

air_data <-
  air |>
  mutate(
    ozone_level = ifelse(o3 >= 10, "high", "low"),
    ozone_level = factor(ozone_level),
    date1 = lubridate::as_date(date)
  ) |>
  na.omit() |>
  mutate_if(is.character, as.factor)

data_split <- initial_split(data = air_data, prop = 3/4)
train_data <- training(data_split)
test_data  <- testing(data_split)

names(train_data)

air_rec <-
  recipe(ozone_level ~ ., data = train_data) |>
  step_rm(o3, date)

air_data |>
  count(o3)|>
  mutate(prop = n/sum(n))

air_rec |> summary()

air_data |>
  distinct(date1) |>  
  mutate(date1 = as.numeric(date1))

lr_mod <- 
  logistic_reg() |> 
  set_engine("glm")

logi_work <- 
  workflow() |> 
  add_model(lr_mod) |> 
  add_recipe(air_rec)

logi_work


logi_fit <-  
  logi_work |> 
  fit(data = train_data)


air_rec |> summary()

logi_fit |> 
  extract_fit_parsnip() |> 
  tidy()

logi_fit |> 
  extract_recipe()

predict(logi_fit, test_data)
predict(logi_fit, test_data, type = "prob")


pred_test <- 
  test_data %>%
  bind_cols(predict(logi_fit, test_data, type = "prob"))

air_rec |> summary()

pred_test |> 
  roc_curve(truth = ozone_level, .pred_high) |> 
  autoplot()

# AUC
pred_test |> 
  roc_auc(truth = ozone_level, .pred_high)
