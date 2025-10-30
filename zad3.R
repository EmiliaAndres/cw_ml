library(ranger)
library(modeldata)
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


# regresja logistyczna ----------------------------------------------------


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



cv_splits <- vfold_cv(train_data, v = 5)

# LR
lr_res <- fit_resamples(
  logi_work,
  resamples = cv_splits,
  metrics = metric_set(roc_auc, accuracy)
)


# RF
rf_res <- fit_resamples(
  rf_work,
  resamples = cv_splits,
  metrics = metric_set(roc_auc, accuracy)
)

collect_metrics(lr_res)
collect_metrics(rf_res)


boot_splits <- bootstraps(train_data, times = 25)

# LR
lr_boot <- fit_resamples(
  logi_work,
  resamples = boot_splits,
  metrics = metric_set(roc_auc, accuracy)
)

# RF
rf_boot <- fit_resamples(
  rf_work,
  resamples = boot_splits,
  metrics = metric_set(roc_auc, accuracy)
)

collect_metrics(lr_boot)
collect_metrics(rf_boot)



# Najlepszy model (np. LR)
lr_final_fit <- last_fit(logi_work, split = data_split)

collect_metrics(lr_final_fit)
collect_predictions(lr_final_fit)





# las losowy --------------------------------------------------------------



# Model lasu losowego
rf_mod <- rand_forest(trees = 500) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Workflow lasu losowego
rf_work <- workflow() |> 
  add_model(rf_mod) |> 
  add_recipe(air_rec)


rf_mod <- rand_forest(
  trees = 500,       
  min_n = 5          
) |> 
  set_engine("ranger") |> 
  set_mode("classification")

rf_work <- workflow() |> 
  add_model(rf_mod) |> 
  add_recipe(air_rec)


cv_splits <- vfold_cv(train_data, v = 5)

rf_res <- fit_resamples(
  rf_work,
  resamples = cv_splits,
  metrics = metric_set(roc_auc, accuracy, brier_class)
)

collect_metrics(rf_res)


boot_splits <- bootstraps(train_data, times = 25)

rf_boot <- fit_resamples(
  rf_work,
  resamples = boot_splits,
  metrics = metric_set(roc_auc, accuracy, brier_class)
)

collect_metrics(rf_boot)


Po porównaniu tabel (collect_metrics) z obu sposobów widać, że las losowy
posiada wyższą dokładność (parametr accuracy) 0.910 do 0.878 dla regresji,
co oznacza, że lepiej klasyfikuje on dni na podstawie stężenia ozonu (niskie/wysokie)

Spadek brier_class dla modelu lasu losowego pokazuje, że las losowy jest wstanie 
lepiej przewidzieć prawdopodobieństwo wystąpienia wysokiego poziomu ozonu.


Wnioski - las losowy okazał się lepszym wyborem jeśli chodzi o predykcję poziomu ozonu.