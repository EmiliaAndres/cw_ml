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



# optymalizacja hiperparametrów -------------------------------------------


rf_mod_tune <- rand_forest(
  trees = 500,      
  mtry = tune(),     # liczba zmiennych losowanych na każdym podziale
  min_n = tune()     # minimalna liczba obserwacji w węźle
) |> 
  set_engine("ranger") |> 
  set_mode("classification")


rf_work_tune <- workflow() |> 
  add_model(rf_mod_tune) |> 
  add_recipe(air_rec)

#optymalizujemy mtry i min_n
rf_grid <- grid_regular(
  mtry(range = c(1, ncol(train_data) - 2)), 
  min_n(range = c(2, 10)),
  levels = 5   #każdy parametr testowany na 5ciu poziomach
)


cv_splits <- vfold_cv(train_data, v = 5)


rf_tune_res <- tune_grid(
  rf_work_tune,
  resamples = cv_splits,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy)
)


collect_metrics(rf_tune_res)


best_params <- select_best(rf_tune_res, metric = "roc_auc")
best_params


rf_final <- finalize_model(rf_mod_tune, best_params)

rf_final_work <- workflow() |> 
  add_model(rf_final) |> 
  add_recipe(air_rec)


rf_final_fit <- last_fit(rf_final_work, split = data_split)

collect_metrics(rf_final_fit)
collect_predictions(rf_final_fit)


# Wykres AUC względem mtry, AUC mierzy zdolność modelu do rozróżniania klas
rf_tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = mtry, y = mean, color = factor(min_n))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Zależność AUC od mtry i min_n",
    x = "mtry (liczba zmiennych losowanych przy podziale)",
    y = "Średnie AUC",
    color = "min_n"
  ) +
  theme_minimal()
