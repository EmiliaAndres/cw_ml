library(tidymodels)
library(glmnet)
library(ranger)
library(rpart)
library(readr)
library(vip)
library(ggthemes)
library(openair)
library(gt)
library(skimr)
library(ggplot2)

tidymodels_prefer()


#wczytywanie danych
dane <- importAURN(site = "kc1", year = 2021) |>
  select(o3, nox, no2, no, ws, wd, air_temp) |>
  na.omit()


#kierunki wiatru
wd_factor <- function(wd) {
  dirs <- c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  cut(wd %% 360,
      breaks = seq(0, 360, length.out = 17),
      labels = dirs,
      include.lowest = TRUE,
      right = FALSE)
}


dane <- dane |> mutate(wd_cat = wd_factor(wd))


#podział
set.seed(123)
split <- initial_split(dane, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)


recipe_glmnet <- recipe(o3 ~ nox + no2 + no + ws + wd_cat + air_temp, data = train_data) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_predictors())


recipe_tree <- recipe(o3 ~ nox + no2 + no + ws + wd_cat + air_temp, data = train_data)


#regresja
model_glmnet <- linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") |> 
  set_mode("regression")

#drzewo decyzyjne
model_rpart <- decision_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

#random forest
model_rf <- rand_forest(mtry = tune(), trees = 500, min_n = tune()) |> 
  set_engine("ranger") |> 
  set_mode("regression")


wf_glmnet <- workflow() |> add_model(model_glmnet) |> add_recipe(recipe_glmnet)
wf_rpart <- workflow() |> add_model(model_rpart) |> add_recipe(recipe_tree)
wf_rf <- workflow() |> add_model(model_rf) |> add_recipe(recipe_tree)


folds <- vfold_cv(train_data, v = 5)



tune_glmnet <- tune_grid(wf_glmnet, resamples = folds, grid = 20)
tune_rpart <- tune_grid(wf_rpart, resamples = folds, grid = 20)
tune_rf <- tune_grid(wf_rf, resamples = folds, grid = 20)



best_glmnet <- select_best(tune_glmnet, metric = "rmse")
best_rpart  <- select_best(tune_rpart,  metric = "rmse")
best_rf     <- select_best(tune_rf,     metric = "rmse")



final_glmnet <- finalize_workflow(wf_glmnet, best_glmnet)
final_rpart <- finalize_workflow(wf_rpart, best_rpart)
final_rf <- finalize_workflow(wf_rf, best_rf)



fit_glmnet <- fit(final_glmnet, data = train_data)
fit_rpart <- fit(final_rpart, data = train_data)
fit_rf <- fit(final_rf, data = train_data)


#predykcje
pred_glmnet <- predict(fit_glmnet, test_data) |> bind_cols(test_data)
pred_rpart <- predict(fit_rpart, test_data) |> bind_cols(test_data)
pred_rf <- predict(fit_rf, test_data) |> bind_cols(test_data)


metrics <- metric_set(rmse, rsq, mae)

metrics(pred_glmnet, truth = o3, estimate = .pred)
metrics(pred_rpart, truth = o3, estimate = .pred)
metrics(pred_rf, truth = o3, estimate = .pred)


#porownanie graficzne
plot_compare <- function(data, model_name) {
  ggplot(data, aes(x = .pred, y = o3)) +
    geom_point(alpha = 0.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "blue") +
    theme_minimal() +
    labs(title = paste("Rodzaj modelu:", model_name), x = "Prognozowane O3", y = "Rzeczywiste O3")
}


plot_compare(pred_glmnet, "Regresja GLMNet")
plot_compare(pred_rpart, "Drzewo decyzyjne")
plot_compare(pred_rf, "Las losowy")
