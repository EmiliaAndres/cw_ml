library(tidymodels)
library(rpart)
library(rpart.plot) 
library(vip)
library(dials)
library(modeldata)


# optymalizacja cost_complexity -----------------------------------------------

data("cells", package = "modeldata")
cells
glimpse(cells)


#podział zbioru
set.seed(123)
cells_split <- initial_split(cells, strata = class)
cells_train <- training(cells_split)
cells_test  <- testing(cells_split)


tree_spec <- decision_tree(
  cost_complexity = tune(),  #określamy jaki parametr chcemy optymalizować
  tree_depth = 10,           #głębokość drzewa
  min_n = 5                  #minimalna liczba obserwacji w węźle
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_recipe <- recipe(class ~ ., data = cells_train)

tree_wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(tree_recipe)

set.seed(234)
cells_folds <- vfold_cv(cells_train, v = 5, strata = class) #walidacja krzyżowa


tree_grid <- grid_regular(cost_complexity(range = c(-4, -1)), levels = 10)

set.seed(345)
tree_tuned <- tune_grid(
  tree_wf,
  resamples = cells_folds,
  grid = tree_grid,
  metrics = metric_set(accuracy)
)


collect_metrics(tree_tuned) %>%
  arrange(desc(mean))

tree_tuned %>%
  collect_metrics() %>%
  ggplot(aes(cost_complexity, mean)) +
  geom_line(size = 1.3, alpha = 0.7, color = "steelblue") +
  geom_point(size = 2, color = "steelblue") +
  facet_wrap(~ .metric, scales = "free_y", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Wpływ hiperparametru cost_complexity na jakość drzewa decyzyjnego",
    x = "cost_complexity (log10)",
    y = "Wartość metryki"
  )



best_tree <- select_best(tree_tuned, metric = "accuracy")
best_tree


final_tree_wf <- finalize_workflow(tree_wf, best_tree)
final_tree_fit <- fit(final_tree_wf, data = cells_train)

final_preds <- predict(final_tree_fit, cells_test) %>%
  bind_cols(cells_test)

metrics(final_preds, truth = class, estimate = .pred_class)
conf_mat(final_preds, truth = class, estimate = .pred_class)


#wnioski:

#accuracy około 80%
#zgodność (kap) około 0.56
#optymalne wyniki dla cost_complexity około 10^(-3)
#skoro cost_complexity decyduje o liczbie poziomów w drzewie to wynik mówi, że
#najlepiej sprawdziły się pośrednie wartości co może przekładać się na kompromis 
#pomiędzy złożonością drzewa a dokładonością klasyfikacji