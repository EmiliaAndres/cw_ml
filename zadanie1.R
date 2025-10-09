library(tidymodels)
library(parsnip)
library(dplyr)

# Helper packages
library(readr)       # import danych
library(broom.mixed) # konwersja 
library(dotwhisker)  # wizualizacja

#import danych
colnames(airquality) <- tolower(colnames(airquality))

#wstepne przygotowanie danych - np. usuniecie brakow danych i kolumny 'day'
air <-
  airquality |>
  as_tibble() |>
  na.omit() |> 
  select(-day) |> 
  mutate(month = factor(month)) 

air

#wstepne narysowanie i sprawdzenie jak zachowuja sie dane

#temperatura a miesiac
ggplot(air,
       aes(x = month, 
           y = temp)) + 
  geom_point() + 
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = .7)

#ozon a wiatr sloneczny
ggplot(air,
       aes(x = ozone, 
           y = solar.r)) + 
  geom_point() + 
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = .7)


#przygotowanie formy funkcyjnej modelu i jego silnika (metoda najmniejszych kwadratow)
linear_reg()

lm_mod <- linear_reg()

lm_fit <- 
  lm_mod %>% 
  fit(month ~ temp, data = air) #czy miesiace maja wplyw na temperature
tidy(lm_fit) %>% 
  dwplot(dot_args = list(size = 2, color = "black"),
         whisker_args = list(color = "black"),
         vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2))


#nowe przykładowe dane do sprawdzenia modelu (przewidzenia pogody)
new_points <- expand.grid(temp = 80, month = c(7, 8, 9))
new_points

#funkcja predict przewidująca na podstawie dostarczonych danych
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")
conf_int_pred

#rysowanie wykresu:
plot_data <- 
  new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

ggplot(plot_data, aes(x = month)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper),
                width = .2) + 
  labs(y = "temp")
