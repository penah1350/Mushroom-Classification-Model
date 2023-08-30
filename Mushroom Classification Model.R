library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)


raw <- fread("mushrooms.csv")
raw %>% glimpse()

names(raw) <- names(raw) %>% str_replace_all("-","_") %>% 
  str_replace_all("\\%","_")

raw$class %>% table() %>% prop.table()

columns <- raw %>% colnames()
for (i in columns) {
  raw[[i]] <- raw[[i]] %>% str_replace_all("'","") %>% as.factor()
}

raw$class <- raw$class %>% recode(" 'e'=1 ; 'p'=0 ") %>% as_factor()

raw$class %>% view()

# Initialize H2O cluster
h2o.init()
h2o_data <- raw %>% as.h2o()

h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- "class"
features <- raw %>% select(-class) %>% names()


# Fitting h2o model ----
model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  balance_classes = T,
  nfolds = 10, seed = 123,
  max_runtime_secs = 480)

model@leaderboard %>% as.data.frame()
model@leader 


# Predicting the Test set results ----
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict %>% view()

# Threshold ----  
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> treshold

treshold

# Confusion Matrix----
confmat <- model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))
#Accuracy

model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))


#AUC

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc


#Gini

model@leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)


