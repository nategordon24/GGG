
test <- vroom("test.csv")


datarobot_preds <- vroom("datarobot2.csv", delim = ",")


tree_submission <- datarobot_preds %>%
  bind_cols(test) %>%
  select(id, type_PREDICTION) %>%
  rename(type = type_PREDICTION)


vroom_write(tree_submission, "./DataRobotPredictions.csv", delim = ",")
