# 2021-DSAI-HW3

使用LSTM模型，用過去一個禮拜的consumption,generation去預測明日的consumption,generation。投標的方式為根據預測每一個小時的consumption,generation，使其生產的與消耗的電度數相同，換句話說就是如果用電量大於產電量就買進電力，相反的用電量小於產電量，就賣電。