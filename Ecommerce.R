# ----------------------------
# 1. Load packages
# ----------------------------
if (!require(readxl)) install.packages("readxl")
if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret")
if (!require(nnet)) install.packages("nnet")
if (!require(randomForest)) install.packages("randomForest")
if (!require(xgboost)) install.packages("xgboost")
if (!require(Matrix)) install.packages("Matrix")
if (!require(ggplot2)) install.packages("ggplot2")

library(readxl)
library(dplyr)
library(caret)
library(nnet)
library(randomForest)
library(xgboost)
library(Matrix)
library(ggplot2)

set.seed(42)

# ----------------------------
# 2. Load data
# ----------------------------
sales <- read.csv("Online_Sales.csv", stringsAsFactors = FALSE)
tax <- read_excel("Tax_amount.xlsx")
customers <- read_excel("CustomersData.xlsx")
coupons <- read.csv("Discount_Coupon.csv", stringsAsFactors = FALSE)
spend <- read.csv("Marketing_Spend.csv", stringsAsFactors = FALSE)

# ----------------------------
# 3. Clean and Prepare Data
# ----------------------------
# Parse date
sales$Transaction_Date <- trimws(sales$Transaction_Date)
sales$Transaction_Date <- as.Date(sales$Transaction_Date, format = "%m/%d/%Y")

# Discount info
sales$Month <- format(sales$Transaction_Date, "%b")
sales <- merge(sales, coupons, by.x = c("Month", "Product_Category"), by.y = c("Month", "Product_Category"), all.x = TRUE)
sales$Discount_pct[is.na(sales$Discount_pct)] <- 0
sales$Discount_pct[sales$Coupon_Status != "Used"] <- 0
sales$Discount_pct <- sales$Discount_pct / 100

# GST
sales <- merge(sales, tax, by = "Product_Category", all.x = TRUE)

# Invoice value
sales$Invoice_Value <- sales$Quantity * sales$Avg_Price * (1 - sales$Discount_pct) * (1 + sales$GST) + sales$Delivery_Charges

# ----------------------------
# 4. RFM Segmentation & Clustering
# ----------------------------
customer_level <- sales %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(as.Date("2019-12-31") - max(Transaction_Date, na.rm = TRUE)),
    Frequency = n_distinct(Transaction_ID),
    Monetary = sum(Invoice_Value, na.rm = TRUE)
  )

customer_level <- customer_level %>%
  mutate(
    R_rank = ntile(desc(Recency), 4),
    F_rank = ntile(Frequency, 4),
    M_rank = ntile(Monetary, 4),
    RFM_score = R_rank + F_rank + M_rank,
    Segment = case_when(
      RFM_score >= 10 ~ "Premium",
      RFM_score >= 7  ~ "Gold",
      RFM_score >= 5  ~ "Silver",
      TRUE            ~ "Standard"
    )
  )
print(table(customer_level$Segment))

rfm_data <- customer_level %>%
  select(CustomerID, Recency, Frequency, Monetary) %>%
  filter(complete.cases(.))

rfm_matrix <- scale(rfm_data[, c("Recency", "Frequency", "Monetary")])
wss <- sapply(1:10, function(k) kmeans(rfm_matrix, centers = k, nstart = 20)$tot.withinss)
plot(1:10, wss, type = "b", pch = 19, xlab = "Number of Clusters K", ylab = "Total Within-Cluster SS", main = "K-means Elbow")

km4 <- kmeans(rfm_matrix, centers = 4, nstart = 50)
rfm_data$Cluster <- as.factor(km4$cluster)

customer_level_kmeans <- left_join(customer_level, rfm_data[, c("CustomerID", "Cluster")], by = "CustomerID")
cluster_profile <- customer_level_kmeans %>%
  group_by(Cluster) %>%
  summarise(
    Customers = n(),
    Avg_Recency = mean(Recency, na.rm = TRUE),
    Avg_Frequency = mean(Frequency, na.rm = TRUE),
    Avg_Monetary = mean(Monetary, na.rm = TRUE)
  )
print(cluster_profile)

ggplot(customer_level_kmeans, aes(x = Frequency, y = Monetary, color = Cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "K-means Customer Segments", x = "Frequency", y = "Monetary Value") +
  theme_minimal()

# ----------------------------
# 5. Feature Engineering for CLV Modeling
# ----------------------------
# Example: Use RFM as features for CLV tiering
quantiles <- quantile(customer_level$Monetary, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
customer_level$CLV_Tier <- cut(
  customer_level$Monetary,
  breaks = quantiles,
  labels = c("Low", "Medium", "High"),
  include.lowest = TRUE
)

df <- customer_level %>%
  filter(!is.na(Recency) & !is.na(Frequency) & !is.na(Monetary) & !is.na(CLV_Tier))

df$CLV_Tier <- as.factor(df$CLV_Tier)

# ----------------------------
# 6. Train/Test Split for CLV
# ----------------------------
trainIndex <- createDataPartition(df$CLV_Tier, p = 0.7, list = FALSE)
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

# ----------------------------
# 7. Multinomial Logistic Regression
# ----------------------------
multi_logit <- multinom(CLV_Tier ~ Recency + Frequency + Monetary, data = trainData)
pred_logit <- predict(multi_logit, testData)
cm_logit <- confusionMatrix(pred_logit, testData$CLV_Tier)
print(cm_logit)

# ----------------------------
# 8. Random Forest
# ----------------------------
rf_model <- randomForest(CLV_Tier ~ Recency + Frequency + Monetary, data = trainData, ntree = 100)
pred_rf <- predict(rf_model, testData)
cm_rf <- confusionMatrix(pred_rf, testData$CLV_Tier)
print(cm_rf)

imp <- importance(rf_model)
df_imp <- data.frame(Feature = rownames(imp), Importance = imp[,1])
ggplot(df_imp, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_col(fill="skyblue") +
  coord_flip() +
  labs(title="Random Forest Feature Importance", x="Feature", y="Importance")

# ----------------------------
# 9. XGBoost
# ----------------------------
xgb_train <- as.matrix(trainData[, c("Recency", "Frequency", "Monetary")])
xgb_test <- as.matrix(testData[, c("Recency", "Frequency", "Monetary")])
label_train <- as.numeric(trainData$CLV_Tier) - 1
label_test  <- as.numeric(testData$CLV_Tier) - 1

dtrain <- xgb.DMatrix(data = xgb_train, label = label_train)
dtest  <- xgb.DMatrix(data = xgb_test, label = label_test)

params <- list(
  objective = "multi:softmax",
  num_class = 3,
  eval_metric = "mlogloss"
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 0
)

pred_xgb <- predict(xgb_model, xgb_test)
class_levels <- levels(df$CLV_Tier)
pred_xgb_factor <- factor(class_levels[pred_xgb + 1], levels = class_levels)
cm_xgb <- confusionMatrix(pred_xgb_factor, testData$CLV_Tier)
print(cm_xgb)

importance_xgb <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_xgb)

# ----------------------------
# 10. Feature Engineering for Next Purchase Prediction
# ----------------------------
features <- sales %>%
  filter(Transaction_Date < as.Date("2019-10-01")) %>%
  arrange(CustomerID, Transaction_Date) %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(as.Date("2019-09-30") - max(Transaction_Date)),
    Frequency = n(),
    Monetary = sum(Invoice_Value),
    AvgInterval = ifelse(n() > 1, mean(diff(sort(Transaction_Date))), NA),
    LastInterval = ifelse(n() > 1, as.numeric(last(Transaction_Date) - nth(Transaction_Date, n()-1)), NA),
    FirstPurchaseMonth = format(min(Transaction_Date), "%b"),
    UsedCouponEver = as.integer(any(Coupon_Status == "Used"))
  )

next_purchases <- sales %>%
  filter(Transaction_Date >= as.Date("2019-10-01")) %>%
  group_by(CustomerID) %>%
  summarise(NextPurchaseDate = min(Transaction_Date))

training_data <- merge(features, next_purchases, by = "CustomerID", all.x = TRUE)
training_data$NextPurchaseDayCount <- as.numeric(training_data$NextPurchaseDate - as.Date("2019-09-30"))
training_data$NextPurchaseDayCount[is.na(training_data$NextPurchaseDayCount)] <- 999

training_data$NextPurchaseBin <- cut(
  training_data$NextPurchaseDayCount,
  breaks = c(-Inf, 30, 60, 90, Inf),
  labels = c("0-30", "31-60", "61-90", "90+")
)

print(table(training_data$NextPurchaseBin))

# Cross-Selling Analysis (Market Basket) - R Code
install.packages("devtools")
devtools::install_github("mhahsler/arules")
library(arules)

# Prepare the transactions object
# Each Transaction_ID becomes a 'basket' of unique Product_Description
trans_list <- split(sales$Product_Description, sales$Transaction_ID)
trans_list <- lapply(trans_list, unique)  # remove duplicates in a basket
trans <- as(trans_list, "transactions")

# Optional: Plot top 20 most frequent items
itemFrequencyPlot(trans, topN = 20, type = "absolute", main = "Top 20 Items by Frequency")

# Association Rule Mining: Apriori algorithm
# Parameters: min support (e.g. 0.5%), min confidence (10%), at least 2 items in a rule
rules <- apriori(trans, parameter = list(supp = 0.005, conf = 0.1, minlen = 2))

# View the top 10 rules by lift (strongest associations)
inspect(head(sort(rules, by = "lift"), 10))

# Find most common item pairs (2-item sets) and triples (3-item sets)
itemsets2 <- eclat(trans, parameter = list(supp = 0.005, maxlen = 2))
itemsets3 <- eclat(trans, parameter = list(supp = 0.002, maxlen = 3))

cat("\nTop 10 most common 2-item sets:\n")
inspect(head(sort(itemsets2, by = "support"), 10))
cat("\nTop 5 most common 3-item sets:\n")
inspect(head(sort(itemsets3, by = "support"), 5))

