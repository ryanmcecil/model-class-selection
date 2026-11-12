library(readr)
library(dplyr)
library(fastDummies)
library(readxl)

save_data <- function(df, dataset_name){
  Y <- df %>% select(Y)
  X <- df %>% select(-Y)
  Y$Y <- as.numeric(Y$Y)
  if (any(sapply(X, is.factor))) {
    X_encoded <- dummy_cols(X, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
  } else {
    X_encoded <- X
  }
  write.csv(X_encoded, paste("../processed/",dataset_name, "_X_mat.csv", sep=""), row.names = FALSE)
  write.csv(Y, paste("../processed/", dataset_name, "_Y_mat.csv", sep=""), row.names = FALSE)
}


# COMPAS
compas <- read.csv("compas-scores-two-years.csv", header=TRUE, sep = ",")
compas <- dplyr::select(compas, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out,
                    juv_fel_count, juv_misd_count, juv_other_count) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')
compas$juv_fel_count <- ifelse(compas$juv_fel_count > 0, 1, 0)
compas$juv_misd_count <- ifelse(compas$juv_misd_count > 0, 1, 0)
compas$juv_other_count <- ifelse(compas$juv_other_count > 0, 1, 0)
cols <- c('age', 'sex', 'priors_count', 'two_year_recid', 'juv_fel_count', 'juv_misd_count', 'juv_other_count')
compas <- compas[, cols]
compas$Y <- compas$two_year_recid
compas$two_year_recid <- NULL
write.csv(compas, "../processed/compas.csv")
compas <- compas %>%
  mutate(
    age = as.numeric(age),
    sex = as.factor(sex),
    priors_count = case_when(
      priors_count == 0 ~ "priors = 0",
      priors_count == 1 ~ "priors = 1",
      priors_count >= 2 & priors_count <= 3 ~ "priors = 2 to 3",
      priors_count > 3 ~ "priors > 3"
    ),
    juv_fel_count = as.factor(juv_fel_count),
    juv_misd_count = as.factor(juv_misd_count),
    juv_other_count = as.factor(juv_other_count),
    Y = as.numeric(Y)
  )
compas$priors_count <- as.factor(compas$priors_count)
save_data(compas, 'compas')

# TELCO
telco <- read.csv("telco.csv", header=TRUE, sep = ",")
telco$customerID <- NULL
#telco$PaperlessBilling <- NULL
telco$PaymentMethod <- NULL
telco$MultipleLines <- NULL
telco$InternetService <- NULL
telco$OnlineSecurity <- NULL
telco$OnlineBackup <- NULL
telco$DeviceProtection <- NULL
telco$TechSupport <- NULL
telco$StreamingTV <- NULL
telco$StreamingMovies <- NULL
telco$TotalCharges <- NULL
telco$Y <- telco$Churn
telco$Churn <- NULL
telco$MonthlyCharges <- NULL
#telco$tenure <- NULL
telco$Contract <- NULL
telco <- na.omit(telco)
write.csv(telco, "../processed/telco.csv")

telco <- telco %>%
  mutate(
    gender = as.factor(gender),
    SeniorCitizen = as.factor(SeniorCitizen),
    Partner = as.factor(Partner),
    Dependents = as.factor(Dependents),
    tenure = as.numeric(tenure),
    PhoneService = as.factor(PhoneService),
    PaperlessBilling = as.factor(PaperlessBilling),
    Y = as.factor(Y)  # Assuming "Yes"/"No" classification
  )
telco$Y = ifelse(telco$Y == "Yes", 1, 0)
save_data(telco, 'telco')


# MONKS
monks <- read.table("monks-1.csv", header = FALSE)
monks$V8 <- NULL
monks$Y <- monks$V1
monks$V1 <- NULL
write.csv(monks, "../processed/monks.csv")
monks[] <- lapply(monks, as.factor)
monks$Y <- as.numeric(monks$Y) - 1
save_data(monks, 'monks')

# Adult
adult <- read.csv("adult.data", header=FALSE, sep = ",")
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income')
adult[adult == ' ?'] = NA
adult[adult == '?'] = NA
adult <- na.omit(adult)
adult$workclass <- as.factor(adult$workclass)
adult$fnlwgt <- NULL
adult$`marital-status` <- as.factor(adult$`marital-status`)
adult$occupation <- as.factor(adult$occupation)
adult$relationship <- as.factor(adult$relationship)
adult$race <- as.factor(adult$race)
adult$sex <- as.factor(adult$sex)
adult$education <- NULL
adult$`native-country` <- factor(ifelse(adult$`native-country` == " United-States", "US", "Non-US"))
adult$workclass <- fct_collapse(adult$workclass,
  Gov = c(" Federal-gov", " Local-gov", " State-gov"),
  SelfEmp = c(" Self-emp-inc", " Self-emp-not-inc"),
  Private = c(" Private", " Without-pay"))
adult$`marital-status` <- fct_collapse(adult$`marital-status`,
  Married = c(" Married-civ-spouse", " Married-AF-spouse"))
adult$occupation <- fct_collapse(adult$occupation,
  ProtectiveArmed = c(" Protective-serv", " Armed-Forces"),
  OtherService = c(" Other-service", " Priv-house-serv"))

adult$Y <- adult$income
adult$income <- NULL
adult$Y <- as.numeric(as.factor(adult$Y)) - 1
save_data(adult, 'adult')

# German
german <- read.csv("german.data", header=FALSE, sep = " ")
german$V1 <- as.factor(german$V1)
german$V3 <- as.factor(german$V3)
german$V4 <- as.factor(german$V4)
german$V6 <- as.factor(german$V6)
german$V7 <- as.factor(german$V7)
german$V9 <- as.factor(german$V9)
german$V10 <- as.factor(german$V10)
german$V12 <- as.factor(german$V12)
german$V14 <- as.factor(german$V14)
german$V15 <- as.factor(german$V15)
german$V17 <- as.factor(german$V17)
german$V19 <- as.factor(german$V19)
german$V20 <- as.factor(german$V20)
german$V4  <- fct_collapse(as.factor(german$V4), A4other = c("A44", "A45", "A48", "A410"))
german$V17 <- fct_collapse(as.factor(german$V17), A17unskilled = c("A171", "A172"))
german$V20 <- NULL
german$Y <- german$V21
german$V21 <- NULL
german$Y <- german$Y - 1
save_data(german, 'german')

# Boston Housing
bs <- read.csv("boston.csv", header=TRUE, sep = ",")
bs$B <- NULL
bs$BIAS_COL <- NULL
bs$CHAS <- as.factor(bs$CHAS)
bs$Y <- bs$MEDV
bs$MEDV <- NULL
save_data(bs, 'boston')

# auto mpg
auto <- read.csv("auto-mpg.csv", header=TRUE, sep = ",")
auto <- auto[!auto$horsepower == "?", ]
auto$car.name <- NULL
auto$Y <- auto$mpg
auto$mpg <- NULL
auto$horsepower <- as.numeric(as.character(auto$horsepower))
auto$origin <- as.factor(auto$origin)
save_data(auto, 'auto')

# bike share dataset
bike <- read.csv("bike.csv", header=TRUE, sep = ",")
bike$dteday <- NULL
bike$yr <- NULL
bike$mnth <- NULL
bike$hr <- NULL
bike$weekday <- NULL
bike$holiday <- as.factor(bike$holiday)
bike$season <- as.factor(bike$season)
bike$workingday <- as.factor(bike$workingday)
bike$weathersit <- as.factor(bike$weathersit)
bike$Y <- bike$cnt
bike$cnt <- NULL
bike$instant <- NULL
bike$casual <- NULL
bike$registered <- NULL
bike$atemp <- NULL
bike <- bike[bike$hum != 0, ]
bike$Y <- bike$Y / 100
save_data(bike, 'bike')

# wine quality dataset
wine_w <- read.csv("winequality-white.csv", header = TRUE, sep = ";")
wine_w <- unique(wine_w)
wine_w$Y <- wine_w$quality
wine_w$quality <- NULL
save_data(wine_w, 'wine')

# abalone
load('RealData.RData') # From Mentch, Siyu
abalone$Y <- abalone$Rings
abalone$Rings <- NULL
save_data(abalone, 'abalone')

# concrete
concrete$Y <- concrete$Concrete.compressive.strength.MPa..megapascals..
concrete$Concrete.compressive.strength.MPa..megapascals.. <- NULL
save_data(concrete, 'concrete')

# Comp hardware performance
url <- "https://archive.ics.uci.edu/static/public/29/computer+hardware.zip"
tmp <- tempfile(fileext = ".zip")
download.file(url, tmp, mode = "wb")
cpu <- read.csv(
  unz(tmp, "machine.data"),
  header = FALSE,
  col.names = c("vendor", "model", "MYCT", "MMIN", "MMAX",
                "CACH", "CHMIN", "CHMAX", "PRP", "ERP"),
  stringsAsFactors = FALSE
)
unlink(tmp)
cpu$model <- NULL
cpu$ERP <- NULL
cpu$vendor <- as.factor(cpu$vendor)
cpu$Y <- cpu$PRP
cpu$PRP <- NULL
cpu$vendor <- fct_lump_min(as.factor(cpu$vendor), min = 5)
cpu$Y <- cpu$Y / 100
save_data(cpu, 'cpm')

# CSM
csm <- read_excel("2014 and 2015 CSM dataset.xlsx")
csm$Movie <- NULL
csm$Year <- as.factor(csm$Year)
csm$Genre <- as.factor(csm$Genre)
library(forcats)
csm$Genre <- fct_lump_min(csm$Genre, min = 11, other_level = "Other")
csm$Gross <- log(csm$Gross)
csm$Budget <- log(csm$Budget)
csm$Sequel <- as.factor(csm$Sequel)
csm$Likes <- sqrt(csm$Likes)
csm$Views <- sqrt(csm$Views)
csm$Dislikes <- sqrt(csm$Dislikes)
csm$Comments <- sqrt(csm$Comments)
csm$`Aggregate Followers` <- log(csm$`Aggregate Followers`)
csm$Y <- csm$Gross
csm$Gross <- NULL
csm <- na.omit(csm)
save_data(csm, 'csm')

# Facebook metrics
fb <- read.csv("dataset_Facebook.csv", header=TRUE, sep = ";")
fb$Type <- as.factor(fb$Type)
fb$Category <- as.factor(fb$Category)
fb$Post.Weekday <- as.factor(fb$Post.Weekday)
fb$Lifetime.Engaged.Users <- NULL
fb$Lifetime.People.who.have.liked.your.Page.and.engaged.with.your.post <- NULL
fb$Lifetime.Post.Consumers <- NULL
fb$Lifetime.Post.Consumptions <- NULL
fb$Lifetime.Post.Impressions.by.people.who.have.liked.your.Page <- NULL
fb$Lifetime.Post.reach.by.people.who.like.your.Page <- NULL
fb$Lifetime.Post.Total.Impressions <- NULL
fb$Lifetime.Post.Total.Reach <- NULL
fb$comment <- NULL
fb$like <- NULL
fb$share <- NULL
fb$Y <- fb$Total.Interactions
fb$Total.Interactions <- NULL
fb$Post.Hour <- NULL
fb = na.omit(fb)
fb <- fb[fb$Y < 1000, ]
fb$Y <- fb$Y / 10
save_data(fb, 'fb')

# Servo
servo <- read.table('servo.data', sep = ',')
servo$V1 <- as.factor(servo$V1)
servo$V2 <- as.factor(servo$V2)
servo$Y <- servo$V5
servo$V5 <- NULL
save_data(servo, 'servo')

# Melting Point
url <- "https://api.openml.org/data/v1/download/52542/mtp2.arff"
tmp <- tempfile(fileext = ".arff")
download.file(url, tmp, mode = "wb")
mtp2 <- foreign::read.arff(tmp)
dim(mtp2)   # should be 274 x 1143
desc <- setdiff(names(mtp2), "oz1143")
near_const <- sapply(mtp2[desc], function(v) max(table(v)) / length(v) > 0.95)
mtp2 <- mtp2[, c(desc[!near_const], "oz1143")]
desc <- setdiff(names(mtp2), "oz1143")
dominated <- sapply(mtp2[desc], function(v) isTRUE(max(abs(scale(v))) > 10))
mtp2 <- mtp2[, c(desc[!dominated], "oz1143")]
mtp2$Y <- mtp2$oz1143
mtp2$oz1143 <- NULL
save_data(mtp2, 'meltingpoint')
