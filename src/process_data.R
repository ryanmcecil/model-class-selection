library(readr)
library(dplyr)
library(fastDummies)

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
adult$education <- as.factor(adult$education)
adult$`marital-status` <- as.factor(adult$`marital-status`)
adult$occupation <- as.factor(adult$occupation)
adult$relationship <- as.factor(adult$relationship)
adult$race <- as.factor(adult$race)
adult$sex <- as.factor(adult$sex)
adult$`native-country` <- as.factor(adult$`native-country`)
adult$Y <- adult$income
adult$income <- NULL
adult$Y <- as.numeric(as.factor(adult$Y)) - 1
save_data(adult, 'adult')

# German
german <- read.csv("german.data", header=FALSE, sep = " ")
german$V1 <- as.factor(german$V1)
german$V3 <- as.factor(german$V3)
german$V4 <- as.factor(german$V4)
german$V6 <- as.factor(german$V7)
german$V6 <- as.factor(german$V7)
german$V9 <- as.factor(german$V9)
german$V10 <- as.factor(german$V10)
german$V12 <- as.factor(german$V12)
german$V14 <- as.factor(german$V14)
german$V15 <- as.factor(german$V15)
german$V17 <- as.factor(german$V17)
german$V19 <- as.factor(german$V19)
german$V20 <- as.factor(german$V20)
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
auto$horsepower <- as.numeric(auto$horsepower)
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
save_data(bike, 'bike')

# wine quality dataset
wine <- read.csv("winequality-red.csv", header=TRUE, sep = ";")
wine$Y <- wine$quality
wine$quality <- NULL
save_data(wine, 'wine')

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
cpu$vendor <- as.factor(cpu$vendor)
cpu$Y <- cpu$PRP
cpu$PRP <- NULL
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
save_data(fb, 'fb')

# Parkinsons
parkinsons$Y <- parkinsons$total_UPDRS
parkinsons$total_UPDRS <- NULL
save_data(parkinsons, 'parkinsons')

# Servo
servo <- read.table('servo.data', sep = ',')
servo$V1 <- as.factor(servo$V1)
servo$V2 <- as.factor(servo$V2)
servo$Y <- servo$V5
servo$V5 <- NULL
save_data(servo, 'servo')

# Melting Point
mtp2$Y <- mtp2$oz1143
mtp2$oz1143 <- NULL
save_data(mtp2, 'meltingpoint')
