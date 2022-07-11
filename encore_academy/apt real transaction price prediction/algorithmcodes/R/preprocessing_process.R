a <- X_merge_train_data_adjusted_index0_kind1_mart_subway_hospital_theater_facility_government_school_variables_typechange_addressbylaw
#ls()
summary(a)
length(is.na(a))
length(colnames(a))
nrow(a)
colnames(a)


a$city <-as.factor(a$city) # city factoring
class(a$city)

a$address_by_law_first5<- as.factor(a$address_by_law_first5) #address_by_law factoring
str(a)
a$transaction_year <- substr(a$transaction_year_month,1,4)
a$transaction_month <- substr(a$transaction_year_month,5,6) #거래년도 년, 월 분리
str(a)
a$transaction_month <-as.factor(a$transaction_month) #거래 월 factoring
str(a)
nrow(a)
colnames(a)
c <- subset(a, select=-c(4)) #거래 년-월
str(c)
summary(c)

d <- c[!is.na(c$tallest_building_in_sites), ] #층수 결측치 처리

nrow(d)

table(d$heat_fuel)
e <- d[d$heat_fuel == "gas" | d$heat_fuel == "cogeneration", ] #"-" 연료타입 결측치제거
nrow(e)

f <- e[!is.na(e$heat_fuel),] #연료타입 결측치 제거
nrow(f)
summary(f)

f$heat_type <- as.factor(f$heat_type) #난방타입 factoring
f$heat_fuel <- as.factor(f$heat_fuel) #연료타입 factoring
str(f)

table(f$front_door_structure)
g <- f[(f$front_door_structure == "corridor" | f$front_door_structure =="mixed" | f$front_door_structure == "stairway"), ]
g$front_door_structure <- as.factor(g$front_door_structure)
nrow(g)
str(g)
summary(g)
g$apartment_id <- as.factor(g$apartment_id)  #apt_id factoring
str(g)
summary(g)
h <- g[!is.na(g$city),] # 결측치값 정제
summary(h)
nrow(h)
h$real_price <- h$transaction_real_price
h$real_price
colnames(h)

i <- h[,-c(19)]
str(i)
i$transaction_year <- as.numeric(i$transaction_year) #transaction_year numbering
str(i)
ncol(i)
head(i)

###############################이상치 검즘#########################################

summary(i)

boxplot(i$exclusive_use_area)
table(i$exclusive_use_area >= 200)
i_exclusive_use_area <- i %>% filter(exclusive_use_area < 200) 
nrow(i_exclusive_use_area)

boxplot(i$floor) #40 층
table(i$floor >= 40)
i_floor <- i_exclusive_use_area %>% filter(floor <= 40)
nrow(i_floor)

boxplot(i$total_parking_capacity_in_site)
table(i$total_parking_capacity_in_site >= 6000)
i_parking <- i_floor %>% filter(total_parking_capacity_in_site < 6000)
nrow(i_parking)

boxplot(i$total_household_count_in_sites)
#table(x$total_household_count_in_sites >= 5000)
table(i$total_household_count_in_sites >= 5000)
i_household <- i_parking %>% filter(total_household_count_in_sites < 5000)
nrow(i_household)

boxplot(i$apartment_building_count_in_sites) #60동
table(i$apartment_building_count_in_sites >= 60)
i_building_count <- i_household %>% filter(apartment_building_count_in_sites < 60)
nrow(i_building_count)

boxplot(i$tallest_building_in_sites)# 40층
table(i$tallest_building_in_sites > 50)# 40층
i_tallest_building <- i_building_count %>%  filter(tallest_building_in_sites <= 50)
nrow(i_tallest_building)

boxplot(i$lowest_building_in_sites) #30층
table(i$lowest_building_in_sites > 30)
i_lowest_building <- i_tallest_building %>% filter(lowest_building_in_sites <= 30)
nrow(i_lowest_building)

boxplot(i$supply_area) #200
table(i$supply_area > 200)
i_supply_area <- i_lowest_building %>% filter(supply_area <= 200)
nrow(i_supply_area)

boxplot(i$bathroom_count) #4개 5개
table(i$bathroom_count >= 4)
i_bathroom <- i_supply_area %>% filter(bathroom_count < 4)
nrow(i_bathroom)

boxplot(i$room_count) # 6개~
table(i$room_count >= 6)
i_room <- i_bathroom %>% filter(room_count < 6)
nrow(i_room)

boxplot(i_room$total_household_count_of_area_type) 
table(i_room$total_household_count_of_area_type > 1500)
i_area_type <- i_room %>% filter(total_household_count_of_area_type <= 1500)

nrow(i_area_type)
colnames(i_area_type)

i_withoutroomid <- i_area_type[,-c(14)]
str(i_withoutroomid)
######################################범주형 leveling#############################################################
j<- i_withoutroomid
colnames(j)
str(j)

#heat type
levels(j$heat_type) <- c(0,1,2)
table(j$heat_type)

#heat_fuel
levels(j$heat_fuel) <- c(0,1)
table(j$heat_fuel)

levels(j$front_door_structure) <- c(0,1,2)
table(j$front_door_structure)
nrow(j)
str(j)

write.csv(j, file = "realestateDataForAnalysis8.csv")

#---------------------------------------------자료정제 완료-------------------------------------------

set.seed(1004)
#install.packages("caret")
#library("caret")

training_index <- createDataPartition(j$real_price, p=0.8, list = FALSE)
real_t <- j[training_index, ]
real_v <- j[-training_index, ]
real_model_t <- lm(real_price~., data = real_t)
summary(real_model_t)

#levels(h$address_by_law)

#h %>% filter(address_by_law == "2671025028")

real_pred_v <- predict(real_model_t, newdata = real_v)
real_pred_v
rmse <- sqrt(sum((real_pred_v - real_v$real_price)^2)/length(real_v$real_price))
rmse

str(j)

#-----------------------------------------------------DUMMY VARIABLE----------------------------------
# 1 만개로 테스트
#i_onemil <- i[sample(10000),]
#View(i_onemil)

#library("foreach")

# Helper function, use the other one
# takes a column name (pointing to a factor variable) and a dataset 
# returns a dataframe containing a 1-in-K coding for this factor variable

col_to_dummy <- function(colname, data) {
  # tmp is a dataframe of K columns, where K is the number of levels of the factor in colname
  # it is a 1-in-K dummy variable coding
  levelnames <- levels(data[[colname]])
  dummy <- foreach(i=1:length(levelnames), .combine=cbind) %do% {
    as.numeric(as.numeric(data[[colname]])==i)
  }
  dummy <- as.data.frame(dummy)
  names(dummy) <- paste0(colname, ":", levelnames)
  dummy
}


factor_to_dummy <- function(obsdata) {
  
  # finding the columns containing a factor variable
  col_factor <- unlist(lapply(FUN=is.factor, obsdata))
  
  # if they are none, then nothing to do
  if(!any(col_factor)) {
    return(obsdata)
  }
  # otherwise
  # for each of these, convert it to dummy variables using col_to_dummy
  foreach(colname=names(which(col_factor)), .combine = cbind, 
          .init = obsdata[,-which(col_factor)]) %do% {
            col_to_dummy(colname, obsdata)
          }
  # each resulting data.frame is c-bound with the dataset without factors
}

df_with_dummy_vars <- factor_to_dummy(j)

df_with_dummy_vars$transaction_real_price <- df_with_dummy_vars$real_price #real price 맨 뒤로 보내기 

df_with_dummy_vars <- df_with_dummy_vars[,-c(56)]

colnames(df_with_dummy_vars)
df_with_dummy_vars

str(df_with_dummy_vars)

#---------------------------- zero var 처리--------------------------------------

which(apply(df_with_dummy_vars, 2, var)==0) # 확인

rawData_origin <- df_with_dummy_vars[ , apply(df_with_dummy_vars, 2, var) != 0] #제거

#_____________________________차원축소___________________________________________

#install.packages("MRMR")
#install.packages("e1071")
##### read data #####

#rawData= df_with_dummy_vars 
#read.csv("realestatedForRegression-1.csv")
rawData_random <- rawData_origin[sample(100000), ]
#rawData <- rawData_origin   #[1:100000,]
rawData <- rawData_random
ncol(rawData)
nrow(rawData)
View(rawData) 
colnames(rawData)
##### functions #####


kfolds = function(rawData, k){
  n = nrow(rawData)
  set.seed(123)
  randData = rawData[sample(n),]
  num = trunc(n/k)
  foldsList = list()
  for(i in 1:(k-1)){
    foldsList[[i]] = randData[(1+(i-1)*num):(i*num),]
  }
  foldsList[[k]] = randData[(1+(k-1)*num):n,]
  return(foldsList)
}

calc.r2pred = function(y, yhat){
  delta = y - yhat
  press = sum(delta^2)
  tss = sum((y-mean(y))^2)
  r2pred = 1-press/tss
  return(r2pred)
}


##### SVM-RFE #####

svmrfe = function(svm.data, numoffeatures, Type, Kernel = "linear"){
  #library("e1071")
  x = svm.data[,-ncol(svm.data)]
  y = svm.data[,ncol(svm.data)]
  n = ncol(x)
  
  survivingFeaturesIndexes = seq(1:n)
  featureRankedList = vector(length=n)
  rankedFeatureIndex = n
  
  while(length(survivingFeaturesIndexes)>0) { 
    # SVM 모형 학습
    svmModel = svm(x[, survivingFeaturesIndexes], y, type=Type, kernel = Kernel)
    # SVM의 가중치 벡터 계산
    w = t(svmModel$coefs)%*%svmModel$SV
    
    # 가중치 벡터를 제곱하여 순서를 정하는데 사용
    rankingCriteria = w * w
    
    # 변수들의 순서를 정함
    ranking = sort(rankingCriteria, index.return = TRUE)$ix
    
    # featureRankedList를 업데이트 (가장 영향력이 부족한 변수를 낮은 순위에 저장)
    featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]]
    rankedFeatureIndex = rankedFeatureIndex - 1
    
    # 가장 영향력이 부족한 변수를 제거
    (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]])
  }
  
  index = sort(featureRankedList[1:numoffeatures])
  selectedData = subset(x, select = index)
  selectedData = cbind(selectedData, y)
  return(selectedData)
}

#selectedData1 = svmrfe(rawData, 15, "C-classification")
selectedData1 = svmrfe(rawData, 118, "eps-regression")

##### mRMR #####

mRMR = function(rawData, numoffeatures){
  #library("mRMRe")
  Data = rawData
  Data[,ncol(Data)] = as.numeric(Data[,ncol(Data)])
  dd = mRMR.data(data = Data)
  model.mRMR = mRMR.classic(data=dd, target_indices = c(ncol(Data)), feature_count = numoffeatures)
  selectedindices = as.numeric(solutions(model.mRMR)[[1]])
  selectedData = Data[,selectedindices]
  finalData = cbind(selectedData, Y = Data[,ncol(Data)])
  return(finalData)
}
ncol(rawData)
colnames((rawData))
selectedData2 = mRMR(rawData, 118)
selectedData2
write.csv(selectedData2, 'featureSelectionByMRMRwithNoOutlier.csv')

##### PCA #####

pca = function(rawData){
  x = rawData[,-ncol(rawData)]
  y = rawData[,ncol(rawData)]
  pc = prcomp(x, scale = TRUE)
  
  k = 0
  R = 0
  while(R < 0.9){
    k = k + 1
    R = sum(pc[[1]][1:k])/sum(pc[[1]])
  }
  selectedData = cbind(pc[[5]][1:nrow(rawData), 1:k], y)
  finalData = as.data.frame(selectedData)
  return(finalData)
}

selectedData3 = pca(rawData)
selectedData3
write.csv(selectedData3, 'featureSelectionByPCAwithNoOutlier.csv')

##### SVM & SVR #####

svmNsvr = function(rawData, k, Kernel, Type){
  library("e1071")
  ## Kernel = "linear", "polynomial", "radial", "sigmoid"
  ## Type = "C-classification", "eps-regression"
  foldsList = kfolds(rawData, k)
  y = vector()
  yhat = vector()
  for(i in 1:k){
    trainingData = data.frame(Reduce(rbind, foldsList[-i]))
    validationData = foldsList[[i]]
    y = append(y, as.vector(as.matrix(validationData[ncol(validationData)])))
    input = trainingData[-ncol(trainingData)]
    output = trainingData[,ncol(trainingData)]
    
    model.svm = svm(input, output, kernel = Kernel, type=Type)
    pred = predict(model.svm, validationData[-ncol(validationData)])
    pred = as.numeric(as.character(pred))
    yhat = append(yhat, pred)
  }
  input = rawData[-ncol(rawData)]
  output = rawData[ncol(rawData)]
  model.svm.full = svm(input, output, kernel = Kernel, type=Type)
  pred = predict(model.svm.full, input)
  pred = as.numeric(as.character(pred))
  output = as.vector(as.matrix(output))
  
  if(Type == "C-classification"){
    accuracy.pred = 1-sum(abs(y-yhat))/length(y)
    accuracy = 1-sum(abs(output-pred))/length(output)
    return(c(accuracy, accuracy.pred))
  } else {
    r2 = 1-sum((output-pred)^2)/sum((output-mean(output))^2)
    r2pred = calc.r2pred(y, yhat)
    return(c(r2, r2pred))
  }
}


##### run code #####

result = svmNsvr(rawData, 5, "radial", "C-classification")
result.svmrfe = svmNsvr(selectedData1, 5, "radial", "C-classification")
result.mRMR = svmNsvr(selectedData2, 5, "radial", "C-classification")
result.pca = svmNsvr(selectedData3, 5, "radial", "C-classification")
