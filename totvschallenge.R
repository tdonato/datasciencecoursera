install.packages("jsonlite")
install.packages("RJSONIO")
install.packages("caret")
install.packages("xgboost")

library(jsonlite)
require(RJSONIO)
library(caret)
library(xgboost)

#Inicializacao da funcao de 'split'
splitDf <- function( dataframe, seed=NULL ) {
  
  if ( !is.null( seed ) ) set.seed( seed )
  
  trainindex <- createDataPartition( y = dataframe$label, p = 0.7, list = F )
  trainset <- dataframe[ trainindex, ]
  testset <- dataframe[ -trainindex, ]
  list( trainset = trainset, testset = testset )
}

json_path <- "C:/totvschallenge/sample.txt"
json_data <- fromJSON( json_path )

json_file <- lapply(json_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
})

d <- do.call("rbind", json_file)
data <- data.frame( d )

#Transformacao dos atributos para o formato "character"
for ( i in 1:ncol( data ) ) {
  
  if ( ( substr( names( data )[i], 1, 14 ) == "dets.prod.uCom" ) |
       ( substr( names( data )[i], 1, 15 ) == "dets.prod.xProd" ) |
       ( substr( names( data )[i], 1, 10 ) == "dets.nItem" ) |
       ( substr( names( data )[i], 1, 16 ) == "dets.prod.indTot" ) | 
       ( substr( names( data )[i], 1, 5 ) == "emit." ) ) {
    
    data[,i] <- as.character( data[,i] )
  }
}

#Exclusao de dados expurios
data <- subset( data, dets.prod.xProd.6 != "United States" )
data <- subset( data, dets.nItem.4 != "TOTVS Labs" )

#Conversao dos atributos "character" para valores numericos por meio de "levels"
for ( i in 1:ncol( data ) ) {
  
  if ( ( substr( names( data )[i], 1, 14 ) == "dets.prod.uCom" ) |
    ( substr( names( data )[i], 1, 15 ) == "dets.prod.xProd" ) |
      ( substr( names( data )[i], 1, 10 ) == "dets.nItem" ) |
      ( substr( names( data )[i], 1, 16 ) == "dets.prod.indTot" ) | 
      ( substr( names( data )[i], 1, 5 ) == "emit." ) ) {
    
    data[,i] <- as.factor( data[,i] )
    data[,i] <- droplevels( data[,i] )
    data[,i] <- as.numeric( data[,i] )
    
    for ( j in 1:nrow( data ) ) {
      
      data[j,i] <- data[j,i] - 1
    }
    
  } else {
    
    data[,i] <- as.numeric( data[,i] )    
  }
}

#Exclusao de atributos que contem grande quantidade relativa de valores nulos
col.names <- names( data )
zero.rate <- sapply( data[ col.names ], function( dt.col ) {

  sum( dt.col == 0 ) / length( dt.col )

} )

keep.cols <- col.names[ zero.rate < 0.70 ]
data <- data[ keep.cols ]


#Renomeacao do atributo rotulo
names( data )[1] <- "label"

#Criacao do vetor de classes
classes <- unique( data$label )
classes_character <- as.character( classes )

#Divisao do conjunto de dados historicos em treinamento e teste
list_split <- splitDf( data )

train <- list_split[[1]]
valid <- list_split[[2]]

#Segrega os rotulos da base de treinamento
label_train <- train$label
train$label <- NULL

#Segrega os rotulos dos dados de validacao
label_valid <- valid$label
valid$label <- NULL

#Transforma as variaveis de interesse em valores numericos
label_train <- as.character( label_train )
label_valid <- as.character( label_valid )

unlabel_train <- match( label_train, classes )
unlabel_valid <- match( label_valid, classes )

#Decrementa 1 das variaveis de interesse para rodar o modelo
for ( j in 1:length( unlabel_train ) ) {
  
  unlabel_train[j] <- unlabel_train[j] - 1
}

for ( j in 1:length( unlabel_valid ) ) {
  
  unlabel_valid[j] <- unlabel_valid[j] - 1
}

##Sample features
#Construcao da base de treinamento na forma matricial
x.tr <- as.matrix( train )
dtrain <- xgb.DMatrix( x.tr, label = as.numeric( unlabel_train ) )
dtrain <- na.omit( dtrain )

#Construcao da base de validacao na forma matricial
x.val <- as.matrix( valid )
dvalid <- xgb.DMatrix( x.val, label = as.numeric( unlabel_valid ) )
dvalid <- na.omit( dvalid )

#Separacao de um registro para "scoring"
x.sc <- as.matrix( valid[1,] )
dscore <- xgb.DMatrix( x.sc )
dscore <- na.omit( dscore )

#Geracao da 'watchlist'
watchlist = list( valid = dvalid )

bestSoFar = 10000000
allParameter = data.frame( maxd=NULL, eta=NULL, best=NULL )
count = 0

#Inicializacao do vetor de 'split'
ss_vector <- c( 0.5, 0.75, 0.85, 1 )

#Atribui valores aleatorios aos parametros de formacao do modelo 'XGBoost'
depth <- floor( runif( 1, min = 4, max = 10 ) )

for ( k in 1:length( ss_vector) ) {
  
  ss_sample <- ss_vector[k] 
  
  for ( max_depth in c( 9:9 ) ) {
    #for ( max_depth in c( 2:9 ) ) {
    
    for ( eta in seq( from=0.01, to=0.01, by=0.01 ) ) {
      #for ( eta in seq( from=0.01, to=0.3, by=0.01 ) ) {
      
      count = count + 1
      
      #print( count * 100/240 )
      #print( paste0(" Model -  maxDepth:", max_depth, ", eta: ", eta ) )
      
      ###Set XGBoost parameters
      xgb.params <- list( booster = "gbtree", objective = "reg:linear",
                          max_depth = max_depth, eta = eta,
                          colsample_bytree = 0.65, subsample = ss_sample )
      
      ##Train base learner
      xgbModel <- xgb.train( data = dtrain, label = as.numeric( unlabel_train ), params = xgb.params, eval_metric = 'rmse', maximize = F, 
                             watchlist = watchlist, nrounds = 200, early.stop.round = max( 8/eta, 300 ), verbose = 0, print.every.n = 100 )
      
      
      if ( xgbModel$bestScore < bestSoFar ) {
        
        #Armazenamento do melhor modelo
        bestmodel <- xgbModel
      }
    } 
  }  
}

#Classifica os dados de SCORING com a aplicacao dos modelos
scoring <- predict( bestmodel, dscore )
