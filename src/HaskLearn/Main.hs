{-# OPTIONS_GHC -Wno-x-partial #-}

module HaskLearn.Main where

import qualified HaskLearn.Models.KNN as KNN
import HaskLearn.Preprocessing
import HaskLearn.Metrics
import System.Random (mkStdGen)

readSSVFromStdin :: IO ([[Double]], [Double])
readSSVFromStdin = do
  contents <- getContents
  let linesOfFile = lines contents
      parsed = map parseSSVLine linesOfFile
      features = map init parsed
      targets = map last parsed
  return (features, targets)
  where
    parseSSVLine :: String -> [Double]
    parseSSVLine line = map read $ splitBy ';' (trim line)
    
    trim :: String -> String
    trim = f . f
      where f = reverse . dropWhile (== ' ')
    
    splitBy :: Char -> String -> [String]
    splitBy delimiter = foldr f [[]]
      where
        f _ [] = []
        f c (x:xs) | c == delimiter = []:x:xs
                   | otherwise = (c:x):xs

main :: IO ()
main = do
  
  let splitRatio = 0.2
  
  let seed = 42
  let kValue = 5
  
  let mode = KNN.Classification
  
  let weights = KNN.Distance

  (allFeatures, allTargets) <- readSSVFromStdin
  
  let nSamples = length allFeatures
      nFeatures = if null allFeatures then 0 else length (head allFeatures)
  
  if nSamples == 0
    then putStrLn "Ошибка: нет данных"
    else do
      let (trainX, testX, trainY, testY) = trainTestSplit (mkStdGen seed) allFeatures allTargets splitRatio
      
      putStrLn $ "Данные: " ++ show nSamples ++ " строк, " ++ show nFeatures ++ " признаков"
      putStrLn $ "Обучающая выборка: " ++ show (length trainX) ++ " примеров"
      putStrLn $ "Тестовая выборка: " ++ show (length testX) ++ " примеров"


      let model = KNN.fit mode weights kValue trainX trainY
      
      let predictions = KNN.predict model testX

      
      putStrLn "\nРезультаты на тестовой выборке:"
      
      case mode of
        KNN.Regression -> do
          let mseVal = mse testY predictions
          let r2Val  = r2Score testY predictions
          let maeVal = mae testY predictions
          putStrLn $ "  MSE: " ++ show mseVal
          putStrLn $ "  R2:  " ++ show r2Val
          putStrLn $ "  MAE: " ++ show maeVal

        KNN.Classification -> do
          let trueYInt = map (round :: Double -> Int) testY
          let predYInt = map (round :: Double -> Int) predictions
          
          let acc = accuracyScore trueYInt predYInt
          putStrLn $ "  Accuracy: " ++ show acc
          
          let posLabel = 1 :: Int
          
          let precision = precisionScore posLabel trueYInt predYInt
          let recall    = recallScore posLabel trueYInt predYInt
          
          putStrLn $ "  Precision (class " ++ show posLabel ++ "): " ++ show precision
          putStrLn $ "  Recall    (class " ++ show posLabel ++ "): " ++ show recall