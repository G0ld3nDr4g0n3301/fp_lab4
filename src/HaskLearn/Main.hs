{-# OPTIONS_GHC -Wno-x-partial #-}

module HaskLearn.Main where

import HaskLearn.Models.Tree
import HaskLearn.Preprocessing
import HaskLearn.Metrics
import System.Random (mkStdGen)

-- Чтение SSV данных из stdin
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

-- Основная программа
main :: IO ()
main = do
  
  let splitRatio = 0.2
  
  let seed = 42

  let maxDepth = 20

  let minSamples = 10
  
  -- Чтение данных из stdin
  (allFeatures, allTargets) <- readSSVFromStdin
  
  let nSamples = length allFeatures
      nFeatures = if null allFeatures then 0 else length (head allFeatures)
  
  if nSamples == 0
    then putStrLn "Ошибка: нет данных"
    else do
      -- Разделение данных
      let (trainX, testX, trainYDouble, testYDouble) = trainTestSplit (mkStdGen seed) allFeatures allTargets splitRatio
      
      let trainY = trainYDouble
      let testY = testYDouble
      putStrLn $ "Данные: " ++ show nSamples ++ " строк, " ++ show nFeatures ++ " признаков"
      putStrLn $ "Обучающая выборка: " ++ show (length trainX) ++ " примеров"
      putStrLn $ "Тестовая выборка: " ++ show (length testX) ++ " примеров"
      
      -- Обучение модели
      let model = fit True trainX trainY maxDepth minSamples
      
      -- После обучения модели
      let depth = getDepth model
          leaves = getNLeaves model
      putStrLn $ "Глубина дерева: " ++ show depth
      putStrLn $ "Количество листьев: " ++ show leaves

      -- Предсказания
      let testPredictions = predict model testX
      
      {-
      -- Метрики
      let (tp, tn, fp, fn) = getConfusionMatrix 1 testY testPredictions
          accuracy = accuracyScore testY testPredictions
          precision = precisionScore 1 testY testPredictions
          recall = recallScore 1 testY testPredictions
          f1 = f1Score 1 testY testPredictions

      -- Вывод Пезультатов
      putStrLn $ "  TP:   " ++ show tp
      putStrLn $ "  TN:   " ++ show tn
      putStrLn $ "  FP:   " ++ show fp
      putStrLn $ "  FN:   " ++ show fn
      putStrLn $ "  accuracy:   " ++ show accuracy
      putStrLn $ "  precision:   " ++ show precision
      putStrLn $ "  recall:   " ++ show recall
      putStrLn $ "  f1:   " ++ show f1
      -}

      -- Метрики
      let testMAE = mae testY testPredictions
          testMSE = mse testY testPredictions
          testR2 = r2Score testY testPredictions
      
      -- Вывод результатов
      putStrLn "\nМетрики:"
      putStrLn $ "  MSE (test):  " ++ show testMSE
      putStrLn $ "  R2 (test):   " ++ show testR2
      putStrLn $ "  MAE (test):  " ++ show testMAE