{-# OPTIONS_GHC -Wno-x-partial #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}

module HaskLearn.Models.Tree
  ( DecisionTree(..)
  , fit
  , predict
  , getDepth
  , getNLeaves
  ) where

import Data.List (maximumBy, nub, sort, groupBy)
import Data.Ord (comparing)
import Data.Function (on)

-- Тип для дерева решений
data DecisionTree = 
    LeafC Int         -- для классификации: предсказанный класс
  | LeafR Double      -- для регрессии: предсказанное значение
  | TreeNode { 
      featureIdx :: Int,    -- Индекс признака для разделения
      threshold :: Double,  -- Пороговое значение
      leftTree :: DecisionTree, -- Левое поддерево (<= threshold)
      rightTree :: DecisionTree -- Правое поддерево (> threshold)
    } deriving (Show)

-- Вспомогательные функции
mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

mse :: [Double] -> Double
mse vals = 
  let avg = mean vals
      n = fromIntegral (length vals)
  in if n == 0 then 0 else sum (map (\x -> (x - avg) ** 2) vals) / n

-- Вычисление энтропии для классификации
entropy :: [Int] -> Double
entropy labels =
  let total = fromIntegral $ length labels
      classCounts = map (\c -> fromIntegral $ length (filter (== c) labels)) (nub labels)
      proportions = map (/ total) classCounts
  in -sum (map (\p -> if p == 0 then 0 else p * logBase 2 p) proportions)

-- Вычисление MSE для регрессии
mseReduction :: [Double] -> [Double] -> [Double] -> Double
mseReduction parent leftVals rightVals =
  let mseParent = mse parent
      mseLeft = mse leftVals
      mseRight = mse rightVals
      total = fromIntegral $ length parent
      leftWeight = if total == 0 then 0 else fromIntegral (length leftVals) / total
      rightWeight = if total == 0 then 0 else fromIntegral (length rightVals) / total
  in mseParent - (leftWeight * mseLeft + rightWeight * mseRight)

-- Получение всех возможных порогов для признака (средние значения между соседними отсортированными значениями)
getThresholds :: [Double] -> [Double]
getThresholds vals =
  let sortedUnique = nub (sort vals)
  in if length sortedUnique < 2
     then []
     else zipWith (\a b -> (a + b) / 2.0) sortedUnique (tail sortedUnique)

-- Нахождение наилучшего разделения
findBestSplit :: Bool -> [[Double]] -> [Double] -> (Int, Double, Double)
findBestSplit isRegression features targets =
  let nFeatures = if null features then 0 else length (head features)
      nSamples = length targets
      
      -- Для каждого признака находим лучший порог
      splits = do
        fIdx <- [0..nFeatures-1]
        let featureVals = map (!! fIdx) features
            -- Используем средние значения между соседними значениями как пороги
            thresholds = getThresholds featureVals
        
        -- Для каждого порога вычисляем критерий
        map (\th -> 
          let (leftTargets, rightTargets) = 
                foldr (\(featRow, target) (ls, rs) ->
                  if featRow !! fIdx <= th 
                    then (target:ls, rs) 
                    else (ls, target:rs)
                ) ([], []) (zip features targets)
              
              -- Выбираем критерий в зависимости от типа задачи
              criterion = if isRegression
                then mseReduction targets leftTargets rightTargets  -- Для регрессии используем уменьшение MSE
                else let leftLabels = map round leftTargets
                         rightLabels = map round rightTargets
                         parentLabels = map round targets
                     in entropy parentLabels - 
                        (fromIntegral (length leftTargets) / fromIntegral (length targets)) * entropy leftLabels -
                        (fromIntegral (length rightTargets) / fromIntegral (length targets)) * entropy rightLabels
          in (fIdx, th, criterion)) thresholds
      
      -- Выбираем разделение с максимальным критерием
      bestSplit = if null splits 
                  then (-1, 0.0, -1.0)  -- Нет возможных разделений
                  else maximumBy (comparing (\(_, _, crit) -> crit)) splits
  in bestSplit

-- Строим дерево рекурсивно
buildTree :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
buildTree isRegression features targets maxDepth minSamples
  | maxDepth <= 0 = 
      -- Создаем лист в зависимости от типа задачи
      if isRegression
        then LeafR (mean targets)          -- Для регрессии: среднее значение
        else let intTargets = map round targets
                 mostCommon = if null intTargets then 0
                              else head $ maximumBy (comparing length) 
                                   [filter (== c) intTargets | c <- nub intTargets]
             in LeafC mostCommon           -- Для классификации: самый частый класс
  | length targets < 2 * minSamples =  -- Условие минимального количества образцов в листе
      if isRegression
        then LeafR (mean targets)
        else let intTargets = map round targets
                 mostCommon = if null intTargets then 0
                              else head $ maximumBy (comparing length) 
                                   [filter (== c) intTargets | c <- nub intTargets]
             in LeafC mostCommon
  | otherwise =
      let (bestFeature, bestThreshold, bestGain) = findBestSplit isRegression features targets
      in if bestGain <= 1e-10 || bestFeature == -1  -- Нет информативного разделения
         then 
           if isRegression
             then LeafR (mean targets)
             else let intTargets = map round targets
                      mostCommon = if null intTargets then 0
                                   else head $ maximumBy (comparing length) 
                                        [filter (== c) intTargets | c <- nub intTargets]
                  in LeafC mostCommon
         else -- Разделяем данные и строим поддеревья
           let (leftFeatures, leftTargets, rightFeatures, rightTargets) =
                 foldr (\(feats, target) (lf, lt, rf, rt) ->
                   if feats !! bestFeature <= bestThreshold
                     then (feats:lf, target:lt, rf, rt)
                     else (lf, lt, feats:rf, target:rt)
                 ) ([], [], [], []) (zip features targets)
               
               -- Проверяем, что в каждом дочернем узле достаточно образцов
               leftSize = length leftTargets
               rightSize = length rightTargets
               
           in if leftSize < minSamples || rightSize < minSamples
              then -- Недостаточно образцов в одном из дочерних узлов
                if isRegression
                  then LeafR (mean targets)
                  else let intTargets = map round targets
                           mostCommon = if null intTargets then 0
                                        else head $ maximumBy (comparing length) 
                                             [filter (== c) intTargets | c <- nub intTargets]
                       in LeafC mostCommon
              else -- Разделение допустимо
                TreeNode { 
                  featureIdx = bestFeature,
                  threshold = bestThreshold,
                  leftTree = buildTree isRegression leftFeatures leftTargets (maxDepth-1) minSamples,
                  rightTree = buildTree isRegression rightFeatures rightTargets (maxDepth-1) minSamples
                }

-- Предсказание для одного примера
predictOne :: DecisionTree -> [Double] -> Double
predictOne (LeafC cls) _ = fromIntegral cls
predictOne (LeafR val) _ = val
predictOne node features =
  if features !! featureIdx node <= threshold node
    then predictOne (leftTree node) features
    else predictOne (rightTree node) features

-- Предсказание для датасета
predict :: DecisionTree -> [[Double]] -> [Double]
predict tree = map (predictOne tree)

-- Получить глубину дерева
getDepth :: DecisionTree -> Int
getDepth (LeafC _) = 0
getDepth (LeafR _) = 0
getDepth node = 1 + max (getDepth (leftTree node)) (getDepth (rightTree node))

-- Получить количество листьев
getNLeaves :: DecisionTree -> Int
getNLeaves (LeafC _) = 1
getNLeaves (LeafR _) = 1
getNLeaves node = getNLeaves (leftTree node) + getNLeaves (rightTree node)

-- Обучение модели
fit :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
fit = buildTree