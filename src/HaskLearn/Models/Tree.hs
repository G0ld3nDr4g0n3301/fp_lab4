{-# OPTIONS_GHC -Wno-x-partial #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module HaskLearn.Models.Tree
  ( DecisionTree(..)
  , fit
  , predict
  ) where

import Data.List (maximumBy, nub, sort)
import Data.Ord (comparing)

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
  in sum (map (\x -> (x - avg) ** 2) vals) / fromIntegral (length vals)

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
      leftWeight = fromIntegral (length leftVals) / total
      rightWeight = fromIntegral (length rightVals) / total
  in mseParent - (leftWeight * mseLeft + rightWeight * mseRight)

-- Нахождение наилучшего разделения
findBestSplit :: Bool -> [[Double]] -> [Double] -> (Int, Double, Double)
findBestSplit isRegression features targets =
  let nFeatures = length (head features)
      
      -- Для каждого признака находим лучший порог
      splits = do
        fIdx <- [0..nFeatures-1]
        let featureVals = map (!! fIdx) features
            sortedPairs = sort (zip featureVals targets)
            thresholds = nub featureVals
        
        -- Для каждого порога вычисляем критерий
        map (\th -> 
          let (leftTargets, rightTargets) = 
                foldr (\(val, target) (ls, rs) ->
                  if val <= th 
                    then (target:ls, rs) 
                    else (ls, target:rs)
                ) ([], []) sortedPairs
              
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
      bestSplit = maximumBy (comparing (\(_, _, crit) -> crit)) splits
  in bestSplit

-- Строим дерево рекурсивно
buildTree :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
buildTree isRegression features targets maxDepth minSamples
  | maxDepth <= 0 || length targets <= minSamples =
      -- Создаем лист в зависимости от типа задачи
      if isRegression
        then LeafR (mean targets)          -- Для регрессии: среднее значение
        else let intTargets = map round targets
                 mostCommon = head $ maximumBy (comparing length) 
                                  [filter (== c) intTargets | c <- nub intTargets]
             in LeafC mostCommon           -- Для классификации: самый частый класс
  | otherwise =
      let (bestFeature, bestThreshold, bestGain) = findBestSplit isRegression features targets
      in if bestGain <= 0
         then -- Нет информативного разделения
           if isRegression
             then LeafR (mean targets)
             else let intTargets = map round targets
                      mostCommon = head $ maximumBy (comparing length) 
                                       [filter (== c) intTargets | c <- nub intTargets]
                  in LeafC mostCommon
         else -- Разделяем данные и строим поддеревья
           let (leftFeatures, leftTargets, rightFeatures, rightTargets) =
                 foldr (\(feats, target) (lf, lt, rf, rt) ->
                   if feats !! bestFeature <= bestThreshold
                     then (feats:lf, target:lt, rf, rt)
                     else (lf, lt, feats:rf, target:rt)
                 ) ([], [], [], []) (zip features targets)
           in TreeNode { 
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

-- Обучение модели
fit :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
fit = buildTree