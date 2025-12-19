{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# OPTIONS_GHC -Wno-x-partial #-}

module HaskLearn.Models.Tree
  ( DecisionTree (..),
    fit,
    predict,
    getDepth,
    getNLeaves,
  )
where

import Data.Function (on)
import Data.List (groupBy, maximumBy, nub, sort)
import Data.Ord (comparing)

data DecisionTree
  = LeafC Int
  | LeafR Double
  | TreeNode
      { featureIdx :: Int,
        threshold :: Double,
        leftTree :: DecisionTree,
        rightTree :: DecisionTree
      }
  deriving (Show)

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

mse :: [Double] -> Double
mse vals =
  let avg = mean vals
      n = fromIntegral (length vals)
   in if n == 0 then 0 else sum (map (\x -> (x - avg) ** 2) vals) / n

entropy :: [Int] -> Double
entropy labels =
  let total = fromIntegral $ length labels
      classCounts = map (\c -> fromIntegral $ length (filter (== c) labels)) (nub labels)
      proportions = map (/ total) classCounts
   in -sum (map (\p -> if p == 0 then 0 else p * logBase 2 p) proportions)

mseReduction :: [Double] -> [Double] -> [Double] -> Double
mseReduction parent leftVals rightVals =
  let mseParent = mse parent
      mseLeft = mse leftVals
      mseRight = mse rightVals
      total = fromIntegral $ length parent
      leftWeight = if total == 0 then 0 else fromIntegral (length leftVals) / total
      rightWeight = if total == 0 then 0 else fromIntegral (length rightVals) / total
   in mseParent - (leftWeight * mseLeft + rightWeight * mseRight)

getThresholds :: [Double] -> [Double]
getThresholds vals =
  let sortedUnique = nub (sort vals)
   in if length sortedUnique < 2
        then []
        else zipWith (\a b -> (a + b) / 2.0) sortedUnique (tail sortedUnique)

findBestSplit :: Bool -> [[Double]] -> [Double] -> (Int, Double, Double)
findBestSplit isRegression features targets =
  let nFeatures = if null features then 0 else length (head features)
      nSamples = length targets

      splits = do
        fIdx <- [0 .. nFeatures - 1]
        let featureVals = map (!! fIdx) features
            thresholds = getThresholds featureVals

        map
          ( \th ->
              let (leftTargets, rightTargets) =
                    foldr
                      ( \(featRow, target) (ls, rs) ->
                          if featRow !! fIdx <= th
                            then (target : ls, rs)
                            else (ls, target : rs)
                      )
                      ([], [])
                      (zip features targets)

                  criterion =
                    if isRegression
                      then mseReduction targets leftTargets rightTargets
                      else
                        let leftLabels = map round leftTargets
                            rightLabels = map round rightTargets
                            parentLabels = map round targets
                         in entropy parentLabels
                              - (fromIntegral (length leftTargets) / fromIntegral (length targets)) * entropy leftLabels
                              - (fromIntegral (length rightTargets) / fromIntegral (length targets)) * entropy rightLabels
               in (fIdx, th, criterion)
          )
          thresholds

      bestSplit =
        if null splits
          then (-1, 0.0, -1.0)
          else maximumBy (comparing (\(_, _, crit) -> crit)) splits
   in bestSplit

buildTree :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
buildTree isRegression features targets maxDepth minSamples
  | maxDepth <= 0 =
      if isRegression
        then LeafR (mean targets)
        else
          let intTargets = map round targets
              mostCommon =
                if null intTargets
                  then 0
                  else
                    head $
                      maximumBy
                        (comparing length)
                        [filter (== c) intTargets | c <- nub intTargets]
           in LeafC mostCommon
  | length targets < 2 * minSamples =
      if isRegression
        then LeafR (mean targets)
        else
          let intTargets = map round targets
              mostCommon =
                if null intTargets
                  then 0
                  else
                    head $
                      maximumBy
                        (comparing length)
                        [filter (== c) intTargets | c <- nub intTargets]
           in LeafC mostCommon
  | otherwise =
      let (bestFeature, bestThreshold, bestGain) = findBestSplit isRegression features targets
       in if bestGain <= 1e-10 || bestFeature == -1
            then
              if isRegression
                then LeafR (mean targets)
                else
                  let intTargets = map round targets
                      mostCommon =
                        if null intTargets
                          then 0
                          else
                            head $
                              maximumBy
                                (comparing length)
                                [filter (== c) intTargets | c <- nub intTargets]
                   in LeafC mostCommon
            else
              let (leftFeatures, leftTargets, rightFeatures, rightTargets) =
                    foldr
                      ( \(feats, target) (lf, lt, rf, rt) ->
                          if feats !! bestFeature <= bestThreshold
                            then (feats : lf, target : lt, rf, rt)
                            else (lf, lt, feats : rf, target : rt)
                      )
                      ([], [], [], [])
                      (zip features targets)

                  leftSize = length leftTargets
                  rightSize = length rightTargets
               in if leftSize < minSamples || rightSize < minSamples
                    then
                      if isRegression
                        then LeafR (mean targets)
                        else
                          let intTargets = map round targets
                              mostCommon =
                                if null intTargets
                                  then 0
                                  else
                                    head $
                                      maximumBy
                                        (comparing length)
                                        [filter (== c) intTargets | c <- nub intTargets]
                           in LeafC mostCommon
                    else
                      TreeNode
                        { featureIdx = bestFeature,
                          threshold = bestThreshold,
                          leftTree = buildTree isRegression leftFeatures leftTargets (maxDepth - 1) minSamples,
                          rightTree = buildTree isRegression rightFeatures rightTargets (maxDepth - 1) minSamples
                        }

predictOne :: DecisionTree -> [Double] -> Double
predictOne (LeafC cls) _ = fromIntegral cls
predictOne (LeafR val) _ = val
predictOne node features =
  if features !! featureIdx node <= threshold node
    then predictOne (leftTree node) features
    else predictOne (rightTree node) features

predict :: DecisionTree -> [[Double]] -> [Double]
predict tree = map (predictOne tree)

getDepth :: DecisionTree -> Int
getDepth (LeafC _) = 0
getDepth (LeafR _) = 0
getDepth node = 1 + max (getDepth (leftTree node)) (getDepth (rightTree node))

getNLeaves :: DecisionTree -> Int
getNLeaves (LeafC _) = 1
getNLeaves (LeafR _) = 1
getNLeaves node = getNLeaves (leftTree node) + getNLeaves (rightTree node)

fit :: Bool -> [[Double]] -> [Double] -> Int -> Int -> DecisionTree
fit = buildTree
