module HaskLearn.Preprocessing where

import Data.List (sortBy)
import Data.Ord (comparing)
import System.Random (RandomGen, randoms)

trainTestSplit :: (RandomGen g) => g -> [a] -> [b] -> Double -> ([a], [a], [b], [b])
trainTestSplit gen x y testSize
  | length x /= length y = error "Features and target must have the same number of samples"
  | otherwise =
      let dataset = zip x y
          n = length dataset
          nTest = floor $ fromIntegral n * testSize
          rs = randoms gen :: [Int]
          shuffled = map snd $ sortBy (comparing fst) $ zip rs dataset
          (testSet, trainSet) = splitAt nTest shuffled

          (xTest, yTest) = unzip testSet
          (xTrain, yTrain) = unzip trainSet
       in (xTrain, xTest, yTrain, yTest)
