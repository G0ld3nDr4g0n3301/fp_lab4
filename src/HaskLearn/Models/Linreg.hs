{-# OPTIONS_GHC -Wno-x-partial #-}

module HaskLearn.Models.Linreg
  ( LinearModel(..)
  , fit
  , predict
  ) where

import System.Random (RandomGen, randomRs, split)

dot :: [Double] -> [Double] -> Double
dot xs ys = sum $ zipWith (*) xs ys

transpose :: [[a]] -> [[a]]
transpose [] = []
transpose ([]:_) = []
transpose xs = map head xs : transpose (map tail xs)

addInterceptColumn :: [[Double]] -> [[Double]]
addInterceptColumn = map (1:)

newtype LinearModel = LinearModel [Double] deriving (Show)

predictOne :: LinearModel -> [Double] -> Double
predictOne (LinearModel weights) features =
  dot weights (1:features)

predict :: LinearModel -> [[Double]] -> [Double]
predict model = map (predictOne model)

computeGradient :: [[Double]] -> [Double] -> [Double] -> [Double]
computeGradient xWithIntercept y predictions =
  let errors = zipWith (-) predictions y
      m = fromIntegral $ length y
      gradients = map (\featureCol ->
        (2/m) * sum (zipWith (*) errors featureCol)
        ) (transpose xWithIntercept)
  in gradients

gradientDescentStep :: Double -> [Double] -> [[Double]] -> [Double] -> [Double]
gradientDescentStep learningRate weights xWithIntercept y =
  let predictions = map (dot weights) xWithIntercept
      gradient = computeGradient xWithIntercept y predictions
  in zipWith (-) weights (map (* learningRate) gradient)

fit :: (RandomGen g) => [[Double]] -> [Double] -> Double -> Int -> g -> (LinearModel, g)
fit x y learningRate iterations g =
  let nFeatures = length (head x)
      (initialWeights, g') = initializeWeights (nFeatures + 1) g
      xWithIntercept = addInterceptColumn x
      
      iterateWeights :: [Double] -> Int -> [Double]
      iterateWeights weights iter
        | iter <= 0 = weights
        | otherwise = 
            let newWeights = gradientDescentStep learningRate weights xWithIntercept y
            in iterateWeights newWeights (iter - 1)
      
      finalWeights = iterateWeights initialWeights iterations
  in (LinearModel finalWeights, g')
  where
    initializeWeights :: (RandomGen g'') => Int -> g'' -> ([Double], g'')
    initializeWeights nFeatures g'' =
      let (g1, g2) = split g''
          weights = take nFeatures $ randomRs (-0.01, 0.01) g1
      in (weights, g2)