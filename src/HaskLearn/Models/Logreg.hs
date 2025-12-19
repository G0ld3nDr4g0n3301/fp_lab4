{-# OPTIONS_GHC -Wno-x-partial #-}

module HaskLearn.Models.Logreg
  ( LogisticModel (..),
    fit,
    predict,
    predictProba,
    logLoss,
  )
where

import System.Random (RandomGen, randomRs, split)

dot :: [Double] -> [Double] -> Double
dot xs ys = sum $ zipWith (*) xs ys

transpose :: [[a]] -> [[a]]
transpose [] = []
transpose ([] : _) = []
transpose xs = map head xs : transpose (map tail xs)

addInterceptColumn :: [[Double]] -> [[Double]]
addInterceptColumn = map (1 :)

sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

newtype LogisticModel = LogisticModel [Double] deriving (Show)

predictProbaOne :: LogisticModel -> [Double] -> Double
predictProbaOne (LogisticModel weights) features =
  sigmoid $ dot weights (1 : features)

predictProba :: LogisticModel -> [[Double]] -> [Double]
predictProba model = map (predictProbaOne model)

predictOne :: LogisticModel -> [Double] -> Int
predictOne model features =
  let prob = predictProbaOne model features
   in if prob >= 0.5 then 1 else 0

predict :: LogisticModel -> [[Double]] -> [Int]
predict model = map (predictOne model)

logLoss :: LogisticModel -> [[Double]] -> [Int] -> Double
logLoss model xs ys =
  let probs = predictProba model xs
      n = fromIntegral $ length ys
      losses =
        zipWith3
          ( \y pTrue pFalse ->
              if y == 1
                then -log (max 1e-15 pTrue)
                else -log (max 1e-15 pFalse)
          )
          ys
          probs
          (map (1 -) probs)
   in sum losses / n

computeGradient :: [[Double]] -> [Int] -> [Double] -> [Double]
computeGradient xWithIntercept y probs =
  let yDouble = map fromIntegral y
      errors = zipWith (-) probs yDouble
      m = fromIntegral $ length y
      gradients =
        map
          ( \featureCol ->
              (1 / m) * sum (zipWith (*) errors featureCol)
          )
          (transpose xWithIntercept)
   in gradients

gradientDescentStep :: Double -> [Double] -> [[Double]] -> [Int] -> [Double]
gradientDescentStep learningRate weights xWithIntercept y =
  let probs = map (sigmoid . dot weights) xWithIntercept
      gradient = computeGradient xWithIntercept y probs
   in zipWith (-) weights (map (* learningRate) gradient)

fit :: (RandomGen g) => [[Double]] -> [Int] -> Double -> Int -> g -> (LogisticModel, g)
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
   in (LogisticModel finalWeights, g')
  where
    initializeWeights :: (RandomGen g'') => Int -> g'' -> ([Double], g'')
    initializeWeights nFeatures g'' =
      let (g1, g2) = split g''
          weights = take nFeatures $ randomRs (-0.01, 0.01) g1
       in (weights, g2)
