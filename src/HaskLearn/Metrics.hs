module HaskLearn.Metrics where

meanAbsoluteError :: [Double] -> [Double] -> Double
meanAbsoluteError yTrue yPred = 
    sum (zipWith (\yt yp -> abs (yt - yp)) yTrue yPred) / fromIntegral (length yTrue)

meanSquaredError :: [Double] -> [Double] -> Double
meanSquaredError yTrue yPred = 
    sum (zipWith (\yt yp -> (yt - yp) ^ (2 :: Int)) yTrue yPred) / fromIntegral (length yTrue)

rmse :: [Double] -> [Double] -> Double
rmse yTrue yPred = sqrt (meanSquaredError yTrue yPred)

r2Score :: [Double] -> [Double] -> Double
r2Score yTrue yPred = 
    let yMean = sum yTrue / fromIntegral (length yTrue)
        ssTotal = sum $ map (\y -> (y - yMean) ^ (2 :: Int)) yTrue
        ssRes   = sum $ zipWith (\yt yp -> (yt - yp) ^ (2 :: Int)) yTrue yPred
    in 1 - (ssRes / ssTotal)

accuracyScore :: (Eq a) => [a] -> [a] -> Double
accuracyScore yTrue yPred = 
    let correct = length $ filter id $ zipWith (==) yTrue yPred
    in fromIntegral correct / fromIntegral (length yTrue)

type ConfusionMatrix = (Int, Int, Int, Int) 

getConfusionMatrix :: (Eq a) => a -> [a] -> [a] -> ConfusionMatrix
getConfusionMatrix posLabel yTrue yPred = foldl' count (0,0,0,0) (zip yTrue yPred)
  where
    count (tp, tn, fp, fn) (yt, yp)
        | yt == posLabel && yp == posLabel = (tp + 1, tn, fp, fn)
        | yt /= posLabel && yp /= posLabel = (tp, tn + 1, fp, fn)
        | yt /= posLabel && yp == posLabel = (tp, tn, fp + 1, fn)
        | yt == posLabel && yp /= posLabel = (tp, tn, fp, fn + 1)
        | otherwise = (tp, tn, fp, fn)

precisionScore :: (Eq a) => a -> [a] -> [a] -> Double
precisionScore posLabel yTrue yPred = 
    let (tp, _, fp, _) = getConfusionMatrix posLabel yTrue yPred
    in if (tp + fp) == 0 then 0.0 else fromIntegral tp / fromIntegral (tp + fp)

recallScore :: (Eq a) => a -> [a] -> [a] -> Double
recallScore posLabel yTrue yPred = 
    let (tp, _, _, fn) = getConfusionMatrix posLabel yTrue yPred
    in if (tp + fn) == 0 then 0.0 else fromIntegral tp / fromIntegral (tp + fn)

f1Score :: (Eq a) => a -> [a] -> [a] -> Double
f1Score posLabel yTrue yPred = 
    let p = precisionScore posLabel yTrue yPred
        r = recallScore posLabel yTrue yPred
    in if (p + r) == 0 then 0.0 else 2 * (p * r) / (p + r)